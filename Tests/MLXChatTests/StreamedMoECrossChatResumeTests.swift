import Foundation
import Metal
import Testing
import TurboFieldfare
@testable import MLXChat

/// Live check that a rewindable runner resumed from another chat's cache, or
/// from a warmed system-prompt prefix, generates exactly what a fresh run of
/// the same prompt generates. Opt-in (`MLXCHAT_RUN_SSD_RESUME_CHECK=1`); needs
/// the Flint and LFM2.5 SSD installs.
@Suite(.serialized) struct StreamedMoECrossChatResumeTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXCHAT_RUN_SSD_RESUME_CHECK"] == "1"
    }

    private static func prompt(_ variant: StreamedMoEVariant, _ tokenizer: GFTokenizer,
                               _ question: String) throws -> [Int32] {
        let system = PromptBuilder.systemPrompt()
        let history = [EngineMessage(role: .user, text: question)]
        let rendered = variant.architecture == .flint
            ? GFTokenizer.flintPrompt(turns: [("user", question)], system: system)
            : try StreamedMoEPrompt.render(tokenizer: tokenizer, history: history,
                                           systemPrompt: system, thinkingEnabled: true)
        return tokenizer.encode(rendered, addBOS: false)
    }

    @Test(.timeLimit(.minutes(30)), arguments: ["flint", "lfm25"])
    func resumedChatMatchesAFreshRun(key: String) async throws {
        let variant = key == "flint" ? StreamedMoEInstall.flint : StreamedMoEInstall.lfm25
        guard Self.enabled,
              FileManager.default.fileExists(
                atPath: variant.managedDirectory.appendingPathComponent("manifest.json").path)
        else { return }
        let directory = variant.managedDirectory
        let tokenizer = try await GFTokenizer.load(forModelDirectory: directory)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
                                   streamingMode: .pread(slotCount: 16),
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let runner: any ContinuableLogitProducer = key == "flint"
            ? try FlintForwardRunner(model: model, context: context, maxContext: 4_096)
            : try LFM2MoEForwardRunner(model: model, context: context, maxContext: 4_096)
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))
        let config = GenerationConfig(maxNewTokens: 16, temperature: 0)

        func generate(_ ids: [Int32], start: RawCompletionStart) async throws -> RawDecodeResult {
            try await runRawCompletion(producer: runner, tokenizer: tokenizer, promptIds: ids,
                                       config: config, context: context, scratch: scratch,
                                       start: start) { _ in }
        }
        func generated(_ result: RawDecodeResult) -> [Int32] {
            Array(result.kvBackedTokenIDs.suffix(result.newTokens))
        }
        func shared(_ cached: [Int32], _ next: [Int32]) -> RawCompletionStart {
            let count = zip(cached, next).prefix { $0 == $1 }.count
            return .resume(cachedPromptTokens: min(count, next.count - 1))
        }

        let chatA = try Self.prompt(variant, tokenizer, "What is 17 plus 25?")
        let chatB = try Self.prompt(variant, tokenizer, "Name a primary colour.")
        let fresh = generated(try await generate(chatB, start: .reset))

        // Chat A runs to the end, then chat B resumes at the shared prefix.
        let afterA = try await generate(chatA, start: .reset)
        let resumed = generated(try await generate(chatB, start: shared(afterA.kvBackedTokenIDs, chatB)))

        // A warmed prefix (system prompt and user header) serves chat B too.
        let prefixCount = zip(chatA, chatB).prefix { $0 == $1 }.count
        try await runRawPrefill(producer: runner, promptIds: Array(chatA.prefix(prefixCount)),
                                scratch: scratch)
        let warmed = generated(try await generate(chatB, start: .resume(cachedPromptTokens: prefixCount)))

        print("RESUME_CHECK \(key): shared \(prefixCount) of \(chatB.count), fresh \(fresh), resumed \(resumed), warmed \(warmed)")
        #expect(prefixCount > 20)
        #expect(resumed == fresh)
        #expect(warmed == fresh)
    }

    /// Flint's sliding layers in a ring small enough to wrap many times give
    /// the same greedy tokens as full-length layers, and a rewind past the
    /// wrap still replays to the same logits.
    @Test(.timeLimit(.minutes(30)))
    func flintSlidingRingMatchesFullLengthKV() async throws {
        let variant = StreamedMoEInstall.flint
        guard Self.enabled,
              FileManager.default.fileExists(
                atPath: variant.managedDirectory.appendingPathComponent("manifest.json").path)
        else { return }
        let directory = variant.managedDirectory
        let tokenizer = try await GFTokenizer.load(forModelDirectory: directory)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
                                   streamingMode: .pread(slotCount: 16),
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))
        let ids = try Self.prompt(variant, tokenizer, "Write a long story about a lighthouse keeper.")
        let config = GenerationConfig(maxNewTokens: 720, temperature: 0)
        func run(_ runner: FlintForwardRunner) async throws -> [Int32] {
            let result = try await runRawCompletion(producer: runner, tokenizer: tokenizer,
                                                    promptIds: ids, config: config,
                                                    context: context, scratch: scratch) { _ in }
            return result.kvBackedTokenIDs
        }
        let full = try FlintForwardRunner(model: model, context: context, maxContext: 4_096,
                                          slidingRewindMargin: 8_192)
        #expect(full.maxRewindDistance == Int.max)
        let fullTokens = try await run(full)
        let ring = try FlintForwardRunner(model: model, context: context, maxContext: 4_096,
                                          slidingRewindMargin: 64)
        #expect(ring.maxRewindDistance == 64)
        let ringTokens = try await run(ring)
        #expect(ringTokens.count > 513 + 64 + 100, "the run must wrap the ring")

        // Rewind 48 positions (inside the margin) and replay them.
        let end = ring.continuationPosition
        try ring.prepareForContinuation(expectedPosition: end - 48)
        let logits = try #require(context.device.makeBuffer(
            length: model.config.vocabSize * 2, options: .storageModeShared))
        for position in (end - 48)..<end {
            try await ring.produce(token: ringTokens[position], position: position, into: logits)
        }
        let p = logits.contents().assumingMemoryBound(to: Float16.self)
        let replayed = (0..<model.config.vocabSize).max { p[$0] < p[$1] }.map(Int32.init)
        #expect(throws: (any Error).self) { try ring.prepareForContinuation(expectedPosition: end - 65) }
        print("RING_CHECK tokens \(ringTokens.count), equal \(ringTokens == fullTokens), replayed next \(replayed ?? -1)")
        #expect(ringTokens == fullTokens)
    }
}
