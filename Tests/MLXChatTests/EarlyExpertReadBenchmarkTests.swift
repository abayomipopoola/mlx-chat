import Foundation
import Metal
import Testing
import TurboFieldfare
@testable import MLXChat

/// Live A/B of the next-layer early expert read on the Gemma/Qwen runner:
/// interleaved off/on greedy runs in one process, decode tok/s for each, and
/// a check that the generated tokens are identical. Opt-in
/// (`MLXCHAT_RUN_EARLY_READ_BENCH=1`); `MLXCHAT_EARLY_READ_MODELS` picks
/// variants (qat, ornith15, standard, qwen36), default all installed.
@Suite(.serialized) struct EarlyExpertReadBenchmarkTests {
    private static let environment = ProcessInfo.processInfo.environment
    private static let variants: [String: StreamedMoEVariant] = [
        "qat": StreamedMoEInstall.qat, "ornith15": StreamedMoEInstall.ornith15,
        "standard": StreamedMoEInstall.standard, "qwen36": StreamedMoEInstall.qwen36,
    ]

    @Test(.timeLimit(.minutes(60)))
    func earlyReadKeepsTokensAndSpeedsDecode() async throws {
        guard Self.environment["MLXCHAT_RUN_EARLY_READ_BENCH"] == "1" else { return }
        let keys = (Self.environment["MLXCHAT_EARLY_READ_MODELS"] ?? "qat,ornith15,standard,qwen36")
            .split(separator: ",").map(String.init)
        // A cold warm-up run, then counterbalanced pairs (off on, on off, ...)
        // so drift from cache warm-up or heat lands on both sides.
        let pairs = Int(Self.environment["MLXCHAT_EARLY_READ_PAIRS"] ?? "") ?? 3
        let order: [RuntimeEarlyExpertRead] = [.off] + (0..<pairs).flatMap {
            $0.isMultiple(of: 2) ? [RuntimeEarlyExpertRead.off, .router] : [.router, .off]
        }
        for key in keys {
            guard let variant = Self.variants[key],
                  FileManager.default.fileExists(atPath: variant.managedDirectory
                    .appendingPathComponent("manifest.json").path)
            else { continue }
            let directory = variant.managedDirectory
            let tokenizer = try await GFTokenizer.load(forModelDirectory: directory)
            let context = try MetalContext()
            let model = try Model.load(directoryURL: directory, device: context.device,
                                       streamingMode: .pread(slotCount: 16),
                                       integrityPolicy: .sizeCheckTrustedReceipt)
            let scratch = try RawCompletionScratch(
                context: context, vocab: model.config.vocabSize,
                logitSoftcap: Float(model.config.finalLogitSoftcap))
            let prompt = try StreamedMoEPrompt.render(
                tokenizer: tokenizer,
                history: [EngineMessage(role: .user,
                                        text: "Explain in a short paragraph how a bicycle gear works.")],
                systemPrompt: PromptBuilder.systemPrompt(),
                thinkingEnabled: false)
            let ids = tokenizer.encode(prompt, addBOS: false)
            var rates: [RuntimeEarlyExpertRead: [Double]] = [:]
            var tokens: [RuntimeEarlyExpertRead: [Int32]] = [:]
            var precision: Double?
            for (runIndex, mode) in order.enumerated() {
                let runner = try RealForwardRunner(
                    model: model, context: context, maxContext: 4_096,
                    runtimeConfiguration: RuntimeConfiguration(
                        expertCacheSlots: 16, prefillEnabled: true, prefillChunkTokens: 128,
                        forceLogitsHead: true, earlyExpertRead: mode))
                let result = try await runRawCompletion(
                    producer: runner, tokenizer: tokenizer, promptIds: ids,
                    config: GenerationConfig(maxNewTokens: 128, temperature: 0),
                    context: context, scratch: scratch) { _ in }
                if runIndex > 0 {
                    rates[mode, default: []].append(Double(result.newTokens) / result.decodeSeconds)
                }
                let generated = Array(result.kvBackedTokenIDs.suffix(result.newTokens))
                if let earlier = tokens[mode] {
                    #expect(earlier == generated, "\(key) \(mode) runs differ")
                }
                tokens[mode] = generated
                if mode == .router { precision = runner.earlyExpertReadStats?.precision }
            }
            func summary(_ values: [Double]) -> String {
                let sorted = values.sorted()
                let median = sorted.isEmpty ? 0 : sorted[sorted.count / 2]
                return String(format: "median %.2f [", median)
                    + values.map { String(format: "%.1f", $0) }.joined(separator: " ") + "]"
            }
            print("""
                EARLY_READ \(key): off \(summary(rates[.off] ?? [])) | on \(summary(rates[.router] ?? [])) \
                | precision \(precision.map { String(format: "%.2f", $0) } ?? "-")
                """)
            #expect(tokens[.off] == tokens[.router], "\(key) early read changed the output")
        }
    }

    /// The same A/B for LFM2.5's next-layer prefetch (`prefetchesNextLayer`).
    @Test(.timeLimit(.minutes(30)))
    func lfm2PrefetchKeepsTokensAndSpeedsDecode() async throws {
        let variant = StreamedMoEInstall.lfm25
        guard Self.environment["MLXCHAT_RUN_EARLY_READ_BENCH"] == "1",
              FileManager.default.fileExists(atPath: variant.managedDirectory
                .appendingPathComponent("manifest.json").path)
        else { return }
        let directory = variant.managedDirectory
        let tokenizer = try await GFTokenizer.load(forModelDirectory: directory)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
                                   streamingMode: .pread(slotCount: 16),
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let scratch = try RawCompletionScratch(
            context: context, vocab: model.config.vocabSize,
            logitSoftcap: Float(model.config.finalLogitSoftcap))
        let prompt = StreamedMoEPrompt.renderChatMLThinkingPrompt(
            history: [EngineMessage(role: .user,
                                    text: "Explain in a short paragraph how a bicycle gear works.")],
            systemPrompt: PromptBuilder.systemPrompt(), family: .lfm)
        let ids = tokenizer.encode(prompt, addBOS: false)
        let pairs = Int(Self.environment["MLXCHAT_EARLY_READ_PAIRS"] ?? "") ?? 3
        let order: [Bool] = [false] + (0..<pairs).flatMap {
            $0.isMultiple(of: 2) ? [false, true] : [true, false]
        }
        var rates: [Bool: [Double]] = [:]
        var tokens: [Bool: [Int32]] = [:]
        for (runIndex, prefetch) in order.enumerated() {
            let runner = try LFM2MoEForwardRunner(model: model, context: context, maxContext: 4_096)
            runner.prefetchesNextLayer = prefetch
            let result = try await runRawCompletion(
                producer: runner, tokenizer: tokenizer, promptIds: ids,
                config: GenerationConfig(maxNewTokens: 128, temperature: 0),
                context: context, scratch: scratch) { _ in }
            if runIndex > 0 {
                rates[prefetch, default: []].append(Double(result.newTokens) / result.decodeSeconds)
            }
            let generated = Array(result.kvBackedTokenIDs.suffix(result.newTokens))
            if let earlier = tokens[prefetch] { #expect(earlier == generated) }
            tokens[prefetch] = generated
        }
        func summary(_ values: [Double]) -> String {
            let sorted = values.sorted()
            let median = sorted.isEmpty ? 0 : sorted[sorted.count / 2]
            return String(format: "median %.2f [", median)
                + values.map { String(format: "%.1f", $0) }.joined(separator: " ") + "]"
        }
        print("EARLY_READ lfm25 prefetch: off \(summary(rates[false] ?? [])) | on \(summary(rates[true] ?? []))")
        #expect(tokens[false] == tokens[true], "LFM2 prefetch changed the output")
    }
}
