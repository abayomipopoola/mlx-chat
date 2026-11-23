import Foundation
import Metal
import MLX
import MLXLLM
import MLXLMCommon
import Testing
import TurboFieldfare
@testable import MLXChat

/// Live check of the LFM2.5 SSD runner against the in-memory MLX model on the
/// same 4-bit checkpoint. Opt-in (`MLXCHAT_RUN_LFM_SSD_SMOKE=1`) because it
/// needs both the repacked `.gturbo` (13.6 GB) and the MLX copy (13.6 GB).
@Suite(.serialized) struct LFM2SSDIntegrationTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXCHAT_RUN_LFM_SSD_SMOKE"] == "1"
    }

    private static let ssdDirectory = StreamedMoEInstall.lfm25.managedDirectory
    private static var mlxDirectory: URL? {
        ModelStore.localMLXModelDirectory(for: StreamedMoEInstall.lfm25.sourceRepoID)
    }

    private static func promptIDs(_ tokenizer: GFTokenizer, _ question: String) -> [Int32] {
        let prompt = StreamedMoEPrompt.renderChatMLThinkingPrompt(
            history: [EngineMessage(role: .user, text: question)],
            systemPrompt: "You are a helpful assistant.",
            family: .lfm)
        return tokenizer.encode(prompt, addBOS: false)
    }

    private static func ranked(_ logits: [Float], top: Int) -> [Int] {
        Array(logits.indices.sorted { logits[$0] > logits[$1] }.prefix(top))
    }

    @Test(.timeLimit(.minutes(30)))
    func ssdRunnerMatchesMLXOnNextTokenAndGreedyContinuation() async throws {
        guard Self.enabled, let mlxDirectory = Self.mlxDirectory,
              FileManager.default.fileExists(
                atPath: Self.ssdDirectory.appendingPathComponent("manifest.json").path)
        else { return }

        let tokenizer = try await GFTokenizer.load(forModelDirectory: Self.ssdDirectory)
        #expect(tokenizer.chatMLFamily == .lfm)
        let ids = Self.promptIDs(tokenizer, "What is 17 plus 25? Answer with the number only.")
        #expect(ids.first == tokenizer.bosID)
        let steps = 16

        // SSD runner, greedy.
        let context = try MetalContext()
        let model = try Model.load(directoryURL: Self.ssdDirectory, device: context.device,
                                   streamingMode: .pread(slotCount: 16),
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        #expect(model.config.family == .lfm2moe)
        let runner = try LFM2MoEForwardRunner(model: model, context: context, maxContext: 4_096)
        let logitsBuffer = try #require(context.device.makeBuffer(
            length: model.config.vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))
        func readLogits() -> [Float] {
            let p = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
            return (0..<model.config.vocabSize).map { Float(p[$0]) }
        }
        let started = Date()
        for (position, token) in ids.enumerated() {
            try await runner.produce(token: token, position: position, into: logitsBuffer)
        }
        let prefillSeconds = Date().timeIntervalSince(started)
        let ssdFirst = readLogits()
        var ssdTokens = [Self.ranked(ssdFirst, top: 1)[0]]
        let decodeStart = Date()
        for step in 1..<steps {
            try await runner.produce(token: Int32(ssdTokens[step - 1]),
                                     position: ids.count + step - 1, into: logitsBuffer)
            ssdTokens.append(Self.ranked(readLogits(), top: 1)[0])
        }
        let decodeTPS = Double(steps - 1) / Date().timeIntervalSince(decodeStart)

        // Rewind to the last prompt token and run it again: the conv state is
        // rebuilt by replay, so the logits must match the first pass.
        try runner.prepareForContinuation(expectedPosition: ids.count - 1)
        try await runner.produce(token: ids.last!, position: ids.count - 1, into: logitsBuffer)
        let rewound = readLogits()
        let maxDrift = zip(rewound, ssdFirst).map { abs($0 - $1) }.max() ?? 0
        #expect(maxDrift < 1e-3, "rewound logits drifted by \(maxDrift)")
        #expect(runner.continuationPosition == ids.count)

        // MLX reference, greedy with a KV cache.
        let configData = try Data(contentsOf: mlxDirectory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        let lm = try await LLMTypeRegistry.shared.createModel(
            configuration: configData, modelType: base.modelType)
        try await loadWeights(modelDirectory: mlxDirectory, model: lm,
                              perLayerQuantization: base.perLayerQuantization)
        let cache = try lm.newCache(parameters: nil)
        func lastLogits(_ tokens: [Int32]) -> [Float] {
            let input = LMInput.Text(tokens: MLXArray(tokens).reshaped(1, tokens.count))
            let logits: MLXArray = lm(input, cache: cache, state: nil).logits
            let last: MLXArray = logits[0, tokens.count - 1]
            return last.asType(.float32).asArray(Float.self)
        }
        func argmax(_ row: [Float]) -> Int {
            row.indices.max { row[$0] < row[$1] }!
        }
        let mlxFirst = lastLogits(ids)
        var mlxTokens: [Int] = [argmax(mlxFirst)]
        for _ in 1..<steps {
            mlxTokens.append(argmax(lastLogits([Int32(mlxTokens.last!)])))
        }

        let ssdTop5 = Self.ranked(ssdFirst, top: 5)
        let mlxTop5 = Self.ranked(mlxFirst, top: 5)
        let overlap = Set(ssdTop5).intersection(mlxTop5).count
        let n = Float(ssdFirst.count)
        let meanA = ssdFirst.reduce(0, +) / n, meanB = mlxFirst.reduce(0, +) / n
        var cov: Float = 0, varA: Float = 0, varB: Float = 0
        for i in ssdFirst.indices {
            let a = ssdFirst[i] - meanA, b = mlxFirst[i] - meanB
            cov += a * b; varA += a * a; varB += b * b
        }
        let correlation = cov / (varA.squareRoot() * varB.squareRoot())
        let matching = zip(ssdTokens, mlxTokens).prefix { $0 == $1 }.count
        print("""
            LFM2 SSD vs MLX: top1 \(ssdTop5[0]) vs \(mlxTop5[0]), top5 overlap \(overlap)/5, \
            logit correlation \(correlation), greedy prefix match \(matching)/\(steps), \
            prompt \(ids.count) tokens in \(String(format: "%.1f", prefillSeconds)) s, \
            decode \(String(format: "%.2f", decodeTPS)) tok/s
            SSD text: \(tokenizer.decode(ssdTokens.map(Int32.init)))
            MLX text: \(tokenizer.decode(mlxTokens.map(Int32.init)))
            """)
        #expect(ssdTop5[0] == mlxTop5[0])
        #expect(overlap >= 4)
        // FP16 activations here vs BF16 in MLX drift ~0.3 per logit over 40 layers.
        #expect(correlation > 0.98)
        #expect(matching >= 8)
    }

    @Test(.timeLimit(.minutes(30)))
    func ssdEngineAnswersThroughTheAppPath() async throws {
        guard Self.enabled,
              FileManager.default.fileExists(
                atPath: Self.ssdDirectory.appendingPathComponent("manifest.json").path)
        else { return }
        let catalog = try #require(ModelCatalog.model(for: StreamedMoEInstall.lfm25.modelID))
        let engine = StreamedMoEEngine(model: catalog, modelDirectory: Self.ssdDirectory)
        try await engine.load(progress: { _ in })
        defer { engine.unload() }
        var content = ""
        var thinking = ""
        for try await event in engine.stream(
            history: [EngineMessage(role: .user,
                                    text: "What is 17 plus 25? Answer with the number only.")],
            systemPrompt: "You are a helpful assistant.",
            temperature: 0,
            conversationID: UUID()
        ) {
            switch event {
            case .delta(let delta): content += delta
            case .thinkingDelta(let delta): thinking += delta
            case .thinkingReclassified: thinking = ""
            case .prefill: break
            case .completed: break
            }
        }
        print("LFM2 SSD app answer: \(content.prefix(80)) | thought \(thinking.count) chars")
        #expect(content.contains("42"), "answered: \(content.prefix(200))")
    }

    /// Batched prefill against token-by-token `produce` on a long prompt:
    /// last-token logits and greedy continuation; chunk sizes agree exactly.
    @Test(.timeLimit(.minutes(30)))
    func batchedPrefillMatchesTokenByToken() async throws {
        guard Self.enabled,
              FileManager.default.fileExists(
                atPath: Self.ssdDirectory.appendingPathComponent("manifest.json").path)
        else { return }
        let tokenizer = try await GFTokenizer.load(forModelDirectory: Self.ssdDirectory)
        let prompt = StreamedMoEPrompt.renderChatMLThinkingPrompt(
            history: [EngineMessage(role: .user, text: "Summarise this text.\n\n"
                                    + FlintSSDBenchmarkTests.longText(words: 1_100))],
            systemPrompt: "You are a helpful assistant.", family: .lfm)
        let ids = tokenizer.encode(prompt, addBOS: false)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: Self.ssdDirectory, device: context.device,
                                   streamingMode: .pread(slotCount: 16),
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let vocab = model.config.vocabSize
        func makeScratch() throws -> RawCompletionScratch {
            try RawCompletionScratch(context: context, vocab: vocab,
                                     logitSoftcap: Float(model.config.finalLogitSoftcap))
        }
        func read(_ buffer: MTLBuffer) -> [Float] {
            let p = buffer.contents().assumingMemoryBound(to: Float16.self)
            return (0..<vocab).map { Float(p[$0]) }
        }
        func argmax(_ x: [Float]) -> Int32 { Int32(x.indices.max { x[$0] < x[$1] }!) }
        func greedy(_ runner: LFM2MoEForwardRunner, from start: [Float], steps: Int) async throws -> [Int32] {
            let logits = try #require(context.device.makeBuffer(length: vocab * 2, options: .storageModeShared))
            var tokens: [Int32] = []
            var current = argmax(start)
            for _ in 0..<steps {
                tokens.append(current)
                try await runner.produce(token: current, position: runner.continuationPosition,
                                         into: logits)
                current = argmax(read(logits))
            }
            return tokens
        }

        let reference = try LFM2MoEForwardRunner(model: model, context: context, maxContext: 4_096)
        let referenceLogits = try #require(context.device.makeBuffer(length: vocab * 2, options: .storageModeShared))
        let referenceStart = Date()
        for (position, token) in ids.enumerated() {
            try await reference.produce(token: token, position: position, into: referenceLogits)
        }
        let referenceSeconds = Date().timeIntervalSince(referenceStart)
        var batched: [Int: (runner: LFM2MoEForwardRunner, logits: [Float], seconds: Double)] = [:]
        let chunks = (ProcessInfo.processInfo.environment["MLXCHAT_LFM_PREFILL_CHUNKS"] ?? "2048,64,17")
            .split(separator: ",").compactMap { Int($0) }
        for chunk in chunks {
            let runner = try LFM2MoEForwardRunner(model: model, context: context, maxContext: 4_096)
            runner.prefillChunkTokens = chunk
            runner.usesTensorQMM = ProcessInfo.processInfo.environment["MLXCHAT_LFM_PREFILL_TENSOR"] != "0"
            if let exact = ProcessInfo.processInfo.environment["MLXCHAT_LFM_PREFILL_EXACT"] {
                runner.exactQMMKinds = Set(exact.split(separator: ",").map(String.init))
            }
            let scratch = try makeScratch()
            let started = Date()
            try await runRawPrefill(producer: runner, promptIds: ids, scratch: scratch)
            batched[chunk] = (runner, read(scratch.logitsBuffer), Date().timeIntervalSince(started))
        }
        let main = try #require(batched[2_048])
        var layerReport: [String] = []
        for layer in 0..<model.config.numLayers {
            var maxError: Float = 0, sum: Float = 0, n = 0
            let positions: [Int?] = model.config.layerIsShortConv(layer)
                ? [nil] : Array(max(0, ids.count - 400)..<ids.count)
            for position in positions {
                guard let x = reference.debugCacheRow(layer: layer, position: position),
                      let y = main.runner.debugCacheRow(layer: layer, position: position) else { continue }
                for i in x.indices {
                    let e = abs(Float(x[i]) - Float(y[i]))
                    maxError = max(maxError, e); sum += e; n += 1
                }
            }
            if n > 0 {
                layerReport.append("L\(layer)\(model.config.layerIsShortConv(layer) ? "c" : "a") \(String(format: "%.3f/%.4f", maxError, sum / Float(n)))")
            }
        }
        print("LFM_PREFILL_CHECK layers (max/mean): " + layerReport.joined(separator: ", "))
        let a = read(referenceLogits)
        let b = main.logits
        func top(_ x: [Float], _ k: Int) -> [Int] { Array(x.indices.sorted { x[$0] > x[$1] }.prefix(k)) }
        let overlap = Set(top(a, 5)).intersection(top(b, 5)).count
        let meanA = a.reduce(0, +) / Float(vocab), meanB = b.reduce(0, +) / Float(vocab)
        var cov: Float = 0, varA: Float = 0, varB: Float = 0
        for i in a.indices {
            cov += (a[i] - meanA) * (b[i] - meanB)
            varA += (a[i] - meanA) * (a[i] - meanA)
            varB += (b[i] - meanB) * (b[i] - meanB)
        }
        let correlation = cov / (varA.squareRoot() * varB.squareRoot())
        let referenceGreedy = try await greedy(reference, from: a, steps: 32)
        let batchedGreedy = try await greedy(main.runner, from: b, steps: 32)
        let matching = zip(referenceGreedy, batchedGreedy).prefix { $0 == $1 }.count
        print("""
            LFM_PREFILL_CHECK \(ids.count) tokens; token-by-token \(String(format: "%.1f", Double(ids.count) / referenceSeconds)) tok/s, \
            batched 2048 \(String(format: "%.1f", Double(ids.count) / main.seconds)) tok/s, \
            others \(batched.filter { $0.key != 2_048 }.map { "\($0.key): \(String(format: "%.1f", Double(ids.count) / $0.value.seconds))" }.joined(separator: " "))
            LFM_PREFILL_CHECK logits: top1 \(top(a, 1)) vs \(top(b, 1)), top5 overlap \(overlap)/5, correlation \(correlation), greedy match \(matching)/32, finite \(b.allSatisfy(\.isFinite))
            LFM_PREFILL_CHECK chunks equal: \(batched.filter { $0.key != 2_048 }.map { "\($0.key) \($0.value.logits == b)" }.joined(separator: ", "))
            LFM_PREFILL_CHECK reference text: \(tokenizer.decode(referenceGreedy).debugDescription)
            LFM_PREFILL_CHECK batched text:   \(tokenizer.decode(batchedGreedy).debugDescription)
            """)
        #expect(top(a, 1) == top(b, 1))
        #expect(overlap >= 4)
        #expect(correlation > 0.99)
        for (chunk, run) in batched where chunk != 2_048 {
            #expect(run.logits == b, "chunk \(chunk) differs")
        }
    }
}
