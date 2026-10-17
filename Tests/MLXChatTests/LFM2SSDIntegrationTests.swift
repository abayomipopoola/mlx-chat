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
        try loadWeights(modelDirectory: mlxDirectory, model: lm,
                        perLayerQuantization: base.perLayerQuantization)
        let cache = lm.newCache(parameters: nil)
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
            case .completed: break
            }
        }
        print("LFM2 SSD app answer: \(content.prefix(80)) | thought \(thinking.count) chars")
        #expect(content.contains("42"), "answered: \(content.prefix(200))")
    }
}
