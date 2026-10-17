import AppKit
import Foundation
import Testing
@testable import MLXChat

/// End-to-end smoke harness (network + Metal): downloads the tiny catalog model,
/// loads it via the real factory path, and streams a reply. Proves the
/// download → load → ChatSession → think-filter pipeline without the GUI.
@Suite(.serialized)
struct EngineIntegrationTests {
    @Test(.timeLimit(.minutes(10)))
    func downloadsLoadsAndStreamsTinyModel() async throws {
        let modelID = "mlx-community/Qwen3-0.6B-4bit"
        guard let catalogModel = ModelCatalog.model(for: modelID) else {
            Issue.record("Tiny model missing from catalog")
            return
        }

        let store = await ModelStore()
        try await store.downloadAndWait(modelID)
        #expect(await store.isAvailable(modelID))

        let engine = MLXEngine(model: catalogModel)
        try await engine.load { _ in }

        var content = ""
        var thinking = ""
        var tokensPerSecond: Double?

        let history = [EngineMessage(role: .user, text: "Reply with a short greeting.")]
        for try await event in engine.stream(
            history: history,
            systemPrompt: "You are a concise assistant.",
            temperature: 0.1,
            conversationID: UUID()
        ) {
            switch event {
            case .delta(let delta): content += delta
            case .thinkingDelta(let delta): thinking += delta
            case .thinkingReclassified: thinking = ""
            case .completed(let tps): tokensPerSecond = tps
            }
        }
        engine.unload()

        let visible = content.trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(!visible.isEmpty, "model produced no visible content (thinking: \(thinking.prefix(120)))")
        #expect((tokensPerSecond ?? 0) > 0, "no completion info received")
    }

    /// Opt-in because this downloads about 13 GB across the five official
    /// precisions, then also verifies the community BF16 conversion. Run with
    /// `MLXCHAT_RUN_LFM_SMOKE=1` before shipping changes to LFM support.
    @Test(.timeLimit(.minutes(90)))
    func downloadsLoadsAndStreamsEverySupportedLFM25Variant() async throws {
        guard ProcessInfo.processInfo.environment["MLXCHAT_RUN_LFM_SMOKE"] == "1" else {
            return
        }

        let official = ModelCatalog.models.filter {
            $0.huggingFaceRepoID == "LiquidAI/LFM2.5-2.6B-MLX"
                && $0.installRestriction == nil
        }
        let community = try #require(
            ModelCatalog.model(for: "mlx-community/LFM2.5-2.6B-bf16"))

        #expect(official.count == 5)
        let allModels = official + [community]
        let requestedIDs = ProcessInfo.processInfo.environment["MLXCHAT_LFM_MODEL_IDS"]?
            .split(separator: ",")
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
        let models = requestedIDs.map { ids in
            allModels.filter { ids.contains($0.id) }
        } ?? allModels
        #expect(!models.isEmpty, "No requested LFM2.5 model ids matched the catalog")
        for model in models {
            try await verifyModelGenerates(model)
        }
    }

    /// Opt-in because this downloads the 3.46 GB Bonsai 2 9B pack. Run with
    /// `MLXCHAT_RUN_BONSAI_SMOKE=1` before shipping changes to the Prism Hadamard loader.
    /// Unrotated activations still decode, but into nonsense, so the bar is a correct answer.
    @Test(.timeLimit(.minutes(60)))
    func downloadsLoadsAndAnswersWithBonsai2Ternary() async throws {
        guard ProcessInfo.processInfo.environment["MLXCHAT_RUN_BONSAI_SMOKE"] == "1" else {
            return
        }
        let model = try #require(
            ModelCatalog.model(for: "prism-ml/Ternary-Bonsai-2-9B-mlx-2bit"))
        let store = await ModelStore()
        try await store.downloadAndWait(model.id)
        let directory = try #require(ModelStore.localMLXModelDirectory(for: model.id))
        #expect(!FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("mtp/model.safetensors").path))

        let defaults = UserDefaults.standard
        let savedThinking = defaults.object(forKey: Keys.thinkingEnabled)
        defaults.set(false, forKey: Keys.thinkingEnabled)
        defer { defaults.set(savedThinking, forKey: Keys.thinkingEnabled) }

        let engine = MLXEngine(model: model)
        try await engine.load { _ in }
        defer { engine.unload() }

        let sum = try await reply(
            from: engine, to: "What is 17 plus 25? Answer with the number only.")
        #expect(sum.contains("42"), "Bonsai 2 answered: \(sum.prefix(200))")

        let circle = try #require(Self.redCircleOnWhiteJPEG())
        let color = try await reply(
            from: engine,
            to: "What color is the circle in this image? Answer with a single word.",
            images: [circle])
        #expect(color.lowercased().contains("red"), "Bonsai 2 answered: \(color.prefix(200))")
        print("Bonsai 2 smoke passed: \(sum.prefix(40)) | \(color.prefix(40))")
    }

    private func reply(
        from engine: MLXEngine, to prompt: String, images: [Data] = []
    ) async throws -> String {
        var content = ""
        for try await event in engine.stream(
            history: [EngineMessage(role: .user, text: prompt, images: images)],
            systemPrompt: "You are a helpful assistant",
            temperature: 0,
            conversationID: UUID()
        ) {
            if case .delta(let delta) = event { content += delta }
        }
        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func redCircleOnWhiteJPEG() -> Data? {
        let size = NSSize(width: 512, height: 512)
        let image = NSImage(size: size)
        image.lockFocus()
        NSColor.white.setFill()
        NSRect(origin: .zero, size: size).fill()
        NSColor.systemRed.setFill()
        NSBezierPath(ovalIn: NSRect(x: 128, y: 128, width: 256, height: 256)).fill()
        image.unlockFocus()
        return AttachmentImage.jpegData(from: image)
    }

    private func verifyModelGenerates(_ model: CatalogModel) async throws {
        let store = await ModelStore()
        try await store.downloadAndWait(model.id)
        #expect(await store.isAvailable(model.id), "\(model.id) did not download")

        let engine = MLXEngine(model: model)
        try await engine.load { _ in }

        var content = ""
        var thinking = ""
        var tokensPerSecond: Double?
        let history = [EngineMessage(
            role: .user,
            text: "Reply with the single word OK and no punctuation.")]
        for try await event in engine.stream(
            history: history,
            systemPrompt: "Be concise.",
            temperature: 0.1,
            conversationID: UUID()
        ) {
            switch event {
            case .delta(let delta): content += delta
            case .thinkingDelta(let delta): thinking += delta
            case .thinkingReclassified: thinking = ""
            case .completed(let tps): tokensPerSecond = tps
            }
        }
        engine.unload()

        let visible = content.trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(
            !visible.isEmpty,
            "\(model.id) produced no visible response (thinking: \(thinking.prefix(120)))")
        #expect((tokensPerSecond ?? 0) > 0, "\(model.id) produced no completion metrics")
        print("LFM smoke passed: \(model.id) → \(visible.prefix(80))")
    }
}
