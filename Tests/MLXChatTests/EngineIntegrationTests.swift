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
