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
        #expect(await store.isDownloaded(modelID))

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
}
