import AppKit
import Foundation
import Testing
@testable import MLXChat

/// Live VLM smoke: loads a local VLM checkpoint and answers about a solid red image.
/// Skips (stays green) when weights are not on disk; load/stream failures fail the suite.
@Suite(.serialized)
struct VLMIntegrationTests {
    @Test(.timeLimit(.minutes(20)))
    func vlmAnswersAboutImage() async throws {
        let modelID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
        guard ModelStore.localModelDirectory(for: modelID) != nil else {
            print("Skipping VLMIntegrationTests: weights not present for \(modelID)")
            return
        }

        let jpeg = try #require(Self.solidRedJPEG())

        let model = try #require(ModelCatalog.model(for: modelID))
        #expect(model.supportsVision)

        let engine = MLXEngine(model: model)
        try await engine.load(progress: { _ in })

        var content = ""
        var sawDelta = false
        var sawCompleted = false

        let history = [
            EngineMessage(
                role: .user,
                text: "What color is the square in this image? Answer with a single word.",
                images: [jpeg])
        ]
        for try await event in engine.stream(
            history: history,
            systemPrompt: "You are a helpful assistant.",
            temperature: 0,
            conversationID: UUID()
        ) {
            switch event {
            case .delta(let delta):
                content += delta
                sawDelta = true
            case .thinkingDelta, .thinkingReclassified:
                break
            case .completed:
                sawCompleted = true
            }
        }
        engine.unload()

        #expect(sawDelta, "expected at least one .delta event")
        #expect(sawCompleted, "expected a .completed event")
        #expect(
            content.lowercased().contains("red"),
            "reply did not mention red: \(content.prefix(200))")
    }

    @Test(.timeLimit(.minutes(20)))
    func vlmGemma4AnswersAboutImage() async throws {
        let modelID = "mlx-community/gemma-4-12B-it-4bit"
        guard ModelStore.localModelDirectory(for: modelID) != nil else {
            print("Skipping VLMIntegrationTests: weights not present for \(modelID)")
            return
        }

        let jpeg = try #require(Self.redCircleOnWhiteJPEG())

        let model = try #require(ModelCatalog.model(for: modelID))
        #expect(model.supportsVision)

        let engine = MLXEngine(model: model)
        try await engine.load(progress: { _ in })

        var content = ""
        var sawDelta = false
        var sawCompleted = false

        let history = [
            EngineMessage(
                role: .user,
                text: "What color is the circle in this image? Answer with a single word.",
                images: [jpeg])
        ]
        for try await event in engine.stream(
            history: history,
            systemPrompt: "You are a helpful assistant.",
            temperature: 0,
            conversationID: UUID()
        ) {
            switch event {
            case .delta(let delta):
                content += delta
                sawDelta = true
            case .thinkingDelta, .thinkingReclassified:
                break
            case .completed:
                sawCompleted = true
            }
        }
        engine.unload()

        #expect(sawDelta, "expected at least one .delta event")
        #expect(sawCompleted, "expected a .completed event")
        #expect(
            content.lowercased().contains("red"),
            "reply did not mention red: \(content.prefix(200))")
    }

    @Test(.timeLimit(.minutes(20)))
    func vlmGemma4TextOnlyTurn() async throws {
        let modelID = "mlx-community/gemma-4-12B-it-4bit"
        guard ModelStore.localModelDirectory(for: modelID) != nil else {
            print("Skipping VLMIntegrationTests: weights not present for \(modelID)")
            return
        }

        let model = try #require(ModelCatalog.model(for: modelID))
        #expect(model.supportsVision)

        let engine = MLXEngine(model: model)
        try await engine.load(progress: { _ in })

        var content = ""
        var sawDelta = false
        var sawCompleted = false

        let history = [
            EngineMessage(
                role: .user,
                text: "What is 2+2? Answer with a single number.")
        ]
        for try await event in engine.stream(
            history: history,
            systemPrompt: "You are a helpful assistant.",
            temperature: 0,
            conversationID: UUID()
        ) {
            switch event {
            case .delta(let delta):
                content += delta
                sawDelta = true
            case .thinkingDelta, .thinkingReclassified:
                break
            case .completed:
                sawCompleted = true
            }
        }
        engine.unload()

        #expect(sawDelta, "expected at least one .delta event")
        #expect(sawCompleted, "expected a .completed event")
        #expect(
            content.contains("4"),
            "reply did not mention 4: \(content.prefix(200))")
    }

    /// Solid red 512×512 JPEG for VLM color-recognition smoke tests.
    private static func solidRedJPEG() -> Data? {
        let size = NSSize(width: 512, height: 512)
        let image = NSImage(size: size)
        image.lockFocus()
        NSColor.systemRed.setFill()
        NSRect(origin: .zero, size: size).fill()
        image.unlockFocus()
        return AttachmentImage.jpegData(from: image)
    }

    /// Red circle on white for the Gemma probe. gemma4_unified's 4-bit vision
    /// embedder LayerNorms each patch, so a uniform color field collapses to a
    /// constant (pure red misreads as "Black" even in reference mlx-vlm with
    /// correct pixels — verified upstream, mlx-swift-lm 3.31.4, no fix yet).
    /// Structured images work (live-verified in-app), so the Gemma probe uses
    /// one; the assertion bar (reply must say "red") is unchanged.
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
}
