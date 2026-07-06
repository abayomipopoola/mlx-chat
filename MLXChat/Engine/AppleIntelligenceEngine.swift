import Foundation
import FoundationModels

/// Built-in Apple Intelligence engine (FoundationModels, macOS 26).
final class AppleIntelligenceEngine: ChatEngine {
    let id = appleIntelligenceEngineID

    /// nil when the model is ready to use; otherwise a user-facing reason.
    static func availabilityMessage() -> String? {
        switch SystemLanguageModel.default.availability {
        case .available:
            return nil
        case .unavailable(.deviceNotEligible):
            return "This Mac does not support Apple Intelligence."
        case .unavailable(.appleIntelligenceNotEnabled):
            return "Turn on Apple Intelligence in System Settings."
        case .unavailable(.modelNotReady):
            return "Apple Intelligence is still preparing. Try again shortly."
        case .unavailable(let other):
            return "Apple Intelligence is unavailable: \(other)"
        }
    }

    static var isAvailable: Bool { availabilityMessage() == nil }

    /// Token budget for history trimming.
    static var contextTokens: Int {
        if #available(macOS 26.4, *) {
            return SystemLanguageModel.default.contextSize
        }
        return 4096
    }

    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        if let message = Self.availabilityMessage() {
            throw EngineError.unavailable(message)
        }
        progress(1)
    }

    func unload() {}

    func stream(
        history: [EngineMessage],
        systemPrompt: String,
        temperature: Double,
        conversationID: UUID
    ) -> AsyncThrowingStream<EngineEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    if let message = Self.availabilityMessage() {
                        throw EngineError.unavailable(message)
                    }
                    let trimmed = PromptBuilder.trimmed(history: history, ctxTokens: Self.contextTokens)
                    guard let pending = trimmed.last, pending.role == .user else {
                        throw EngineError.generationFailed("No pending user message.")
                    }

                    var entries: [Transcript.Entry] = [
                        .instructions(.init(
                            segments: [.text(.init(content: systemPrompt))],
                            toolDefinitions: []))
                    ]
                    for message in trimmed.dropLast() {
                        let segment: Transcript.Segment = .text(.init(content: message.text))
                        switch message.role {
                        case .user:
                            entries.append(.prompt(.init(segments: [segment])))
                        case .assistant:
                            entries.append(.response(.init(assetIDs: [], segments: [segment])))
                        }
                    }

                    let session = LanguageModelSession(
                        model: .default, tools: [],
                        transcript: Transcript(entries: entries))
                    let options = GenerationOptions(
                        sampling: nil,
                        temperature: min(max(temperature, 0), 1),
                        maximumResponseTokens: nil)
                    let stream = session.streamResponse(to: Prompt(pending.text), options: options)

                    // FoundationModels yields CUMULATIVE snapshots, not deltas.
                    var emitted = 0
                    for try await snapshot in stream {
                        let text = snapshot.content
                        guard text.count > emitted else { continue }
                        continuation.yield(.delta(String(text.dropFirst(emitted))))
                        emitted = text.count
                    }
                    continuation.yield(.completed(tokensPerSecond: nil))
                    continuation.finish()
                } catch is CancellationError {
                    // Keep partial text; end quietly.
                    continuation.yield(.completed(tokensPerSecond: nil))
                    continuation.finish()
                } catch let error as LanguageModelSession.GenerationError {
                    switch error {
                    case .exceededContextWindowSize:
                        continuation.finish(throwing: EngineError.contextExceeded(
                            "This conversation exceeds Apple Intelligence's context window. Start a new chat."))
                    case .guardrailViolation:
                        continuation.finish(throwing: EngineError.generationFailed(
                            "Apple Intelligence declined this request (safety guardrails)."))
                    default:
                        continuation.finish(throwing: EngineError.generationFailed(
                            error.localizedDescription))
                    }
                } catch let error as EngineError {
                    continuation.finish(throwing: error)
                } catch {
                    continuation.finish(throwing: EngineError.generationFailed(error.localizedDescription))
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
