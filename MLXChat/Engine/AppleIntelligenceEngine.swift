import Foundation
import FoundationModels

/// Built-in Apple Intelligence engine (FoundationModels, macOS 26).
final class AppleIntelligenceEngine: ChatEngine {
    let id = appleIntelligenceEngineID
    let isLoaded = true

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

    /// Foundation Models often wraps a whole reply in a markdown fence when the
    /// prompt mentions Markdown. Other engines write the same format inline.
    static func chatStylePrompt(_ systemPrompt: String) -> String {
        systemPrompt
            + "\n\nWrite the reply as chat text. Do not wrap the whole answer in a markdown, json, or xml fence."
    }

    /// Token budget for history trimming.
    static var contextTokens: Int {
        if #available(macOS 26.4, *) {
            return SystemLanguageModel.default.contextSize
        }
        return 4096
    }

    /// Compute the incremental text delta from FoundationModels' cumulative snapshot
    /// using UTF-8 scalar-aligned offsets so combining scalars and ZWJ sequences are
    /// never dropped when grapheme cluster count does not increase.
    static func snapshotDelta(
        previousUTF8Offset: Int,
        current: String
    ) -> (delta: String, nextUTF8Offset: Int) {
        let utf8 = current.utf8
        let count = utf8.count
        guard count > previousUTF8Offset else {
            return ("", previousUTF8Offset)
        }
        var index = utf8.index(utf8.startIndex, offsetBy: max(0, previousUTF8Offset))
        while index > utf8.startIndex, index.samePosition(in: current.unicodeScalars) == nil {
            index = utf8.index(before: index)
        }
        return (String(current.unicodeScalars[index...]), count)
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
                            segments: [.text(.init(content: Self.chatStylePrompt(systemPrompt)))],
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
                    var emittedUTF8 = 0
                    for try await snapshot in stream {
                        let step = Self.snapshotDelta(
                            previousUTF8Offset: emittedUTF8,
                            current: snapshot.content)
                        guard !step.delta.isEmpty else { continue }
                        continuation.yield(.delta(step.delta))
                        emittedUTF8 = step.nextUTF8Offset
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
