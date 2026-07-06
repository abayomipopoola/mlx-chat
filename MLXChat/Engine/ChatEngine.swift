import Foundation

struct EngineMessage {
    enum Role { case user, assistant }
    let role: Role
    let text: String
}

enum EngineEvent {
    case delta(String)
    case thinkingDelta(String)
    /// The whole response turned out to be an unclosed think block: the caller must
    /// discard its thinking buffer — the full text was re-delivered via `.delta`.
    case thinkingReclassified
    case completed(tokensPerSecond: Double?)
}

enum EngineError: LocalizedError {
    case unavailable(String)
    case contextExceeded(String)
    case generationFailed(String)

    var errorDescription: String? {
        switch self {
        case .unavailable(let message), .contextExceeded(let message), .generationFailed(let message):
            return message
        }
    }
}

/// One chat backend (Apple Intelligence, or one MLX model).
/// `history` always ends with the pending user message.
protocol ChatEngine: AnyObject {
    var id: String { get }
    /// RAM load only; downloads happen in EngineRuntime before this. Idempotent.
    func load(progress: @escaping @Sendable (Double) -> Void) async throws
    func unload()
    func stream(
        history: [EngineMessage],
        systemPrompt: String,
        temperature: Double,
        conversationID: UUID
    ) -> AsyncThrowingStream<EngineEvent, Error>
}
