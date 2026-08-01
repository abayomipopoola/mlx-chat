import Foundation

struct EngineMessage: Sendable {
    enum Role: Sendable { case user, assistant }
    let role: Role
    let text: String
    /// Downscaled JPEG image payloads for the pending user turn (VLM only).
    /// Default empty so existing call sites compile unchanged.
    let images: [Data]

    init(role: Role, text: String, images: [Data] = []) {
        self.role = role
        self.text = text
        self.images = images
    }
}

enum EngineEvent: Sendable {
    case delta(String)
    case thinkingDelta(String)
    /// The whole response turned out to be an unclosed think block: the caller must
    /// discard its thinking buffer — the full text was re-delivered via `.delta`.
    case thinkingReclassified
    case completed(tokensPerSecond: Double?)
}

enum EngineError: LocalizedError, Sendable {
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

/// One chat backend (Apple Intelligence, MLX, or the streamed MoE runtime).
/// `history` always ends with the pending user message.
protocol ChatEngine: AnyObject {
    var id: String { get }
    var isLoaded: Bool { get }
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
