import Foundation
import SwiftData

@Model
final class Conversation {
    @Attribute(.unique) var id: UUID
    /// "New Chat" until the first user message, then that text word-boundary-trimmed to <= 50 chars.
    var title: String
    var createdAt: Date
    /// Touched on every message append; the sidebar sorts by this, descending.
    var updatedAt: Date
    @Relationship(deleteRule: .cascade, inverse: \ChatMessage.conversation)
    var messages: [ChatMessage]

    init(title: String = "New Chat") {
        self.id = UUID()
        self.title = title
        self.createdAt = .now
        self.updatedAt = .now
        self.messages = []
    }

    /// Messages in stable order. Always use this, never the raw relationship order.
    var orderedMessages: [ChatMessage] {
        messages.sorted { $0.sortIndex < $1.sortIndex }
    }
}

@Model
final class ChatMessage {
    @Attribute(.unique) var id: UUID
    /// "user" | "assistant"
    var role: String
    /// Visible text; <think> content is excluded.
    var content: String
    /// Captured <think> body, nil when none.
    var thinking: String?
    /// Wall-clock duration of the think phase.
    var thinkingSeconds: Double?
    var createdAt: Date
    /// Monotonic per conversation; ALWAYS order by this.
    var sortIndex: Int
    /// Assistant messages: filled on completion.
    var tokensPerSecond: Double?
    /// Engine id that produced an assistant message.
    var modelID: String?
    /// Downscaled JPEG payload for a vision attachment (user messages only).
    @Attribute(.externalStorage) var imageData: Data?
    /// Display name of an attached file (user messages only).
    var attachmentName: String?
    /// Extracted text of an attached file, length-capped at extraction time.
    var attachmentText: String?
    var conversation: Conversation?

    init(
        role: String,
        content: String,
        sortIndex: Int,
        modelID: String? = nil,
        imageData: Data? = nil,
        attachmentName: String? = nil,
        attachmentText: String? = nil
    ) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.createdAt = .now
        self.sortIndex = sortIndex
        self.modelID = modelID
        self.imageData = imageData
        self.attachmentName = attachmentName
        self.attachmentText = attachmentText
    }

    var isUser: Bool { role == "user" }
}
