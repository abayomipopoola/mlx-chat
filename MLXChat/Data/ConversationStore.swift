import Foundation
import SwiftData

/// CRUD + search over conversations. Owned by the main window; all access on the main actor.
@MainActor
@Observable
final class ConversationStore {
    private let context: ModelContext

    init(context: ModelContext) {
        self.context = context
    }

    // MARK: - CRUD

    @discardableResult
    func newConversation() -> Conversation {
        let conversation = Conversation()
        context.insert(conversation)
        try? context.save()
        return conversation
    }

    func delete(_ conversation: Conversation) {
        context.delete(conversation)
        try? context.save()
    }

    func rename(_ conversation: Conversation, to title: String) {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        conversation.title = trimmed
        try? context.save()
    }

    @discardableResult
    func appendMessage(role: String, content: String, to conversation: Conversation, modelID: String? = nil) -> ChatMessage {
        let nextIndex = (conversation.orderedMessages.last?.sortIndex ?? -1) + 1
        let message = ChatMessage(role: role, content: content, sortIndex: nextIndex, modelID: modelID)
        message.conversation = conversation
        context.insert(message)
        conversation.updatedAt = .now
        deriveTitleIfNeeded(for: conversation)
        try? context.save()
        return message
    }

    func deleteMessage(_ message: ChatMessage) {
        context.delete(message)
        try? context.save()
    }

    func save() {
        try? context.save()
    }

    // MARK: - Title derivation

    /// "New Chat" -> first user message, word-boundary-trimmed to <= 50 chars + ellipsis when cut.
    func deriveTitleIfNeeded(for conversation: Conversation) {
        guard conversation.title == "New Chat",
              let firstUser = conversation.orderedMessages.first(where: { $0.isUser })
        else { return }
        conversation.title = Self.derivedTitle(from: firstUser.content)
    }

    nonisolated static func derivedTitle(from text: String) -> String {
        let collapsed = text
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !collapsed.isEmpty else { return "New Chat" }
        guard collapsed.count > 50 else { return collapsed }
        let head = String(collapsed.prefix(50))
        // Cut back to the last word boundary when one exists past the halfway point.
        if let space = head.lastIndex(of: " "), head.distance(from: head.startIndex, to: space) > 25 {
            return String(head[..<space]) + "…"
        }
        return head + "…"
    }

    // MARK: - Search

    /// In-memory filter: title OR any message content. Fine at local scale.
    nonisolated static func matches(_ conversation: Conversation, query: String) -> Bool {
        let trimmed = query.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return true }
        if conversation.title.localizedStandardContains(trimmed) { return true }
        return conversation.messages.contains { $0.content.localizedStandardContains(trimmed) }
    }
}
