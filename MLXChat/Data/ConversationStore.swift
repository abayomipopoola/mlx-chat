import Foundation
import os
import SwiftData

private let persistenceLogger = Logger(
    subsystem: Bundle.main.bundleIdentifier ?? "org.mlxchat.MLXChat",
    category: "persistence")

/// CRUD + search over conversations. Owned by the main window; all access on the main actor.
@MainActor
@Observable
final class ConversationStore {
    private let context: ModelContext
    @ObservationIgnored
    var onSaveError: ((Error) -> Void)?
    @ObservationIgnored
    var onSaveSuccess: (() -> Void)?
    @ObservationIgnored
    var saveExecutor: () throws -> Void
    /// Conversation ids whose current title came from a greeting-only first
    /// user message in this process. Not persisted; saved "hey" chats stay put.
    @ObservationIgnored
    private var greetingDerivedTitleIDs: Set<UUID> = []

    init(context: ModelContext) {
        self.context = context
        self.saveExecutor = { [context] in try context.save() }
    }

    // MARK: - CRUD

    @discardableResult
    func newConversation() -> Conversation {
        let conversation = Conversation()
        context.insert(conversation)
        persist()
        return conversation
    }

    func delete(_ conversation: Conversation) {
        context.delete(conversation)
        persist()
    }

    func rename(_ conversation: Conversation, to title: String) {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        conversation.title = trimmed
        greetingDerivedTitleIDs.remove(conversation.id)
        persist()
    }

    @discardableResult
    func appendMessage(
        role: String,
        content: String,
        to conversation: Conversation,
        modelID: String? = nil,
        imageData: Data? = nil,
        attachmentName: String? = nil,
        attachmentText: String? = nil
    ) -> ChatMessage {
        let nextIndex = (conversation.orderedMessages.last?.sortIndex ?? -1) + 1
        let message = ChatMessage(
            role: role,
            content: content,
            sortIndex: nextIndex,
            modelID: modelID,
            imageData: imageData,
            attachmentName: attachmentName,
            attachmentText: attachmentText)
        message.conversation = conversation
        context.insert(message)
        conversation.updatedAt = .now
        deriveTitleIfNeeded(for: conversation)
        persist()
        return message
    }

    func deleteMessage(_ message: ChatMessage) {
        context.delete(message)
        persist()
    }

    func save() {
        persist()
    }

    private func persist() {
        do {
            try saveExecutor()
        } catch {
            do { try saveExecutor() } catch {
                persistenceLogger.error("SwiftData save failed: \(error.localizedDescription, privacy: .public)")
                onSaveError?(error)
                return
            }
        }
        onSaveSuccess?()
    }

    // MARK: - Title derivation

    /// "New Chat" -> preferred user message, word-boundary-trimmed to <= 50 chars + ellipsis when cut.
    /// A greeting-only title set in this process can upgrade when a later real message arrives.
    func deriveTitleIfNeeded(for conversation: Conversation) {
        let userMessages = conversation.orderedMessages.filter(\.isUser).map(\.content)
        guard let source = Self.preferredTitleSource(userMessages) else { return }

        if conversation.title == "New Chat" {
            conversation.title = Self.derivedTitle(from: source)
            if Self.isGreetingOnly(source) {
                greetingDerivedTitleIDs.insert(conversation.id)
            } else {
                greetingDerivedTitleIDs.remove(conversation.id)
            }
            return
        }

        guard greetingDerivedTitleIDs.contains(conversation.id),
              !Self.isGreetingOnly(source)
        else { return }
        conversation.title = Self.derivedTitle(from: source)
        greetingDerivedTitleIDs.remove(conversation.id)
    }

    nonisolated static func isGreetingOnly(_ text: String) -> Bool {
        var normalized = text
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        while let last = normalized.last, ".,!?…".contains(last) {
            normalized.removeLast()
        }
        if ["hey", "hi", "hello", "yo", "sup", "thanks", "ok", "okay"].contains(normalized) {
            return true
        }
        return normalized.count < 4
    }

    nonisolated static func preferredTitleSource(_ userMessages: [String]) -> String? {
        if let firstReal = userMessages.first(where: { !isGreetingOnly($0) }) {
            return firstReal
        }
        return userMessages.first
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
