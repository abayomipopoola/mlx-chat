import SwiftData
import SwiftUI

struct SidebarView: View {
    @Environment(ChatController.self) private var chat
    @Binding var selectedConversationID: UUID?
    /// Pushes the in-frame Settings page onto the detail column.
    let onOpenSettings: () -> Void
    /// Pops any pushed page (Settings/Models) so the chat is visible again.
    var onNavigateHome: () -> Void = {}

    @Query(sort: \Conversation.updatedAt, order: .reverse)
    private var conversations: [Conversation]

    @State private var searchText = ""
    @State private var renameTarget: Conversation?
    @State private var renameText = ""

    private var filtered: [Conversation] {
        conversations.filter { ConversationStore.matches($0, query: searchText) }
    }

    var body: some View {
        VStack(spacing: 0) {
            Button {
                selectedConversationID = nil
                onNavigateHome()
            } label: {
                Label("New chat", systemImage: "square.and.pencil")
                    .font(Studio.body14Semibold)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.bottom, 10)

            StudioSearchField(placeholder: "Search", text: $searchText)
                .padding(.horizontal, 12)
                .padding(.bottom, 8)

            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(filtered) { conversation in
                        conversationRow(conversation)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.top, 2)
            }

            Divider()
                .overlay(Color.cardStroke)
            Button {
                onOpenSettings()
            } label: {
                Label("Settings", systemImage: "gearshape")
                    .font(Studio.body14Semibold)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
        }
        .padding(.top, Studio.trafficLightInset)
        .background(Color.sidebarBackground)
        .alert("Rename Chat", isPresented: Binding(
            get: { renameTarget != nil },
            set: { if !$0 { renameTarget = nil } }
        )) {
            TextField("Title", text: $renameText)
            Button("Rename") {
                if let target = renameTarget {
                    chat.conversationStore?.rename(target, to: renameText)
                }
                renameTarget = nil
            }
            Button("Cancel", role: .cancel) { renameTarget = nil }
        }
    }

    private static let relativeDate: RelativeDateTimeFormatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.dateTimeStyle = .numeric
        return formatter
    }()

    /// Sub-minute (or clock-skew "future") intervals read as "in 0 seconds"
    /// from the numeric formatter; show a friendly label instead.
    static func relativeLabel(for date: Date) -> String {
        guard Date.now.timeIntervalSince(date) >= 60 else { return "just now" }
        return relativeDate.localizedString(for: date, relativeTo: .now)
    }

    private func conversationRow(_ conversation: Conversation) -> some View {
        Button {
            selectedConversationID = conversation.id
            onNavigateHome()
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text(conversation.title)
                    .font(Studio.body13)
                    .lineLimit(1)
                Text(Self.relativeLabel(for: conversation.updatedAt))
                    .font(.system(size: 11))
                    .foregroundStyle(Color.subtitleGray)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(selectedConversationID == conversation.id
                      ? Color.sidebarSelection : .clear))
        .contextMenu {
            Button("Rename…") {
                renameTarget = conversation
                renameText = conversation.title
            }
            Button("Delete", role: .destructive) {
                delete(conversation)
            }
        }
    }

    private func delete(_ conversation: Conversation) {
        chat.conversationWillBeDeleted(conversation)
        if selectedConversationID == conversation.id {
            selectedConversationID = nil
        }
        chat.conversationStore?.delete(conversation)
    }
}
