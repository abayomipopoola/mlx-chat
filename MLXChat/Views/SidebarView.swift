import SwiftData
import SwiftUI

struct SidebarView: View {
    @Environment(ChatController.self) private var chat
    @Environment(UpdateController.self) private var updates
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
        if ProcessInfo.processInfo.arguments.contains("--capture-hide-history") {
            return []
        }
        return conversations.filter { ConversationStore.matches($0, query: searchText) }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Window chrome (toggle + back/forward) is an overlay on RootView.
            // Reserve the same vertical band so New chat stays below it.
            Color.clear
                .frame(height: Studio.titlebarSidebarClearance)

            SidebarChromeButton(
                title: "New Chat",
                systemImage: "plus",
                shortcutKeys: ["⌘", "N"],
                help: "New Chat (⌘N)",
                identifier: "newChatButton"
            ) {
                selectedConversationID = nil
                onNavigateHome()
            }
            .padding(.horizontal, Studio.sidebarInset)
            .padding(.bottom, 6)

            StudioSearchField(
                placeholder: "Search", text: $searchText,
                ringOnFocusOnly: true,
                accessibilityIdentifier: "sidebarSearchField",
                chromeForeground: .sidebarMeta)
                .padding(.horizontal, Studio.sidebarInset)
                .padding(.bottom, 6)

            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(filtered) { conversation in
                        conversationRow(conversation)
                    }
                }
                .padding(.horizontal, Studio.sidebarInset)
                .padding(.top, 2)
                .padding(.bottom, 6)
            }
            .accessibilityIdentifier("conversationList")
            .overlay(alignment: .bottom) {
                LinearGradient(
                    colors: [Color.sidebarBackground.opacity(0), Color.sidebarBackground],
                    startPoint: .top,
                    endPoint: .bottom)
                    .frame(height: Studio.sidebarListFade)
                    .allowsHitTesting(false)
            }

            if let version = updates.availableVersion {
                HStack {
                    Spacer()
                    Button {
                        updates.checkForUpdates()
                    } label: {
                        Image(systemName: "arrow.down")
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(.white)
                            .frame(width: 36, height: 36)
                            .background(Circle().fill(Color.brandGreen))
                            .contentShape(Circle())
                    }
                    .buttonStyle(.plain)
                    .help("Install MLX Chat \(version)")
                    .accessibilityLabel("Install MLX Chat \(version)")
                    .accessibilityIdentifier("installUpdateButton")
                }
                .padding(.horizontal, Studio.sidebarInset)
                .padding(.top, 8)
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }

            // Outside the conversation ScrollView so history cannot slide over it.
            SidebarChromeButton(
                title: "Settings",
                systemImage: "gearshape",
                shortcutKeys: ["⌘", ","],
                help: "Settings (⌘,)",
                identifier: "settingsGearButton",
                action: onOpenSettings)
            .padding(.horizontal, Studio.sidebarInset)
            .padding(.top, Studio.sidebarSettingsTop)
            .padding(.bottom, 8)
        }
        .animation(.easeOut(duration: 0.2), value: updates.availableVersion)
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
        let selected = selectedConversationID == conversation.id
        return Button {
            selectedConversationID = conversation.id
            onNavigateHome()
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text(conversation.title)
                    .font(Studio.mona(13))
                    .foregroundStyle(selected ? Color.primary : Color.sidebarTitle)
                    .lineLimit(1)
                Text(Self.relativeLabel(for: conversation.updatedAt))
                    .font(Studio.sans(11))
                    .foregroundStyle(Color.sidebarMeta)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("conversationRow")
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(selected ? Color.sidebarSelection : .clear))
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

/// New Chat and Settings: 13 regular, sidebar title color, hover fill,
/// trailing shortcut badge.
private struct SidebarChromeButton: View {
    let title: String
    let systemImage: String
    let shortcutKeys: [String]
    let help: String
    let identifier: String
    let action: () -> Void
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Label(title, systemImage: systemImage)
                    .font(Studio.sans(13))
                    .foregroundStyle(Color.sidebarTitle)
                Spacer(minLength: 8)
                shortcutBadge
                    .opacity(hovered ? 1 : 0)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
        .buttonStyle(QuietHoverFillButtonStyle(cornerRadius: 10))
        .onHover { hovered = $0 }
        .accessibilityIdentifier(identifier)
        .help(help)
        .animation(.easeOut(duration: 0.12), value: hovered)
    }

    private var shortcutBadge: some View {
        HStack(spacing: 4) {
            ForEach(shortcutKeys, id: \.self) { key in
                Text(key)
            }
        }
        .font(Studio.sans(11))
        .foregroundStyle(Color.sidebarMeta)
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(Color.primary.opacity(0.08))
        }
        .accessibilityHidden(true)
    }
}
