import AppKit
import SwiftData
import SwiftUI

extension Notification.Name {
    /// Posted by the app menu (⌘,) to push the in-frame Settings page.
    static let settingsRoute = Notification.Name("openSettingsRoute")
    /// Posted by the View-menu Toggle Sidebar command (⌃⌘S); RootView flips it.
    static let toggleSidebar = Notification.Name("toggleSidebar")
    /// Posted by the File-menu New Chat command (⌃N); RootView clears selection.
    /// Kept at the menu level so the shortcut stays alive when the sidebar is collapsed
    /// (the New chat button leaves the hierarchy with the sidebar).
    static let newChat = Notification.Name("newChat")
    /// Posted by the Esc key monitor; RootView closes any open header dropdown.
    static let dismissHeaderDropdown = Notification.Name("dismissHeaderDropdown")
}

/// Posts `.dismissHeaderDropdown` on every Esc keydown (event passes through
/// untouched, so text fields and sheets keep their own Esc behavior).
private enum EscKeyMonitor {
    private static var installed = false

    static func install() {
        guard !installed else { return }
        installed = true
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            if event.keyCode == 53 {
                NotificationCenter.default.post(name: .dismissHeaderDropdown, object: nil)
            }
            return event
        }
    }
}

@main
struct MLXChatApp: App {
    @State private var chat = ChatController()
    @State private var updates = UpdateController()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(chat)
                .environment(updates)
                .frame(minWidth: 900, minHeight: 600)
                .task { updates.checkInBackground() }
                .task { EscKeyMonitor.install() }
        }
        .modelContainer(for: [Conversation.self, ChatMessage.self])
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 906, height: 889)
        .commands {
            // "Check for Updates…" lives in the app (MLX Chat) menu, right under About.
            CommandGroup(after: .appInfo) {
                Button("Check for Updates…") { updates.checkForUpdates() }
            }
            CommandGroup(replacing: .appSettings) {
                Button("Settings…") {
                    NotificationCenter.default.post(name: .settingsRoute, object: nil)
                }
                .keyboardShortcut(",", modifiers: .command)
            }
            CommandGroup(after: .newItem) {
                Button("New Chat") {
                    NotificationCenter.default.post(name: .newChat, object: nil)
                }
                .keyboardShortcut("n", modifiers: .control)
            }
            CommandGroup(after: .sidebar) {
                Button("Toggle Sidebar") {
                    NotificationCenter.default.post(name: .toggleSidebar, object: nil)
                }
                .keyboardShortcut("s", modifiers: [.control, .command])
            }
        }
    }
}
