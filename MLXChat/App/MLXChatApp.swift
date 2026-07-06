import SwiftData
import SwiftUI

extension Notification.Name {
    /// Posted by the app menu (⌘,) to push the in-frame Settings page.
    static let settingsRoute = Notification.Name("openSettingsRoute")
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
        }
    }
}
