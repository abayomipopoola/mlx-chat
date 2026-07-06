import SwiftData
import SwiftUI

extension Notification.Name {
    /// Posted by the app menu (⌘,) to push the in-frame Settings page.
    static let settingsRoute = Notification.Name("openSettingsRoute")
}

@main
struct MLXChatApp: App {
    @State private var chat = ChatController()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(chat)
                .frame(minWidth: 900, minHeight: 600)
        }
        .modelContainer(for: [Conversation.self, ChatMessage.self])
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 906, height: 889)
        .commands {
            CommandGroup(replacing: .appSettings) {
                Button("Settings…") {
                    NotificationCenter.default.post(name: .settingsRoute, object: nil)
                }
                .keyboardShortcut(",", modifiers: .command)
            }
        }
    }
}
