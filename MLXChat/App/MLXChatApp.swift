import AppKit
import SwiftData
import SwiftUI

/// Capture-only window setup that does not depend on Accessibility automation.
/// The release-video harness gets a native 900×776-point window on every fresh
/// launch, even when AppKit has a stale frame from another display.
private struct ReleaseCaptureWindowConfigurator: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        // Defer past the first layout pass. setFrame during window setup can
        // leave the scene with no on-screen window (same class of bug as
        // mutating @AppStorage in RootView.onAppear).
        scheduleConfigure(view, attempt: 0)
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        // Intentionally empty: re-applying the frame on every SwiftUI pass
        // fights WindowOnScreenNudge and can collapse the window.
    }

    private func scheduleConfigure(_ view: NSView, attempt: Int) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
            if let window = view.window {
                configure(window)
                return
            }
            guard attempt < 12 else { return }
            scheduleConfigure(view, attempt: attempt + 1)
        }
    }

    private func configure(_ window: NSWindow) {
        // The capture harness may be attached to an external display whose
        // coordinate space starts at a negative x. Keep the first frame on
        // the built-in display; a tall social window is placed after launch.
        let screen = NSScreen.screens.first(where: { $0.frame.minX >= 0 })
            ?? NSScreen.main
            ?? NSScreen.screens.first
        let x = (screen?.visibleFrame.minX ?? 0) + 80
        let top = (screen?.visibleFrame.maxY ?? 900) - 80
        window.isRestorable = false
        window.setFrame(
            NSRect(x: x, y: top - 776, width: 900, height: 776),
            display: true)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}

extension Notification.Name {
    /// Posted by the app menu (⌘,) to push the in-frame Settings page.
    static let settingsRoute = Notification.Name("openSettingsRoute")
    /// Posted by the View-menu Toggle Sidebar command (⌃⌘S); RootView flips it.
    static let toggleSidebar = Notification.Name("toggleSidebar")
    /// Posted by the File-menu New Chat command (⌘N); RootView clears selection.
    /// Replaces WindowGroup's New Window shortcut so ⌘N starts a chat, not a window.
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
    init() {
        Studio.registerChatFonts()
        OverlayScrollers.install()
    }

    @State private var chat = ChatController()
    @State private var updates = UpdateController()

    private var capturesReleaseVideo: Bool {
        ProcessInfo.processInfo.arguments.contains("--capture-release-video")
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(chat)
                .environment(updates)
                .frame(minWidth: 900, minHeight: 600)
                .background {
                    if capturesReleaseVideo {
                        ReleaseCaptureWindowConfigurator()
                    }
                }
                .task {
                    if !capturesReleaseVideo { updates.checkInBackground() }
                }
                .task { EscKeyMonitor.install() }
        }
        .modelContainer(
            for: [Conversation.self, ChatMessage.self],
            inMemory: capturesReleaseVideo)
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
            CommandGroup(replacing: .newItem) {
                Button("New Chat") {
                    NotificationCenter.default.post(name: .newChat, object: nil)
                }
                .keyboardShortcut("n", modifiers: .command)
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
