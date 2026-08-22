import AppKit

/// Prefer overlay scrollers even when System Settings is "Always".
/// Custom NSScroller swizzling is intentionally not used: exchanging `tile`
/// from Swift recurses, which left the debug app running with no window.
enum OverlayScrollers {
    private static var installed = false

    static func install() {
        guard !installed else { return }
        installed = true
        UserDefaults.standard.set("WhenScrolling", forKey: "AppleShowScrollBars")
        WindowOnScreenNudge.install()
    }

    static func applyThinVerticalScroller(to scrollView: NSScrollView) {
        guard !(scrollView.verticalScroller is ThinOverlayScroller) else { return }
        let scroller = ThinOverlayScroller()
        scroller.scrollerStyle = .overlay
        scrollView.verticalScroller = scroller
    }
}

/// Restored frames can sit on a disconnected display (negative origin).
/// Move any window that does not intersect a real screen onto the main display.
enum WindowOnScreenNudge {
    private static var installed = false

    static func install() {
        guard !installed else { return }
        installed = true
        NotificationCenter.default.addObserver(
            forName: NSApplication.didFinishLaunchingNotification,
            object: nil,
            queue: .main
        ) { _ in
            NSApp.windows.forEach(moveOnScreen)
        }
        NotificationCenter.default.addObserver(
            forName: NSWindow.didBecomeKeyNotification,
            object: nil,
            queue: .main
        ) { note in
            if let window = note.object as? NSWindow {
                moveOnScreen(window)
            }
        }
        DispatchQueue.main.async {
            NSApp.windows.forEach(moveOnScreen)
        }
    }

    static func moveOnScreen(_ window: NSWindow) {
        let frame = window.frame
        guard frame.width >= 400, frame.height >= 400 else { return }
        let onAScreen = NSScreen.screens.contains { $0.visibleFrame.intersects(frame) }
        guard !onAScreen else { return }
        guard let visible = (NSScreen.screens.first(where: { $0.frame.minX >= 0 })
            ?? NSScreen.main
            ?? NSScreen.screens.first)?.visibleFrame
        else { return }

        var next = frame
        next.origin.x = visible.midX - next.width / 2
        next.origin.y = visible.midY - next.height / 2
        if next.maxX > visible.maxX { next.origin.x = visible.maxX - next.width }
        if next.maxY > visible.maxY { next.origin.y = visible.maxY - next.height }
        if next.minX < visible.minX { next.origin.x = visible.minX }
        if next.minY < visible.minY { next.origin.y = visible.minY }
        window.setFrame(next, display: true)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}

final class ThinOverlayScroller: NSScroller {
    static let knobWidth: CGFloat = 6
    static let trackWidth: CGFloat = 9

    override class func scrollerWidth(
        for controlSize: NSControl.ControlSize,
        scrollerStyle: NSScroller.Style
    ) -> CGFloat {
        trackWidth
    }
}
