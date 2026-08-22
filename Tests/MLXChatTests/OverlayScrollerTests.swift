import AppKit
import Testing
@testable import MLXChat

@Suite struct OverlayScrollerTests {
    @Test func knobTrackIsOverlayThin() {
        #expect(ThinOverlayScroller.knobWidth == 6)
        #expect(ThinOverlayScroller.trackWidth == 9)
        #expect(
            ThinOverlayScroller.scrollerWidth(for: .regular, scrollerStyle: .legacy)
                == ThinOverlayScroller.trackWidth)
    }

    @Test func installIsIdempotent() {
        OverlayScrollers.install()
        OverlayScrollers.install()
    }

    @Test func applyThinVerticalScrollerInstallsOnce() {
        let scrollView = NSScrollView()
        OverlayScrollers.applyThinVerticalScroller(to: scrollView)
        let first = scrollView.verticalScroller
        #expect(first is ThinOverlayScroller)
        OverlayScrollers.applyThinVerticalScroller(to: scrollView)
        #expect(scrollView.verticalScroller === first)
    }

    @Test func overlayStyleWidthIsNine() {
        #expect(
            ThinOverlayScroller.scrollerWidth(for: .regular, scrollerStyle: .overlay) == 9)
    }
}
