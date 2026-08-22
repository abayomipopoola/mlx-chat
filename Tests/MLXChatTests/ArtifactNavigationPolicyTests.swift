import Foundation
import Testing
@testable import MLXChat

@Suite struct ArtifactNavigationPolicyTests {
    @Test func documentLocalSchemesAllowInWebView() {
        #expect(decision("about:blank") == .allowInWebView)
        #expect(ArtifactNavigationPolicy.decision(for: nil, isLinkActivated: false) == .allowInWebView)
        #expect(decision("data:text/html,<p>hi</p>") == .allowInWebView)
        #expect(decision("blob:") == .allowInWebView)
        #expect(decision("applewebdata://") == .allowInWebView)
    }

    @Test func networkAndFileSchemesCancelWhenNotLinkActivated() {
        #expect(decision("https://example.com") == .cancel)
        #expect(decision("http://example.com") == .cancel)
        #expect(decision("ws://example.com") == .cancel)
        #expect(decision("wss://example.com") == .cancel)
        #expect(decision("file:///tmp/page.html") == .cancel)
        #expect(decision("ftp://example.com/file") == .cancel)
        #expect(decision("custom-scheme://payload") == .cancel)
    }

    @Test func missingSchemeCancels() {
        #expect(decision("example.com") == .cancel)
    }

    @Test func linkActivatedSafeSchemesOpenInBrowser() {
        #expect(decision("https://example.com", isLinkActivated: true) == .openInBrowser)
        #expect(decision("http://example.com", isLinkActivated: true) == .openInBrowser)
        #expect(decision("mailto:someone@example.com", isLinkActivated: true) == .openInBrowser)
    }

    @Test func linkActivatedUnsafeSchemesCancel() {
        #expect(decision("file:///tmp/page.html", isLinkActivated: true) == .cancel)
        #expect(decision("javascript:alert(1)", isLinkActivated: true) == .cancel)
    }

    @Test func schemesAreCaseInsensitive() {
        #expect(decision("HTTPS://example.com") == .cancel)
        #expect(decision("ABOUT:blank") == .allowInWebView)
    }

    @Test func jsDrivenHTTPSCancelsAndNeverOpensInBrowser() {
        let result = decision("https://example.com", isLinkActivated: false)
        #expect(result == .cancel)
        #expect(result != .openInBrowser)
    }

    private func decision(_ raw: String, isLinkActivated: Bool = false) -> ArtifactNavigationDecision {
        ArtifactNavigationPolicy.decision(for: URL(string: raw), isLinkActivated: isLinkActivated)
    }
}
