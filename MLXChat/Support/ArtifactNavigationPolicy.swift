import Foundation

enum ArtifactNavigationDecision {
    case allowInWebView
    case openInBrowser
    case cancel
}

/// Decides whether an artifact preview navigation stays in the WKWebView,
/// opens in the system browser, or is cancelled. Pure: no WebKit types.
struct ArtifactNavigationPolicy {
    private static let browserSchemes: Set<String> = ["http", "https", "mailto"]
    private static let webViewSchemes: Set<String> = ["about", "data", "blob", "applewebdata"]

    static func decision(for url: URL?, isLinkActivated: Bool) -> ArtifactNavigationDecision {
        if isLinkActivated, let url {
            if let scheme = url.scheme?.lowercased() {
                if scheme == "about",
                   let fragment = url.fragment, !fragment.isEmpty {
                    return .allowInWebView
                }
                if browserSchemes.contains(scheme) {
                    return .openInBrowser
                }
                return .cancel
            }
            if url.path.isEmpty, let fragment = url.fragment, !fragment.isEmpty {
                return .allowInWebView
            }
            return .cancel
        }

        guard let url else {
            return .allowInWebView
        }

        guard let scheme = url.scheme?.lowercased() else {
            return .cancel
        }

        if webViewSchemes.contains(scheme) {
            return .allowInWebView
        }
        return .cancel
    }
}
