import Highlightr
import MarkdownUI
import SwiftUI

/// MarkdownUI syntax highlighter backed by Highlightr (highlight.js, 185+ languages).
///
/// Highlightr is NOT thread-safe; MarkdownUI invokes `highlightCode` during view body
/// evaluation on the main thread, so a plain class confined by convention is safe here.
final class HighlightrHighlighter: CodeSyntaxHighlighter {
    static let light = HighlightrHighlighter(theme: "xcode")
    static let dark = HighlightrHighlighter(theme: "atom-one-dark")

    static func forScheme(_ scheme: ColorScheme) -> HighlightrHighlighter {
        scheme == .dark ? .dark : .light
    }

    private let highlightr: Highlightr?
    private var cache: [String: Text] = [:]

    private init(theme: String) {
        let highlightr = Highlightr()
        highlightr?.setTheme(to: theme)
        self.highlightr = highlightr
    }

    func highlightCode(_ code: String, language: String?) -> Text {
        let key = (language ?? "") + "\u{0}" + code
        if let cached = cache[key] { return cached }

        guard
            let highlightr,
            let highlighted = highlightr.highlight(code, as: normalized(language), fastRender: true),
            let attributed = try? AttributedString(highlighted, including: \.appKit)
        else {
            return Text(code)
        }

        let text = Text(attributed)
        if cache.count > 300 { cache.removeAll(keepingCapacity: true) }
        cache[key] = text
        return text
    }

    /// Map common aliases; unknown languages fall back to auto-detection.
    private func normalized(_ language: String?) -> String? {
        guard let language = language?.lowercased(), !language.isEmpty else { return nil }
        switch language {
        case "js": return "javascript"
        case "ts": return "typescript"
        case "py": return "python"
        case "sh", "zsh", "shell": return "bash"
        case "yml": return "yaml"
        default: return language
        }
    }
}
