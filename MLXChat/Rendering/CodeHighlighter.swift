import Highlightr
import MarkdownUI
import SwiftUI

/// Two-generation O(1) bounded cache matching `MarkdownContentCache`.
/// When `current` reaches `capacity`, it becomes `previous` and the older
/// generation is dropped; hits in `previous` promote back into `current`.
struct TwoGenerationCache<Key: Hashable, Value> {
    private var current: [Key: Value] = [:]
    private var previous: [Key: Value] = [:]
    let capacity: Int

    init(capacity: Int) {
        self.capacity = max(1, capacity)
    }

    var currentCount: Int { current.count }
    var previousCount: Int { previous.count }
    var count: Int { current.count + previous.count }

    mutating func value(forKey key: Key) -> Value? {
        if let value = current[key] {
            return value
        }
        guard let promoted = previous.removeValue(forKey: key) else {
            return nil
        }
        insert(promoted, forKey: key)
        return promoted
    }

    mutating func insert(_ value: Value, forKey key: Key) {
        if current[key] == nil, current.count >= capacity {
            previous = current
            current = [:]
        }
        previous.removeValue(forKey: key)
        current[key] = value
    }

    mutating func removeAll() {
        current.removeAll()
        previous.removeAll()
    }
}

/// Zero-overhead plain monospace code highlighter used for the actively
/// streaming tail block so 100 ms token flushes skip Highlightr's JS engine.
struct PlainCodeSyntaxHighlighter: CodeSyntaxHighlighter {
    static let shared = PlainCodeSyntaxHighlighter()

    func highlightCode(_ code: String, language: String?) -> Text {
        Text(code)
    }
}

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
    private var cache = TwoGenerationCache<String, Text>(capacity: 300)

    private init(theme: String) {
        let highlightr = Highlightr()
        highlightr?.setTheme(to: theme)
        self.highlightr = highlightr
    }

    func highlightedAttributedString(_ code: String, language: String?) -> AttributedString? {
        guard
            let highlightr,
            let highlighted = highlightr.highlight(code, as: normalized(language), fastRender: true),
            var attributed = try? AttributedString(highlighted, including: \.appKit)
        else {
            return nil
        }
        // Strip Highlightr's default Courier font so MarkdownUI's cascaded
        // monospace theme applies identically before and after streaming ends.
        attributed.appKit.font = nil
        attributed.font = nil
        return attributed
    }

    func highlightCode(_ code: String, language: String?) -> Text {
        let key = (language ?? "") + "\u{0}" + code
        if let cached = cache.value(forKey: key) { return cached }

        guard let attributed = highlightedAttributedString(code, language: language) else {
            return Text(code)
        }

        let text = Text(attributed)
        cache.insert(text, forKey: key)
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
