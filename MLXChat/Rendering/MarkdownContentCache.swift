import Foundation
import MarkdownUI

/// Memoizes parsed Markdown for finished text, so a row that renders again
/// (a send, a finish, reopening a chat) skips the math pass and the parse.
/// Streaming text changes every flush and is not cached.
@MainActor
final class MarkdownContentCache {
    static let shared = MarkdownContentCache()

    private struct Key: Hashable {
        let text: String
        let math: Bool
    }

    /// Two generations: a full `current` becomes `previous`, and a hit in
    /// `previous` moves back to `current`, so text in use stays cached.
    private var current: [Key: MarkdownContent] = [:]
    private var previous: [Key: MarkdownContent] = [:]
    private let capacity: Int

    init(capacity: Int = 256) {
        self.capacity = capacity
    }

    /// `math` runs `MathPreprocessor` first, as message bodies need.
    func content(for text: String, math: Bool = true) -> MarkdownContent {
        let key = Key(text: text, math: math)
        if let cached = current[key] {
            return cached
        }
        let content = previous.removeValue(forKey: key)
            ?? MarkdownContent(math ? MathPreprocessor.preprocess(text) : text)
        if current.count >= capacity {
            previous = current
            current = [:]
        }
        current[key] = content
        return content
    }
}
