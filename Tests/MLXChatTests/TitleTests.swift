import Testing
@testable import MLXChat

/// Contract: ConversationStore.derivedTitle collapses newlines, passes short text
/// through, truncates long text at a word boundary to <= 50 chars + "…", and falls
/// back to "New Chat" for empty input.
@Suite struct TitleTests {
    @Test func shortTextPassesThroughWithNewlinesCollapsed() {
        #expect(ConversationStore.derivedTitle(from: "hello\nworld") == "hello world")
        #expect(ConversationStore.derivedTitle(from: "  trimmed \n") == "trimmed")
    }

    @Test func exactlyFiftyCharsIsNotTruncated() {
        let text = String(repeating: "a", count: 50)
        #expect(ConversationStore.derivedTitle(from: text) == text)
    }

    @Test func longSentenceIsCutAtWordBoundaryWithEllipsis() {
        let sentence = "The quick brown fox jumps over the lazy dog while the sun sets"
        let title = ConversationStore.derivedTitle(from: sentence)

        #expect(title.count <= 51)
        #expect(title.hasSuffix("…"))
        // Word boundary: the kept prefix matches the input and stops right before a space.
        let kept = String(title.dropLast())
        #expect(sentence.hasPrefix(kept))
        let next = sentence.index(sentence.startIndex, offsetBy: kept.count)
        #expect(sentence[next] == " ")
    }

    @Test func unbrokenLongWordIsHardCutWithEllipsis() {
        let title = ConversationStore.derivedTitle(from: String(repeating: "a", count: 80))
        #expect(title.count == 51)
        #expect(title.hasSuffix("…"))
    }

    @Test func emptyOrWhitespaceFallsBackToNewChat() {
        #expect(ConversationStore.derivedTitle(from: "") == "New Chat")
        #expect(ConversationStore.derivedTitle(from: "  \n\t ") == "New Chat")
    }
}
