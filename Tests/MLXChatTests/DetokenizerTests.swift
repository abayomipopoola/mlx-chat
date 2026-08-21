import Testing
import TurboFieldfare

/// Guards the streaming emit cursor.
///
/// Byte-level BPE vocabularies split rare codepoints across tokens, so a decode
/// taken mid-codepoint ends in U+FFFD. Ornith 1.5 has no whole-character token
/// for `⇔ ⟺ ∨ ≡`, which is why those characters — and only those — used to reach
/// the response pane as a replacement character.
@Suite struct EmitCursorTests {
    /// Feeds a sequence of decodes and returns everything the cursor emitted.
    private func stream(_ decodes: [String]) -> String {
        var cursor = GFEmitCursor()
        var out = ""
        for decode in decodes {
            out += cursor.commit(GFEmitCursor.holdingTrailingReplacements(decode))
        }
        return out
    }

    @Test func plainTextEmitsEachDeltaOnce() {
        #expect(stream(["S", "S ", "S is", "S is a knight"]) == "S is a knight")
    }

    @Test func codepointSplitAcrossTokensEmitsTheRealCharacter() {
        // The three bytes of ⇔ (E2 87 94) arrive as three tokens.
        let emitted = stream(["S \u{FFFD}", "S \u{FFFD}", "S ⇔", "S ⇔ ¬R"])
        #expect(emitted == "S ⇔ ¬R")
        #expect(!emitted.contains("\u{FFFD}"))
    }

    @Test func severalSplitCodepointsInOneLineAllSurvive() {
        let emitted = stream([
            "Q \u{FFFD}", "Q ⇔", "Q ⇔ (¬P \u{FFFD}", "Q ⇔ (¬P ∨", "Q ⇔ (¬P ∨ S)",
        ])
        #expect(emitted == "Q ⇔ (¬P ∨ S)")
        #expect(!emitted.contains("\u{FFFD}"))
    }

    @Test func replacementCharacterFollowedByTextIsStillEmitted() {
        // Not a split codepoint — the model produced U+FFFD. Once it stops being
        // trailing it is real text, so it goes out rather than vanishing.
        let emitted = stream(["a \u{FFFD}", "a \u{FFFD} b"])
        #expect(emitted == "a \u{FFFD} b")
    }

    @Test func incompleteCodepointAtEndOfStreamIsDroppedNotShown() {
        #expect(stream(["done", "done \u{FFFD}"]) == "done ")
    }

    @Test func rewrittenPrefixReemitsTheTailInsteadOfLosingIt() {
        // The old cursor returned "" here and the corrected text never appeared.
        #expect(stream(["hel", "help me"]) == "help me")
    }

    @Test func cursorTracksWhatItHasEmitted() {
        var cursor = GFEmitCursor()
        #expect(cursor.emitted.isEmpty)
        _ = cursor.commit("half")
        #expect(cursor.emitted == "half")
    }

    @Test func holdBackOnlyTrimsTrailingReplacements() {
        #expect(GFEmitCursor.holdingTrailingReplacements("ok") == "ok")
        #expect(GFEmitCursor.holdingTrailingReplacements("ok\u{FFFD}\u{FFFD}") == "ok")
        #expect(GFEmitCursor.holdingTrailingReplacements("o\u{FFFD}k") == "o\u{FFFD}k")
        #expect(GFEmitCursor.holdingTrailingReplacements("").isEmpty)
    }
}
