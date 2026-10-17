import Testing
@testable import TurboFieldfare

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

@Suite struct IncrementalDecodeHoldbackTests {
    @Test func testByteFallbackRunHoldsIncompleteCodepoint() {
        var run = ByteFallbackRun()
        #expect(run.push(0xE2) == "")
        #expect(run.push(0x87) == "")
        #expect(run.push(0x94) == "")
        #expect(run.commit() == "⇔")
    }

    @Test func testPartialByteBufferHoldsOrnithSplitCodepoint() {
        var buffer = PartialByteBuffer()
        #expect(buffer.pushBytes([0xE2]) == "")
        #expect(buffer.pushBytes([0x87]) == "")
        #expect(buffer.pushBytes([0x94]) == "⇔")
        #expect(buffer.flush().isEmpty)
    }

    @Test func testPartialByteBufferFlushDropsIncompleteTail() {
        var buffer = PartialByteBuffer()
        #expect(buffer.pushText("done ") == "done ")
        #expect(buffer.pushBytes([0xE2]) == "")
        #expect(buffer.flush() == "")
    }

    @Test func testGemmaPushTextKeepsAccentedPiece() {
        var buffer = PartialByteBuffer()
        #expect(buffer.pushText(GemmaDecoding.fragment("▁caf")) == " caf")
        #expect(buffer.pushText("é") == "é")
        #expect(buffer.flush().isEmpty)
    }

    @Test func testGemmaDecodingDoesNotRescanTokenLists() {
        #expect(GemmaDecoding.isByteFallback("<0xE2>"))
        #expect(GemmaDecoding.byteValue("<0xE2>") == 0xE2)
        #expect(GemmaDecoding.fragment("▁Hello") == " Hello")
        #expect(!GemmaDecoding.isByteFallback("⇔"))
    }

    @Test func testByteLevelBPEMapsWholeTokenGtheToSpaceThe() {
        #expect(ByteLevelBPE.rawBytes(from: "Ġthe") == Array(" the".utf8))
        var buffer = PartialByteBuffer()
        #expect(buffer.pushByteLevelToken("Ġthe") == " the")
        #expect(buffer.flush().isEmpty)
    }

    @Test func testByteLevelBPEMapsChatMLEncodedEAcute() {
        // GPT-2 spelling of UTF-8 C3 A9 (é) is U+00C3 U+00A9, not "Ã" + "é".
        let spelling = "\u{00C3}\u{00A9}"
        #expect(ByteLevelBPE.rawBytes(from: spelling) == [0xC3, 0xA9])
        var buffer = PartialByteBuffer()
        #expect(buffer.pushByteLevelToken(spelling) == "é")
    }

    @Test func testByteLevelWholeTokenStillHoldsSplitCodepoint() {
        func gpt2Spelling(_ byte: UInt8) -> String {
            let scalarValue = ByteLevelBPE.unicodeToByte.first { $0.value == byte }!.key
            return String(Unicode.Scalar(scalarValue)!)
        }
        var buffer = PartialByteBuffer()
        #expect(buffer.pushByteLevelToken(gpt2Spelling(0xE2)) == "")
        #expect(buffer.pushByteLevelToken(gpt2Spelling(0x87)) == "")
        #expect(buffer.pushByteLevelToken(gpt2Spelling(0x94)) == "⇔")
        #expect(buffer.flush().isEmpty)
    }
}
