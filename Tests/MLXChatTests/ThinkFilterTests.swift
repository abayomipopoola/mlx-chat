import Testing
@testable import MLXChat

/// Contract: ThinkTagFilter routes streamed text into thinking vs. content across
/// arbitrary chunk boundaries (a split `</think>`/`<think>` is never leaked), and
/// `finish()` reclassifies a never-closed think block as content.
@Suite struct ThinkFilterTests {
    /// Feeds all chunks, accumulating consume output, then finishes.
    private func run(
        _ chunks: [String], startsInsideThink: Bool, mayEmitThinkTags: Bool
    ) -> (thinking: String, content: String, reclassified: Bool) {
        var filter = ThinkTagFilter(
            startsInsideThink: startsInsideThink, mayEmitThinkTags: mayEmitThinkTags)
        var thinking = ""
        var content = ""
        for chunk in chunks {
            let out = filter.consume(chunk)
            thinking += out.thinking
            content += out.content
        }
        let end = filter.finish()
        thinking += end.thinking
        content += end.content
        return (thinking, content, end.reclassified)
    }

    @Test func preOpenedThinkWithSplitCloseTag() {
        let result = run(
            ["reasoning…", "</th", "ink>\n", "Answer"],
            startsInsideThink: true, mayEmitThinkTags: true)
        #expect(result.thinking == "reasoning…")
        #expect(result.content == "Answer")
        #expect(!result.reclassified)
    }

    @Test func explicitThinkTagsInOneChunk() {
        let result = run(["<think>a</think>b"], startsInsideThink: false, mayEmitThinkTags: true)
        #expect(result.thinking == "a")
        #expect(result.content == "b")
        #expect(!result.reclassified)
    }

    @Test func plainPassthroughWhenNoThinkFlags() {
        let result = run(["hello"], startsInsideThink: false, mayEmitThinkTags: false)
        #expect(result.thinking.isEmpty)
        #expect(result.content == "hello")
        #expect(!result.reclassified)
    }

    @Test func neverClosedThinkIsReclassifiedAsContent() {
        var filter = ThinkTagFilter(startsInsideThink: true, mayEmitThinkTags: true)
        _ = filter.consume("only reasoning text")
        let end = filter.finish()
        #expect(end.reclassified)
        // The FULL text comes back as content; the caller discards its thinking buffer.
        #expect(end.content == "only reasoning text")
    }

    @Test func splitOpeningTagAcrossChunks() {
        let result = run(["<th", "ink>x</think>y"], startsInsideThink: false, mayEmitThinkTags: true)
        #expect(result.thinking == "x")
        #expect(result.content == "y")
        #expect(!result.reclassified)
    }
}
