import Testing
@testable import MLXChat

/// Contract: MessageSegmenter.segment splits assistant markdown into markdown runs
/// and artifact blocks (html/svg any size; other fences at >= 16 lines / >= 1200 chars),
/// preserving small fences inline and flagging unterminated trailing fences.
@Suite struct SegmenterTests {
    @Test func largeCodeFenceBecomesArtifactBetweenMarkdownRuns() {
        let fenceBody = (["# Quicksort demo"] + (1...19).map { "line \($0)" })
            .joined(separator: "\n")
        let text = "Intro paragraph.\n\n```python\n\(fenceBody)\n```\n\nThat's the sort."

        let blocks = MessageSegmenter.segment(text)
        guard blocks.count == 3,
              case .markdown(_, let before) = blocks[0],
              case .artifact(let artifact) = blocks[1],
              case .markdown(_, let after) = blocks[2]
        else {
            Issue.record("expected [markdown, artifact, markdown], got \(blocks)")
            return
        }
        #expect(before.contains("Intro paragraph."))
        #expect(after.contains("That's the sort."))
        #expect(artifact.kind == .code)
        #expect(artifact.isComplete)
        #expect(artifact.lineCount == 20)
        #expect(artifact.language == "python")
        #expect(artifact.content == fenceBody)
        // Leading `# comment` names the artifact.
        #expect(artifact.title == "Quicksort demo")
    }

    @Test func smallFenceStaysEmbeddedInMarkdown() {
        let text = "Before\n\n```python\na = 1\nb = 2\nc = 3\nd = 4\ne = 5\n```\n\nAfter"

        let blocks = MessageSegmenter.segment(text)
        guard blocks.count == 1, case .markdown(_, let markdown) = blocks[0] else {
            Issue.record("expected a single markdown block, got \(blocks)")
            return
        }
        #expect(markdown.contains("```python"))
        #expect(markdown.contains("a = 1"))
        #expect(markdown.contains("Before"))
        #expect(markdown.contains("After"))
    }

    @Test func htmlFenceOfAnySizeBecomesArtifact() {
        let text = "```html\n<div>\n<p>hi</p>\n</div>\n```"

        let blocks = MessageSegmenter.segment(text)
        guard blocks.count == 1, case .artifact(let artifact) = blocks[0] else {
            Issue.record("expected a single artifact block, got \(blocks)")
            return
        }
        #expect(artifact.kind == .html)
        #expect(artifact.isComplete)
        #expect(artifact.content == "<div>\n<p>hi</p>\n</div>")
    }

    @Test func unterminatedTrailingFenceIsIncompleteArtifact() {
        let fenceBody = (0..<16).map { "value_\($0) = \($0)" }.joined(separator: "\n")
        let text = "Intro\n\n```python\n\(fenceBody)"

        let blocks = MessageSegmenter.segment(text)
        guard blocks.count == 2,
              case .markdown = blocks[0],
              case .artifact(let artifact) = blocks[1]
        else {
            Issue.record("expected [markdown, artifact], got \(blocks)")
            return
        }
        #expect(!artifact.isComplete)
        #expect(artifact.kind == .code)
        #expect(artifact.lineCount == 16)
        // No leading comment: falls back to "<Language> code".
        #expect(artifact.title == "Python code")
    }

    @Test func longerCloserClosesAndShortBacktickLineDoesNot() {
        // "``" is content (too short to close); "````" (longer than the opener) closes.
        let text = "```html\n<p>x</p>\n``\ninside still\n````\nAfter"

        let blocks = MessageSegmenter.segment(text)
        guard blocks.count == 2,
              case .artifact(let artifact) = blocks[0],
              case .markdown(_, let after) = blocks[1]
        else {
            Issue.record("expected [artifact, markdown], got \(blocks)")
            return
        }
        #expect(artifact.isComplete)
        #expect(artifact.content == "<p>x</p>\n``\ninside still")
        #expect(after.contains("After"))
    }
}
