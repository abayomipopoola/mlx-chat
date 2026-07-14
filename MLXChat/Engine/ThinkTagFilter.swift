import Foundation

/// Routes streamed model text into thinking vs. visible content.
///
/// The reasoning span is delimited by a pair of markers the model writes inline.
/// The markers are model-specific (see `ThinkingConfig`): Qwen3 / Ternary
/// Bonsai / R1 use `<think>…</think>`; Gemma 4 uses
/// `<|channel>thought … <channel|>`. Three shapes
/// are handled:
/// - `startsInside`: the chat template pre-opened the span (Qwen3.5 with thinking
///   on) — the stream begins mid-reasoning with no opening marker, only a close.
/// - `mayEmit`: the model may open with the marker at the start of the stream.
/// - neither: plain passthrough (non-reasoning model, or thinking switched off).
struct ThinkTagFilter {
    private enum Mode { case pending, thinking, content }

    private var mode: Mode
    private var pendingBuffer = ""
    /// Holdback while scanning for a possibly chunk-split close marker.
    private var carry = ""
    /// Everything emitted as thinking so far — needed for `finish()` reclassification.
    private var emittedThinking = ""

    private let openMarker: String
    private let closeMarker: String
    /// Holding back the close marker's length guarantees a split marker is never emitted.
    private let holdback: Int

    /// - Parameters:
    ///   - open: marker that opens the reasoning span (unused when `startsInside`).
    ///   - close: marker that closes it; text after it is the visible answer.
    ///   - startsInside: the stream begins already inside reasoning (no open marker).
    ///   - mayEmit: the model may write `open` at the start of the stream.
    init(open: String, close: String, startsInside: Bool, mayEmit: Bool) {
        self.openMarker = open
        self.closeMarker = close
        self.holdback = max(close.count, 1)
        if startsInside {
            mode = .thinking
        } else if mayEmit {
            mode = .pending
        } else {
            mode = .content
        }
    }

    /// Plain passthrough — no reasoning to separate.
    init() {
        self.init(open: "", close: "", startsInside: false, mayEmit: false)
    }

    mutating func consume(_ chunk: String) -> (thinking: String, content: String) {
        var input = chunk

        if mode == .pending {
            pendingBuffer += input
            let trimmed = String(pendingBuffer.drop(while: { $0.isWhitespace }))
            if trimmed.isEmpty { return ("", "") }
            if trimmed.hasPrefix(openMarker) {
                mode = .thinking
                input = String(trimmed.dropFirst(openMarker.count))
                pendingBuffer = ""
            } else if openMarker.hasPrefix(trimmed) {
                // Could still become the full open marker: keep buffering.
                return ("", "")
            } else {
                mode = .content
                input = trimmed
                pendingBuffer = ""
            }
        }

        switch mode {
        case .thinking:
            carry += input
            if let range = carry.range(of: closeMarker) {
                let thinking = String(carry[..<range.lowerBound])
                var rest = String(carry[range.upperBound...])
                while rest.hasPrefix("\n") { rest.removeFirst() }
                carry = ""
                mode = .content
                emittedThinking += thinking
                return (thinking, rest)
            }
            // Emit all but the last `holdback` characters.
            guard carry.count > holdback else { return ("", "") }
            let cut = carry.index(carry.endIndex, offsetBy: -holdback)
            let thinking = String(carry[..<cut])
            carry = String(carry[cut...])
            emittedThinking += thinking
            return (thinking, "")

        case .content:
            return ("", input)

        case .pending:
            return ("", "")
        }
    }

    /// Flush at end of stream.
    ///
    /// If the reasoning span never closed, ALL text (previously emitted thinking
    /// included) is reclassified as content: `reclassified == true` and `content`
    /// carries the full text — the caller must discard its thinking buffer and
    /// replace its content.
    mutating func finish() -> (thinking: String, content: String, reclassified: Bool) {
        defer {
            pendingBuffer = ""
            carry = ""
            emittedThinking = ""
        }
        switch mode {
        case .pending:
            return ("", pendingBuffer.trimmingCharacters(in: .whitespacesAndNewlines), false)
        case .thinking:
            return ("", emittedThinking + carry, true)
        case .content:
            return ("", carry, false)
        }
    }
}
