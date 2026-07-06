import Foundation

/// Routes streamed model text into thinking vs. visible content.
///
/// Handles three shapes:
/// - `startsInsideThink`: the chat template pre-opened `<think>` (Qwen3.5 family) —
///   the stream begins mid-reasoning with no opening tag and only emits `</think>`.
/// - `mayEmitThinkTags`: the model may open with an explicit `<think>` tag (Qwen3, R1).
/// - neither: plain passthrough.
struct ThinkTagFilter {
    private enum Mode { case pending, thinking, content }

    private var mode: Mode
    private var pendingBuffer = ""
    /// Holdback while scanning for a possibly chunk-split `</think>`.
    private var carry = ""
    /// Everything emitted as thinking so far — needed for `finish()` reclassification.
    private var emittedThinking = ""

    private static let closeTag = "</think>"
    private static let openTag = "<think>"
    /// `</think>` is 8 chars; holding back 8 guarantees a split tag is never emitted.
    private static let holdback = 8

    init(startsInsideThink: Bool, mayEmitThinkTags: Bool) {
        if startsInsideThink {
            mode = .thinking
        } else if mayEmitThinkTags {
            mode = .pending
        } else {
            mode = .content
        }
    }

    mutating func consume(_ chunk: String) -> (thinking: String, content: String) {
        var input = chunk

        if mode == .pending {
            pendingBuffer += input
            let trimmed = String(pendingBuffer.drop(while: { $0.isWhitespace }))
            if trimmed.isEmpty { return ("", "") }
            if trimmed.hasPrefix(Self.openTag) {
                mode = .thinking
                input = String(trimmed.dropFirst(Self.openTag.count))
                pendingBuffer = ""
            } else if Self.openTag.hasPrefix(trimmed) {
                // Could still become "<think>": keep buffering.
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
            if let range = carry.range(of: Self.closeTag) {
                let thinking = String(carry[..<range.lowerBound])
                var rest = String(carry[range.upperBound...])
                while rest.hasPrefix("\n") { rest.removeFirst() }
                carry = ""
                mode = .content
                emittedThinking += thinking
                return (thinking, rest)
            }
            // Emit all but the last `holdback` characters.
            guard carry.count > Self.holdback else { return ("", "") }
            let cut = carry.index(carry.endIndex, offsetBy: -Self.holdback)
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
    /// If the think block never closed, ALL text (previously emitted thinking included)
    /// is reclassified as content: `reclassified == true` and `content` carries the full
    /// text — the caller must discard its thinking buffer and replace its content.
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
