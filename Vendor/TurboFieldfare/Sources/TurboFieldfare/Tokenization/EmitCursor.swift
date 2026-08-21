import Foundation

/// Tracks how much of a monotonically growing decode has already been handed to
/// a caller, and turns each fresh decode into the delta still owed.
///
/// Split out from `GFDetokenizer` so the emit bookkeeping — the part that made
/// `⇔` reach the transcript as U+FFFD — can be tested without a vocabulary.
public struct GFEmitCursor: Sendable {
    /// Text already returned to the caller.
    public private(set) var emitted: String = ""

    public init() {}

    /// Returns the part of `current` not yet emitted, and records it as emitted.
    public mutating func commit(_ current: String) -> String {
        guard current.hasPrefix(emitted) else {
            // The decoder rewrote a prefix. Re-emit from the point the two texts
            // diverge: a short repeat is visible and the reader can recover from
            // it, whereas dropping the tail loses characters with no trace.
            let shared = current.commonPrefix(with: emitted)
            emitted = current
            return String(current.dropFirst(shared.count))
        }
        let delta = String(current.dropFirst(emitted.count))
        emitted = current
        return delta
    }

    /// Strips a trailing run of U+FFFD.
    ///
    /// Mid-stream, such a run means the decoder saw a codepoint whose bytes have
    /// not all arrived yet; holding it back keeps the placeholder out of
    /// `emitted`, so the decode that completes the codepoint still shares a
    /// prefix with what we emitted and the real character is emitted normally.
    /// A U+FFFD the model itself produced is only withheld while it sits last,
    /// and is emitted as soon as any text follows it.
    public static func holdingTrailingReplacements(_ text: String) -> String {
        guard text.unicodeScalars.last == "\u{FFFD}" else { return text }
        var scalars = text.unicodeScalars
        while scalars.last == "\u{FFFD}" { scalars.removeLast() }
        return String(scalars)
    }
}
