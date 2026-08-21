import Foundation
import Tokenizers

/// Streaming detokenizer for generation loops.
///
/// Two challenges drive the design:
///
/// 1. BPE byte-fallback splits multi-byte codepoints (e.g. emoji) across several
///    tokens. Naively decoding each token in isolation yields broken UTF-8.
/// 2. swift-transformers' decoder silently drops byte-fallback tokens that sit
///    at the **end** of the decoded sequence (the bytes are committed only once
///    a non-byte-fallback token follows). For us this matters at `flush()`.
/// 3. Byte-level BPE vocabularies (ChatML/Qwen/Ornith) split rare codepoints
///    without using the `<0xNN>` spelling, so `isByteFallback` cannot see them.
///    The library decodes such a half-finished tail through
///    `String(decoding:as: UTF8.self)`, which turns it into U+FFFD. Emitting
///    that placeholder is unrecoverable: the next token completes the codepoint,
///    the freshly decoded prefix no longer matches what we already emitted, and
///    the real character is lost. Ornith 1.5 has no whole-character token for
///    `⇔ ⟺ ∨ ≡`, so every one of them used to reach the transcript as `\u{FFFD}`.
///
/// Strategy:
///   - During `push(_:)` we decode the longest prefix of accumulated IDs that
///     does NOT end with byte-fallback tokens, then emit the delta vs. previously
///     emitted text. Any trailing byte-fallback IDs are held back.
///   - A trailing run of U+FFFD is held back too, for as long as it stays
///     trailing. That covers challenge 3 without needing to know the vocabulary
///     spelling: once the codepoint completes, the placeholder disappears from
///     the decode and the real character is emitted in its place.
///   - During `flush()` we decode the stable prefix as above AND manually
///     assemble the trailing byte-fallback bytes into a UTF-8 string. This
///     recovers text the library would otherwise drop on a sequence-ending
///     codepoint.
struct GFDetokenizer {
    @usableFromInline let tokenizer: any Tokenizer
    @usableFromInline var stableIDs: [Int] = []
    @usableFromInline var trailingByteIDs: [Int] = []
    @usableFromInline var cursor = GFEmitCursor()

    init(tokenizer: GFTokenizer) {
        self.tokenizer = tokenizer.tokenizer
    }

    mutating func push(_ id: Int32) -> String {
        let tokenID = Int(id)
        let token = tokenizer.convertIdToToken(tokenID) ?? ""
        if Self.isByteFallback(token) {
            trailingByteIDs.append(tokenID)
            return ""
        }

        if !trailingByteIDs.isEmpty {
            stableIDs.append(contentsOf: trailingByteIDs)
            trailingByteIDs.removeAll(keepingCapacity: true)
        }
        stableIDs.append(tokenID)

        let current = tokenizer.decode(tokens: stableIDs, skipSpecialTokens: true)
        return commitDelta(GFEmitCursor.holdingTrailingReplacements(current))
    }

    mutating func flush() -> String {
        let decoded = stableIDs.isEmpty
            ? ""
            : tokenizer.decode(tokens: stableIDs, skipSpecialTokens: true)
        // The stream is over, so a still-incomplete codepoint is never going to
        // arrive. Drop the placeholder rather than show it. Trim before joining
        // the byte-fallback tail, or the tail would make the placeholder
        // non-trailing and the trim would miss it.
        let stableText = GFEmitCursor.holdingTrailingReplacements(decoded)

        let trailingText = assembleByteFallback(trailingByteIDs)
        return commitDelta(stableText + trailingText)
    }

    @usableFromInline
    mutating func commitDelta(_ current: String) -> String {
        cursor.commit(current)
    }

    @usableFromInline
    func assembleByteFallback(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(ids.count)
        for id in ids {
            guard let tok = tokenizer.convertIdToToken(id) else { continue }
            guard Self.isByteFallback(tok),
                  let byte = UInt8(tok.dropFirst(3).dropLast(), radix: 16)
            else { continue }
            bytes.append(byte)
        }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }

    @usableFromInline
    static func isByteFallback(_ token: String) -> Bool {
        token.count == 6
            && token.hasPrefix("<0x")
            && token.hasSuffix(">")
            && token.dropFirst(3).dropLast().allSatisfy { $0.isHexDigit }
    }
}
