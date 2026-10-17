import Foundation
import Tokenizers

/// The pinned Gemma 4 detokenization pipeline, without HF's cleanup pass.
///
/// `tokenizer.json` declares `decoder: Sequence[Replace("▁" -> " "),
/// ByteFallback, Fuse]`. swift-transformers runs that sequence and then applies
/// `clean_up_tokenization_spaces`, which defaults to **true** when the key is
/// absent — and Gemma's `tokenizer_config.json` omits it.
///
/// That cleanup pass is a legacy heuristic for whitespace tokenizers and is
/// wrong for a metaspace + byte-fallback BPE, which is lossless by
/// construction. It rewrites text the model deliberately produced:
///
/// ```text
/// "he said ' ok ' now"  ->  "he said'ok'now"
/// "step 1 . done"       ->  "step 1. done"
/// "    . indented"      ->  "   . indented"
/// ```
///
/// So `decode(encode(x)) != x`. It also breaks streaming: because the pass runs
/// over the whole string on every call, appending a token can rewrite text that
/// was already emitted, which no append-only stream can represent.
///
/// This type reproduces the declared decoder sequence and stops there. Both the
/// batch (`GFTokenizer.decode`) and streaming (`GFDetokenizer`) paths go through
/// it, so they agree by construction.
enum GemmaDecoding {
    /// `<0xXX>` byte-fallback token, e.g. `<0xE2>`.
    static func isByteFallback(_ token: String) -> Bool {
        token.count == 6
            && token.hasPrefix("<0x")
            && token.hasSuffix(">")
            && token.dropFirst(3).dropLast().allSatisfy { $0.isHexDigit }
    }

    static func byteValue(_ token: String) -> UInt8? {
        guard isByteFallback(token) else { return nil }
        return UInt8(token.dropFirst(3).dropLast(), radix: 16)
    }

    /// One token's contribution to the output: the `Replace` decoder's
    /// `"▁" -> " "` substitution. Byte-fallback tokens are handled by
    /// `ByteFallbackRun` instead — they only decode as a complete run.
    static func fragment(_ token: String) -> String {
        // Runs once per generated token; most tokens contain no "▁", and the
        // guard returns them without Foundation bridging or an allocation.
        guard token.unicodeScalars.contains(sentencePieceUnderline) else { return token }
        var scalars = String.UnicodeScalarView()
        for scalar in token.unicodeScalars {
            scalars.append(scalar == sentencePieceUnderline ? " " : scalar)
        }
        return String(scalars)
    }

    private static let sentencePieceUnderline: Unicode.Scalar = "\u{2581}"
}

/// A run of `<0xXX>` byte-fallback tokens, decoded with the reference HF
/// `tokenizers` ByteFallback semantics: the run commits as a whole — valid
/// UTF-8 becomes its text, anything else (including an incomplete trailing
/// sequence) becomes one U+FFFD per byte of the run. The swift-transformers
/// port differs here (it decodes invalid runs leniently and drops a trailing
/// run outright — issue #58); we follow the Rust reference.
///
/// Streaming: while the run is valid so far, nothing can be emitted, because
/// one more byte can invalidate the whole run retroactively. The moment the
/// run becomes invalid its fate is sealed — every byte, past and future,
/// decodes to exactly one U+FFFD — so an invalid run streams replacement
/// characters live rather than freezing the stream, and buffers nothing.
struct ByteFallbackRun {
    private var bytes: [UInt8] = []
    private var poisoned = false
    /// Continuation bytes still owed by the current lead, and the allowed
    /// range for the next one (RFC 3629 constrains the first continuation
    /// after E0/ED/F0/F4 leads).
    private var pendingContinuations = 0
    private var nextContinuation: ClosedRange<UInt8> = 0x80...0xBF

    /// Text this byte contributes to the stream: `""` while the run is valid
    /// so far, replacement characters once it can no longer become valid.
    mutating func push(_ byte: UInt8) -> String {
        if poisoned { return "\u{FFFD}" }
        bytes.append(byte)
        if accept(byte) { return "" }
        poisoned = true
        defer { bytes.removeAll(keepingCapacity: true) }
        return String(repeating: "\u{FFFD}", count: bytes.count)
    }

    /// Close the run: a non-byte token follows, or the stream ends.
    mutating func commit(dropIncomplete: Bool = false) -> String {
        defer {
            bytes.removeAll(keepingCapacity: true)
            poisoned = false
            pendingContinuations = 0
            nextContinuation = 0x80...0xBF
        }
        guard !poisoned, !bytes.isEmpty else { return "" }
        if let text = String(bytes: bytes, encoding: .utf8) { return text }
        return dropIncomplete ? "" : String(repeating: "\u{FFFD}", count: bytes.count)
    }

    /// Advance the incremental UTF-8 validator. `false` means no suffix can
    /// ever make the run valid again.
    private mutating func accept(_ byte: UInt8) -> Bool {
        if pendingContinuations > 0 {
            guard nextContinuation.contains(byte) else { return false }
            pendingContinuations -= 1
            nextContinuation = 0x80...0xBF
            return true
        }
        switch byte {
        case 0x00...0x7F: break
        case 0xC2...0xDF: pendingContinuations = 1
        case 0xE0: pendingContinuations = 2; nextContinuation = 0xA0...0xBF
        case 0xE1...0xEC, 0xEE, 0xEF: pendingContinuations = 2
        case 0xED: pendingContinuations = 2; nextContinuation = 0x80...0x9F
        case 0xF0: pendingContinuations = 3; nextContinuation = 0x90...0xBF
        case 0xF1...0xF3: pendingContinuations = 3
        case 0xF4: pendingContinuations = 3; nextContinuation = 0x80...0x8F
        default: return false // stray continuation, overlong lead, or > U+10FFFF
        }
        return true
    }
}

/// GPT-2 / Qwen / Ornith `bytes_to_unicode` reverse map.
///
/// Byte-level BPE vocabularies spell rare codepoints as one printable Unicode
/// scalar per raw byte, not as `<0xNN>`. Mapping that scalar back to the byte
/// lets the incremental UTF-8 hold-back reassemble `⇔ ⟺ ∨ ≡` without a
/// mid-stream U+FFFD.
enum ByteLevelBPE {
    static let unicodeToByte: [UInt32: UInt8] = {
        var printable: [Int] = []
        printable.append(contentsOf: 33...126)
        printable.append(contentsOf: 161...172)
        printable.append(contentsOf: 174...255)
        var bytes = printable
        var chars = printable
        var n = 0
        for b in 0..<256 where !printable.contains(b) {
            bytes.append(b)
            chars.append(256 + n)
            n += 1
        }
        var map: [UInt32: UInt8] = [:]
        map.reserveCapacity(256)
        for (byte, char) in zip(bytes, chars) {
            map[UInt32(char)] = UInt8(byte)
        }
        return map
    }()

    static func rawByte(from token: String) -> UInt8? {
        var scalars = token.unicodeScalars.makeIterator()
        guard let first = scalars.next(), scalars.next() == nil else { return nil }
        return unicodeToByte[first.value]
    }

    /// GPT-2 / Qwen / Ornith whole-token reverse map. Every scalar must be in
    /// `bytes_to_unicode`; otherwise the token is not a byte-level spelling.
    static func rawBytes(from token: String) -> [UInt8]? {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(token.unicodeScalars.count)
        for scalar in token.unicodeScalars {
            guard let byte = unicodeToByte[scalar.value] else { return nil }
            bytes.append(byte)
        }
        return bytes
    }
}

/// Incremental UTF-8 assembly for tokens that are not `<0xNN>` byte-fallback
/// spellings. Holds an incomplete trailing codepoint as raw bytes and never
/// emits U+FFFD mid-stream. At flush, a still-incomplete tail is dropped.
struct PartialByteBuffer {
    private var bytes: [UInt8] = []

    mutating func pushText(_ text: String) -> String {
        // Gemma SentencePiece fragments must stay UTF-8. The GPT-2 byte map
        // belongs only on `pushByteLevelToken` (ChatML). Mapping a lone `é`
        // through `rawByte` turns it into lead 0xE9 and the character is lost.
        pushBytes(Array(text.utf8))
    }

    /// ChatML / Qwen / Ornith path: map every vocab scalar through
    /// `unicodeToByte` so `Ġthe` becomes ` the` and `Ã©` becomes `é`.
    mutating func pushByteLevelToken(_ token: String) -> String {
        if let bytes = ByteLevelBPE.rawBytes(from: token) {
            return pushBytes(bytes)
        }
        return pushBytes(Array(token.utf8))
    }

    mutating func pushBytes(_ newBytes: [UInt8]) -> String {
        guard !newBytes.isEmpty else { return "" }
        bytes.append(contentsOf: newBytes)
        return drainCompleteUTF8()
    }

    /// End of stream: emit a complete UTF-8 tail, drop an unfinished codepoint.
    mutating func flush() -> String {
        let emitted = drainCompleteUTF8()
        bytes.removeAll(keepingCapacity: true)
        return emitted
    }

    mutating func commit() -> String {
        flush()
    }

    private mutating func drainCompleteUTF8() -> String {
        guard !bytes.isEmpty else { return "" }
        var index = 0
        var lastComplete = 0
        while index < bytes.count {
            let length = utf8SequenceLength(bytes[index])
            guard length > 0 else {
                // Invalid lead: emit the complete prefix and drop the bad byte
                // without inventing U+FFFD mid-stream.
                let prefix = String(bytes: bytes[0..<lastComplete], encoding: .utf8) ?? ""
                bytes.removeFirst(lastComplete + 1)
                return prefix + drainCompleteUTF8()
            }
            guard index + length <= bytes.count else { break }
            if !isValidUTF8Sequence(bytes[index..<(index + length)]) {
                let prefix = String(bytes: bytes[0..<lastComplete], encoding: .utf8) ?? ""
                bytes.removeFirst(lastComplete + 1)
                return prefix + drainCompleteUTF8()
            }
            index += length
            lastComplete = index
        }
        let emitted = String(bytes: bytes[0..<lastComplete], encoding: .utf8) ?? ""
        bytes.removeFirst(lastComplete)
        return emitted
    }

    private func utf8SequenceLength(_ lead: UInt8) -> Int {
        switch lead {
        case 0x00...0x7F: return 1
        case 0xC2...0xDF: return 2
        case 0xE0...0xEF: return 3
        case 0xF0...0xF4: return 4
        default: return 0
        }
    }

    private func isValidUTF8Sequence(_ slice: ArraySlice<UInt8>) -> Bool {
        let bytes = Array(slice)
        switch bytes.count {
        case 1:
            return bytes[0] <= 0x7F
        case 2:
            return (0xC2...0xDF).contains(bytes[0]) && (0x80...0xBF).contains(bytes[1])
        case 3:
            let c1 = bytes[1]
            let c2 = bytes[2]
            guard (0x80...0xBF).contains(c2) else { return false }
            switch bytes[0] {
            case 0xE0: return (0xA0...0xBF).contains(c1)
            case 0xED: return (0x80...0x9F).contains(c1)
            case 0xE1...0xEC, 0xEE, 0xEF: return (0x80...0xBF).contains(c1)
            default: return false
            }
        case 4:
            let c1 = bytes[1]
            guard (0x80...0xBF).contains(bytes[2]), (0x80...0xBF).contains(bytes[3]) else {
                return false
            }
            switch bytes[0] {
            case 0xF0: return (0x90...0xBF).contains(c1)
            case 0xF1...0xF3: return (0x80...0xBF).contains(c1)
            case 0xF4: return (0x80...0x8F).contains(c1)
            default: return false
            }
        default:
            return false
        }
    }
}
