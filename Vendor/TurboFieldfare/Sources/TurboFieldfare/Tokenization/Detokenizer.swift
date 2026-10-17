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
///   - `GemmaDecoding` + `ByteFallbackRun` give O(1) incremental decode for
///     the declared Gemma sequence (`Replace`, `ByteFallback`, `Fuse`) without
///     re-decoding the accumulated ID list.
///   - Local hold-back is re-implemented as incremental partial-byte buffering
///     (`PartialByteBuffer`) on top of that core: trailing `<0xNN>` bytes stay
///     in `ByteFallbackRun`, and byte-level BPE pieces stay in the partial
///     buffer until they form a complete UTF-8 character. A trailing U+FFFD is
///     never emitted mid-stream. `flush()` assembles a complete trailing byte
///     tail and drops an unfinished codepoint.
///   - ChatML (Qwen/Ornith) maps each whole vocab token through the GPT-2
///     `unicodeToByte` table. A single-scalar map would leave `Ġthe` as
///     literal `Ġthe`. Gemma stays on SentencePiece `▁` / `<0xNN>`.
///
/// `barrierTokenIDs` commits both buffers so channel/tool markers do not steal
/// held text from the previous channel.
struct GFDetokenizer {
    @usableFromInline let tokenizer: any Tokenizer
    let skipSpecialTokens: Bool
    private let dialect: ChatDialect
    private let specialTokenIDs: Set<Int32>
    private let barrierTokenIDs: Set<Int32>
    private var run = ByteFallbackRun()
    private var partial = PartialByteBuffer()

    init(tokenizer: GFTokenizer,
         skipSpecialTokens: Bool = true,
         barrierTokenIDs: Set<Int32> = []) {
        self.tokenizer = tokenizer.tokenizer
        self.skipSpecialTokens = skipSpecialTokens
        self.dialect = tokenizer.dialect
        self.specialTokenIDs = tokenizer.specialTokenIDs
        self.barrierTokenIDs = barrierTokenIDs
    }

    mutating func push(_ id: Int32) -> String {
        guard let token = tokenizer.convertIdToToken(Int(id)) else { return "" }
        if skipSpecialTokens, specialTokenIDs.contains(id) {
            return barrierTokenIDs.contains(id) ? commitHeld() : ""
        }
        if dialect == .chatml {
            return run.commit() + partial.pushByteLevelToken(token)
        }
        if let byte = GemmaDecoding.byteValue(token) {
            return run.push(byte)
        }
        return run.commit() + partial.pushText(GemmaDecoding.fragment(token))
    }

    mutating func flush() -> String {
        // Stream over: assemble a complete trailing byte tail, then drop a
        // still-incomplete codepoint rather than show U+FFFD.
        run.commit(dropIncomplete: true) + partial.flush()
    }

    private mutating func commitHeld() -> String {
        run.commit(dropIncomplete: true) + partial.flush()
    }
}
