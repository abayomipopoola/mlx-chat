import Foundation
import TurboFieldfare

/// Snapshot of the KV left by the last streamed MoE turn (or by warming the
/// system prompt). Without rewind, resume is valid only for a later turn of
/// the same conversation whose prompt IDs strictly extend this history.
struct StreamedMoEKVCacheState: Equatable, Sendable {
    let conversationID: UUID
    let systemPrompt: String
    let thinkingEnabled: Bool
    let kvTokenIDs: [Int32]
}

enum StreamedMoEKVReusePolicy {
    /// Returns `.resume` only when `cache` is a non-empty strict prefix of
    /// `newPromptIDs` and every conversation key still matches. Any trim,
    /// chat switch, prompt change, or token divergence forces `.reset`.
    ///
    /// A runner that can rewind (`allowsRewind`) instead resumes at the
    /// longest shared token prefix, in any conversation: its cache depends
    /// only on the tokens it was fed. A new chat then skips the system prompt
    /// it shares with the last one, and a thinking turn, whose next prompt
    /// drops the thought, still reuses everything before it.
    static func start(
        newPromptIDs: [Int32],
        cache: StreamedMoEKVCacheState?,
        conversationID: UUID,
        systemPrompt: String,
        thinkingEnabled: Bool,
        allowsRewind: Bool = false
    ) -> RawCompletionStart {
        guard let cache, !cache.kvTokenIDs.isEmpty else { return .reset }
        if allowsRewind {
            // Feed at least the last token again for fresh logits, so a
            // regenerated identical prompt rewinds by one token, not to zero.
            let shared = zip(cache.kvTokenIDs, newPromptIDs).prefix { $0 == $1 }.count
            let resumeAt = min(shared, newPromptIDs.count - 1)
            guard resumeAt > 0 else { return .reset }
            return .resume(cachedPromptTokens: resumeAt)
        }
        guard cache.conversationID == conversationID,
              cache.systemPrompt == systemPrompt,
              cache.thinkingEnabled == thinkingEnabled,
              cache.kvTokenIDs.count < newPromptIDs.count,
              newPromptIDs.starts(with: cache.kvTokenIDs)
        else {
            return .reset
        }
        return .resume(cachedPromptTokens: cache.kvTokenIDs.count)
    }
}
