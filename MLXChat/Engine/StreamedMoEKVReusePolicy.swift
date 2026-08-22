import Foundation
import TurboFieldfare

/// Snapshot of the KV left by the last successful streamed MoE turn.
/// Resume is valid only for a later turn of the same conversation whose
/// newly encoded prompt IDs are a strict prefix continuation of this history.
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
    static func start(
        newPromptIDs: [Int32],
        cache: StreamedMoEKVCacheState?,
        conversationID: UUID,
        systemPrompt: String,
        thinkingEnabled: Bool
    ) -> RawCompletionStart {
        guard let cache,
              cache.conversationID == conversationID,
              cache.systemPrompt == systemPrompt,
              cache.thinkingEnabled == thinkingEnabled,
              !cache.kvTokenIDs.isEmpty,
              cache.kvTokenIDs.count < newPromptIDs.count,
              newPromptIDs.starts(with: cache.kvTokenIDs)
        else {
            return .reset
        }
        return .resume(cachedPromptTokens: cache.kvTokenIDs.count)
    }
}
