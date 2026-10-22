import Foundation
import Testing
import TurboFieldfare
@testable import MLXChat

@Suite struct StreamedMoEKVReuseTests {
    private let conversationA = UUID(uuidString: "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA")!
    private let conversationB = UUID(uuidString: "BBBBBBBB-BBBB-BBBB-BBBB-BBBBBBBBBBBB")!
    private let cachedIDs: [Int32] = [1, 2, 3, 4]
    private let continuation: [Int32] = [1, 2, 3, 4, 5, 6]

    @Test func differentConversationIDWithIdenticalPrefixResets() {
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: continuation,
            cache: makeCache(conversationID: conversationA, kvTokenIDs: cachedIDs),
            conversationID: conversationB,
            systemPrompt: "Be concise.",
            thinkingEnabled: true)
        #expect(start == .reset)
    }

    @Test func rewindableRunnerResumesAtTheLongestSharedPrefix() {
        // The cached turn ended in a thought the next prompt no longer
        // carries: [1, 2, 3] is shared, the cached 4 diverges.
        let nextPrompt: [Int32] = [1, 2, 3, 9, 10]
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: nextPrompt,
            cache: makeCache(kvTokenIDs: cachedIDs),
            conversationID: conversationA,
            systemPrompt: "Be concise.",
            thinkingEnabled: true,
            allowsRewind: true)
        #expect(start == .resume(cachedPromptTokens: 3))
        #expect(StreamedMoEKVReusePolicy.start(
            newPromptIDs: nextPrompt,
            cache: makeCache(kvTokenIDs: cachedIDs),
            conversationID: conversationA,
            systemPrompt: "Be concise.",
            thinkingEnabled: true) == .reset)
    }

    @Test func rewindableRunnerResumesAcrossConversationsAtTheSharedPrefix() {
        // A new chat shares the system prompt tokens [1, 2, 3] with the last
        // one; the conversation, prompt text and thinking keys do not gate it.
        let newChat: [Int32] = [1, 2, 3, 8, 9]
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: newChat,
            cache: makeCache(conversationID: conversationA, systemPrompt: "Be concise.",
                             thinkingEnabled: true, kvTokenIDs: cachedIDs),
            conversationID: conversationB,
            systemPrompt: "Be thorough.",
            thinkingEnabled: false,
            allowsRewind: true)
        #expect(start == .resume(cachedPromptTokens: 3))
    }

    @Test func rewindableRunnerRefeedsOnlyTheLastTokenOfAFullyCachedPrompt() {
        // Regenerate sends the same prompt; at least one token must be fed
        // again for fresh logits, so resume one short of the end.
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: [1, 2, 3],
            cache: makeCache(kvTokenIDs: cachedIDs),
            conversationID: conversationB,
            systemPrompt: "Be concise.",
            thinkingEnabled: true,
            allowsRewind: true)
        #expect(start == .resume(cachedPromptTokens: 2))
        #expect(StreamedMoEKVReusePolicy.start(
            newPromptIDs: [1],
            cache: makeCache(kvTokenIDs: cachedIDs),
            conversationID: conversationB,
            systemPrompt: "Be concise.",
            thinkingEnabled: true,
            allowsRewind: true) == .reset)
    }

    @Test func rewindableRunnerResetsWhenNothingIsShared() {
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: [7, 8, 9],
            cache: makeCache(kvTokenIDs: cachedIDs),
            conversationID: conversationA,
            systemPrompt: "Be concise.",
            thinkingEnabled: true,
            allowsRewind: true)
        #expect(start == .reset)
    }

    @Test func trimmedHistoryThatIsNotAPrefixResets() {
        // PromptBuilder.trimmed drops oldest pairs, so the new IDs are a
        // suffix of the old conversation, not a prefix of the cached KV.
        let trimmedPrompt: [Int32] = [3, 4, 5, 6, 7]
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: trimmedPrompt,
            cache: makeCache(kvTokenIDs: cachedIDs),
            conversationID: conversationA,
            systemPrompt: "Be concise.",
            thinkingEnabled: true)
        #expect(start == .reset)
    }

    @Test func systemPromptChangeResets() {
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: continuation,
            cache: makeCache(systemPrompt: "Be concise.", kvTokenIDs: cachedIDs),
            conversationID: conversationA,
            systemPrompt: "Be thorough.",
            thinkingEnabled: true)
        #expect(start == .reset)
    }

    @Test func thinkingEnabledToggleResets() {
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: continuation,
            cache: makeCache(thinkingEnabled: true, kvTokenIDs: cachedIDs),
            conversationID: conversationA,
            systemPrompt: "Be concise.",
            thinkingEnabled: false)
        #expect(start == .reset)
    }

    @Test func turnCloseTokenDivergenceResets() {
        // Same length prefix until a turn-close token differs from the KV.
        let divergedPrompt: [Int32] = [1, 2, 3, 99, 5, 6]
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: divergedPrompt,
            cache: makeCache(kvTokenIDs: cachedIDs),
            conversationID: conversationA,
            systemPrompt: "Be concise.",
            thinkingEnabled: true)
        #expect(start == .reset)
    }

    @Test func nilCacheResets() {
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: continuation,
            cache: nil,
            conversationID: conversationA,
            systemPrompt: "Be concise.",
            thinkingEnabled: true)
        #expect(start == .reset)
    }

    @Test func emptyCacheResets() {
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: continuation,
            cache: makeCache(kvTokenIDs: []),
            conversationID: conversationA,
            systemPrompt: "Be concise.",
            thinkingEnabled: true)
        #expect(start == .reset)
    }

    @Test func firstTurnPromptAloneResets() {
        let firstTurn: [Int32] = [10, 11, 12]
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: firstTurn,
            cache: nil,
            conversationID: conversationA,
            systemPrompt: "Be concise.",
            thinkingEnabled: true)
        #expect(start == .reset)
    }

    @Test func validStrictPrefixContinuationResumesWithCachedCount() {
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: continuation,
            cache: makeCache(kvTokenIDs: cachedIDs),
            conversationID: conversationA,
            systemPrompt: "Be concise.",
            thinkingEnabled: true)
        #expect(start == .resume(cachedPromptTokens: cachedIDs.count))
    }

    @Test func cachedLengthEqualToNewPromptResets() {
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: cachedIDs,
            cache: makeCache(kvTokenIDs: cachedIDs),
            conversationID: conversationA,
            systemPrompt: "Be concise.",
            thinkingEnabled: true)
        #expect(start == .reset)
    }

    @Test func cachedLengthGreaterThanNewPromptResets() {
        let shorterPrompt: [Int32] = [1, 2]
        let start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: shorterPrompt,
            cache: makeCache(kvTokenIDs: cachedIDs),
            conversationID: conversationA,
            systemPrompt: "Be concise.",
            thinkingEnabled: true)
        #expect(start == .reset)
    }

    private func makeCache(
        conversationID: UUID? = nil,
        systemPrompt: String = "Be concise.",
        thinkingEnabled: Bool = true,
        kvTokenIDs: [Int32]
    ) -> StreamedMoEKVCacheState {
        StreamedMoEKVCacheState(
            conversationID: conversationID ?? conversationA,
            systemPrompt: systemPrompt,
            thinkingEnabled: thinkingEnabled,
            kvTokenIDs: kvTokenIDs)
    }
}
