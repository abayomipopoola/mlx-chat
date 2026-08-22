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
