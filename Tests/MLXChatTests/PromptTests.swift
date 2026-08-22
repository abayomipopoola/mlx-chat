import Testing
@testable import MLXChat

/// Contract: PromptBuilder.systemPrompt prepends a non-empty preset before
/// basePrompt (nil/blank yields basePrompt alone), and trimmed(history:ctxTokens:)
/// drops oldest pairs while always keeping the final pending message.
@Suite struct PromptTests {
    @Test func trimDropsOldestPairsAndKeepsSuffix() {
        let history = (0..<40).map { index in
            EngineMessage(
                role: index.isMultiple(of: 2) ? .user : .assistant,
                text: String(repeating: "abcd", count: 1_000))  // ~1000 estimated tokens each
        }

        let trimmed = PromptBuilder.trimmed(history: history, ctxTokens: 8_192)

        #expect(trimmed.count < history.count)
        #expect(!trimmed.isEmpty)
        // The final pending message survives verbatim.
        #expect(trimmed.last?.role == history.last?.role)
        #expect(trimmed.last?.text == history.last?.text)
        // The result is a suffix of the input: only oldest messages were dropped.
        for (kept, original) in zip(trimmed, history.suffix(trimmed.count)) {
            #expect(kept.role == original.role)
            #expect(kept.text == original.text)
        }
    }

    @Test func fortyThousandContextKeepsMoreHistoryThanA32kCap() {
        // Each message ≈ 100 tokens (400 chars). 250 messages ≈ 25_000 tokens —
        // over a 32_768-capped 70% budget (~22_937) and under a 40k 70% budget (28_000).
        let history = (0..<250).map { index in
            EngineMessage(
                role: index.isMultiple(of: 2) ? .user : .assistant,
                text: String(repeating: "abcd", count: 100))
        }
        let trimmed = PromptBuilder.trimmed(history: history, ctxTokens: 40_000)
        func tokens(_ message: EngineMessage) -> Int { (message.text.count + 3) / 4 }
        let kept = trimmed.reduce(0) { $0 + tokens($1) }
        let oldCapBudget = Int(Double(min(40_000, 32_768)) * 0.7)

        #expect(trimmed.count == history.count)
        #expect(kept > oldCapBudget)
        #expect(trimmed.last?.text == history.last?.text)
    }

    @Test func singleOversizedMessageIsNeverDropped() {
        let history = [EngineMessage(role: .user, text: String(repeating: "y", count: 200_000))]

        let trimmed = PromptBuilder.trimmed(history: history, ctxTokens: 8_192)

        #expect(trimmed.count == 1)
        #expect(trimmed[0].role == .user)
        #expect(trimmed[0].text == history[0].text)
    }

    @Test func presetTextIsPrependedBeforeBasePrompt() {
        let prompt = PromptBuilder.systemPrompt(presetText: "X")
        #expect(prompt == "X\n\n" + PromptBuilder.basePrompt)
    }

    @Test func nilOrBlankPresetTextMatchesOldBehavior() {
        #expect(PromptBuilder.systemPrompt(presetText: nil)
            == PromptBuilder.basePrompt)
        #expect(PromptBuilder.systemPrompt(presetText: "  \n ")
            == PromptBuilder.basePrompt)
    }

    @Test func basePromptRequestsPlainFinalAnswersByDefault() {
        #expect(PromptBuilder.basePrompt.contains("keep internal reasoning private"))
        #expect(PromptBuilder.basePrompt.contains("Write the answer as chat text"))
        #expect(PromptBuilder.basePrompt.contains("Do not wrap the whole answer in a markdown"))
    }

    @Test func appleIntelligencePromptForbidsAMarkdownFenceAroundTheReply() {
        let prompt = AppleIntelligenceEngine.chatStylePrompt("Base rules.")
        #expect(prompt.hasPrefix("Base rules.\n\n"))
        #expect(prompt.contains("Write the reply as chat text"))
        #expect(prompt.contains("Do not wrap the whole answer in a markdown"))
    }
}
