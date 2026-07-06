import Testing
@testable import MLXChat

/// Contract: PromptBuilder.systemPrompt gates custom-instruction injection on the
/// personalization toggle (clamped to 1,000 chars), and trimmed(history:ctxTokens:)
/// drops oldest pairs while always keeping the final pending message.
@Suite struct PromptTests {
    @Test func enabledInstructionsAreAppendedUnderHeader() {
        let prompt = PromptBuilder.systemPrompt(
            personalizationEnabled: true, customInstructions: "Always answer in French.")
        #expect(prompt == PromptBuilder.basePrompt
            + "\n\n# User's custom instructions\nAlways answer in French.")
    }

    @Test func disabledOrBlankInstructionsYieldBasePrompt() {
        #expect(PromptBuilder.systemPrompt(personalizationEnabled: false, customInstructions: "x")
            == PromptBuilder.basePrompt)
        #expect(PromptBuilder.systemPrompt(personalizationEnabled: true, customInstructions: "   ")
            == PromptBuilder.basePrompt)
    }

    @Test func overlongInstructionsAreClampedTo1000Chars() {
        let prompt = PromptBuilder.systemPrompt(
            personalizationEnabled: true,
            customInstructions: String(repeating: "x", count: 1_500))
        #expect(prompt == PromptBuilder.basePrompt
            + "\n\n# User's custom instructions\n" + String(repeating: "x", count: 1_000))
    }

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

    @Test func singleOversizedMessageIsNeverDropped() {
        let history = [EngineMessage(role: .user, text: String(repeating: "y", count: 200_000))]

        let trimmed = PromptBuilder.trimmed(history: history, ctxTokens: 8_192)

        #expect(trimmed.count == 1)
        #expect(trimmed[0].role == .user)
        #expect(trimmed[0].text == history[0].text)
    }

    @Test func presetTextIsPrependedBeforeBasePrompt() {
        let prompt = PromptBuilder.systemPrompt(
            presetText: "X", personalizationEnabled: false, customInstructions: "")
        #expect(prompt == "X\n\n" + PromptBuilder.basePrompt)
    }

    @Test func presetTextComposesWithInstructionsAppendix() {
        let prompt = PromptBuilder.systemPrompt(
            presetText: "X", personalizationEnabled: true, customInstructions: "Be brief.")
        #expect(prompt == "X\n\n" + PromptBuilder.basePrompt
            + "\n\n# User's custom instructions\nBe brief.")
    }

    @Test func nilOrBlankPresetTextMatchesOldBehavior() {
        #expect(PromptBuilder.systemPrompt(
            presetText: nil, personalizationEnabled: false, customInstructions: "")
            == PromptBuilder.basePrompt)
        #expect(PromptBuilder.systemPrompt(
            presetText: "  \n ", personalizationEnabled: false, customInstructions: "")
            == PromptBuilder.basePrompt)
        // Blank preset + enabled instructions still takes the pre-preset path exactly.
        #expect(PromptBuilder.systemPrompt(
            presetText: "   ", personalizationEnabled: true, customInstructions: "Be brief.")
            == PromptBuilder.basePrompt + "\n\n# User's custom instructions\nBe brief.")
    }
}
