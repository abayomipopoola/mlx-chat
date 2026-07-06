import Foundation

enum PromptBuilder {
    static let basePrompt = """
        You are a helpful assistant running fully on-device on the user's Mac. Nothing the user writes ever leaves this device.
        Format your responses in Markdown. Use fenced code blocks with a language tag for code. Use $...$ for inline math and $$...$$ for display math whenever you write mathematics.
        """

    /// Assemble the system prompt. `presetText` (the selected prompt preset, possibly
    /// user-edited) is prepended when non-empty; nil/empty falls back to `basePrompt`
    /// alone. Custom instructions are appended only when personalization is enabled.
    static func systemPrompt(
        presetText: String? = nil,
        personalizationEnabled: Bool,
        customInstructions: String
    ) -> String {
        let preset = presetText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let core = preset.isEmpty ? basePrompt : preset + "\n\n" + basePrompt
        let instructions = customInstructions.trimmingCharacters(in: .whitespacesAndNewlines)
        guard personalizationEnabled, !instructions.isEmpty else { return core }
        return core + "\n\n# User's custom instructions\n" + String(instructions.prefix(1000))
    }

    /// Estimate tokens as ~4 chars each; keep the newest messages within 70% of the
    /// context window, dropping oldest user/assistant pairs. The final (pending) message
    /// is never dropped, even if it alone exceeds the budget.
    static func trimmed(history: [EngineMessage], ctxTokens: Int) -> [EngineMessage] {
        let budget = Int(Double(min(ctxTokens, 32_768)) * 0.7)
        func tokens(_ message: EngineMessage) -> Int { (message.text.count + 3) / 4 }

        var total = history.reduce(0) { $0 + tokens($1) }
        guard total > budget, history.count > 1 else { return history }

        var result = history
        // Drop from the front in pairs while over budget, always keeping the last message.
        while total > budget, result.count > 1 {
            let dropCount = result.count > 2 ? 2 : 1
            for _ in 0..<dropCount where result.count > 1 {
                total -= tokens(result.removeFirst())
            }
        }
        return result
    }
}
