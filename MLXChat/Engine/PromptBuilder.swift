import Foundation

enum PromptBuilder {
    static let basePrompt = """
        You are a helpful assistant running fully on-device on the user's Mac. Nothing the user writes ever leaves this device.
        Give only the final answer; keep internal reasoning private. Write the answer as chat text. You may use bold, lists, and short headings in the reply. Do not wrap the whole answer in a markdown, json, xml, or other fence. Use a fenced code block with a language tag only for actual code. Use $...$ for inline math and $$...$$ for display math whenever you write mathematics.
        """

    /// Assemble the system prompt. `presetText` (the selected prompt preset, possibly
    /// user-edited) is prepended when non-empty; nil/empty falls back to `basePrompt`
    /// alone.
    static func systemPrompt(presetText: String? = nil) -> String {
        let preset = presetText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return preset.isEmpty ? basePrompt : preset + "\n\n" + basePrompt
    }

    /// Compose the user-turn text sent to the engine. When a file attachment is
    /// present, appends a `[File: name]` block with the extracted body.
    static func userText(content: String, attachmentName: String?, attachmentText: String?) -> String {
        guard let name = attachmentName, let text = attachmentText else { return content }
        return content + "\n\n[File: \(name)]\n\(text)"
    }

    /// Estimate tokens as ~4 chars each; keep the newest messages within 70% of the
    /// context window, dropping oldest user/assistant pairs. The final (pending) message
    /// is never dropped, even if it alone exceeds the budget.
    static func trimmed(history: [EngineMessage], ctxTokens: Int) -> [EngineMessage] {
        let budget = Int(Double(min(ctxTokens, 131_072)) * 0.7)
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
