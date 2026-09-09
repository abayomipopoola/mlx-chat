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

    /// Compose the user-turn text sent to the engine. Each attachment is appended
    /// as its own `[File: name]` block with the extracted body, in the order given.
    static func userText(content: String, attachments: [(name: String, text: String)]) -> String {
        guard !attachments.isEmpty else { return content }
        let blocks = attachments.map { "[File: \($0.name)]\n\($0.text)" }
        return ([content] + blocks).joined(separator: "\n\n")
    }

    /// Backward-compatible single-attachment overload.
    static func userText(content: String, attachmentName: String?, attachmentText: String?) -> String {
        guard let name = attachmentName, let text = attachmentText else { return content }
        return userText(content: content, attachments: [(name: name, text: text)])
    }

    /// Estimate tokens as ~4 chars each. Used both for history trimming and for the
    /// pre-flight "will this fit?" check the composer runs before a file is attached
    /// or a message is sent.
    static func estimatedTokens(_ text: String) -> Int { (text.count + 3) / 4 }

    /// The usable token budget for a given model context size: 70% of the window
    /// (capped at 131,072), leaving headroom for the reply and template overhead.
    static func budget(ctxTokens: Int) -> Int {
        Int(Double(min(ctxTokens, 131_072)) * 0.7)
    }

    /// Keep the newest messages within `budget(ctxTokens:)`, dropping oldest
    /// user/assistant pairs. The final (pending) message is never dropped, even if
    /// it alone exceeds the budget — callers that attach files should check
    /// `attachmentsExceedBudget` before sending so that case doesn't happen silently.
    static func trimmed(history: [EngineMessage], ctxTokens: Int) -> [EngineMessage] {
        let budget = budget(ctxTokens: ctxTokens)
        func tokens(_ message: EngineMessage) -> Int { estimatedTokens(message.text) }

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

    /// Pre-flight check for the composer: does the draft text plus every pending
    /// attachment already exceed the model's usable budget, on its own, before any
    /// prior conversation history is even considered? If so, sending would either
    /// get silently truncated by the model or fail outright — the composer should
    /// warn the user and let them remove a file instead of finding out after the
    /// call to the API.
    static func attachmentsExceedBudget(
        draft: String,
        attachments: [(name: String, text: String)],
        ctxTokens: Int
    ) -> Bool {
        estimatedPendingTurnTokens(draft: draft, attachments: attachments) > budget(ctxTokens: ctxTokens)
    }

    static func estimatedPendingTurnTokens(
        draft: String,
        attachments: [(name: String, text: String)]
    ) -> Int {
        let draftTokens = estimatedTokens(draft)
        let attachmentTokens = attachments.reduce(0) { $0 + estimatedTokens($1.text) }
        return draftTokens + attachmentTokens
    }
}
