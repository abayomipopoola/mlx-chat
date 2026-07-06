import Foundation

/// System-prompt presets (MLX Studio-style sparkle menu). The selected preset's
/// effective text is prepended to the base prompt for every conversation.
enum PromptPreset: String, CaseIterable, Identifiable {
    case general, creative, roleplay, coding, reasoning, custom

    var id: String { rawValue }

    var displayName: String { rawValue.capitalized }

    var icon: String {
        switch self {
        case .general: return "sparkles"
        case .creative: return "paintpalette"
        case .roleplay: return "theatermasks"
        case .coding: return "chevron.left.forwardslash.chevron.right"
        case .reasoning: return "brain"
        case .custom: return "pencil.circle"
        }
    }

    var defaultText: String {
        switch self {
        case .general:
            return """
                You are a helpful AI assistant. Be concise and direct.
                - Use markdown formatting for better readability
                - For code, use appropriate syntax highlighting
                - Stay focused on the user's question
                """
        case .creative:
            return """
                You are a creative writing assistant with a gift for storytelling and imagination.
                - Write with vivid descriptions and engaging prose
                - Develop compelling characters with distinct voices
                - Build immersive worlds and atmospheres
                - Embrace creative freedom while maintaining narrative coherence
                - Use varied sentence structures and literary techniques
                """
        case .roleplay:
            return """
                You are a roleplay partner skilled in interactive storytelling.
                - Stay in character consistently throughout the conversation
                - React naturally to the user's actions and dialogue
                - Describe actions, emotions, and surroundings vividly
                - Allow the story to develop organically based on user choices
                - Use *asterisks* for actions and regular text for dialogue
                """
        case .coding:
            return """
                You are an expert programming assistant.
                - Write clean, efficient, and well-documented code
                - Follow best practices and design patterns
                - Explain your reasoning when helpful
                - Consider edge cases and error handling
                - Use appropriate syntax highlighting in code blocks
                """
        case .reasoning:
            return """
                You are a logical reasoning assistant skilled in analysis and problem-solving.
                - Break down complex problems into clear steps
                - Show your work and explain your thought process
                - Consider multiple approaches when applicable
                - Verify your conclusions with careful reasoning
                - Be precise with mathematical and logical notation
                """
        case .custom:
            return "You are a helpful AI assistant."
        }
    }
}

enum PromptPresets {
    private static func overrideKey(for preset: PromptPreset) -> String {
        // The Custom preset IS the Settings page's custom instructions —
        // one storage, so editing either surface updates both.
        preset == .custom ? Keys.personalizationInstructions : "prompt.override." + preset.rawValue
    }

    static var selected: PromptPreset {
        get {
            let raw = UserDefaults.standard.string(forKey: Keys.promptPreset) ?? PromptPreset.general.rawValue
            return PromptPreset(rawValue: raw) ?? .general
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: Keys.promptPreset)
            UserDefaults.standard.synchronize()
        }
    }

    static func overrideText(for preset: PromptPreset) -> String? {
        guard let text = UserDefaults.standard.string(forKey: overrideKey(for: preset)),
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }
        return text
    }

    /// Persist an edited prompt; saving the default (or emptiness for non-custom) clears the override.
    static func setOverride(_ text: String, for preset: PromptPreset) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed == preset.defaultText {
            UserDefaults.standard.removeObject(forKey: overrideKey(for: preset))
        } else {
            UserDefaults.standard.set(text, forKey: overrideKey(for: preset))
        }
        UserDefaults.standard.synchronize()
    }

    static func effectiveText(for preset: PromptPreset) -> String {
        overrideText(for: preset) ?? preset.defaultText
    }
}
