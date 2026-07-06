import Foundation
import Testing
@testable import MLXChat

/// Contract: PromptPresets persists per-preset override text in UserDefaults —
/// saving the preset's default text or emptiness CLEARS the override — and
/// effectiveText resolves override ?? defaultText. Serialized because every test
/// shares UserDefaults.standard.
@Suite(.serialized)
struct PromptPresetTests {
    /// Snapshots the touched defaults keys, clears them for a clean slate, runs the
    /// body, then restores the originals so the app's real settings are untouched.
    private func withCleanDefaults(
        presets: [PromptPreset] = [], selection: Bool = false, _ body: () -> Void
    ) {
        let defaults = UserDefaults.standard
        var keys = presets.map { "prompt.override." + $0.rawValue }
        if selection { keys.append(Keys.promptPreset) }
        let saved = keys.map { ($0, defaults.object(forKey: $0)) }
        defer {
            for (key, value) in saved {
                if let value {
                    defaults.set(value, forKey: key)
                } else {
                    defaults.removeObject(forKey: key)
                }
            }
        }
        for key in keys { defaults.removeObject(forKey: key) }
        body()
    }

    @Test func overrideRoundTripsAndWinsOverDefault() {
        withCleanDefaults(presets: [.coding]) {
            #expect(PromptPresets.overrideText(for: .coding) == nil)
            PromptPresets.setOverride("My coding rules", for: .coding)
            #expect(PromptPresets.overrideText(for: .coding) == "My coding rules")
            #expect(PromptPresets.effectiveText(for: .coding) == "My coding rules")
        }
    }

    @Test func savingDefaultTextClearsOverride() {
        withCleanDefaults(presets: [.creative]) {
            PromptPresets.setOverride("Edited", for: .creative)
            PromptPresets.setOverride(PromptPreset.creative.defaultText, for: .creative)
            #expect(PromptPresets.overrideText(for: .creative) == nil)
            #expect(PromptPresets.effectiveText(for: .creative)
                == PromptPreset.creative.defaultText)
        }
    }

    @Test func savingEmptyOrWhitespaceClearsOverride() {
        withCleanDefaults(presets: [.reasoning]) {
            PromptPresets.setOverride("Edited", for: .reasoning)
            PromptPresets.setOverride("", for: .reasoning)
            #expect(PromptPresets.overrideText(for: .reasoning) == nil)

            PromptPresets.setOverride("Edited again", for: .reasoning)
            PromptPresets.setOverride("  \n  ", for: .reasoning)
            #expect(PromptPresets.overrideText(for: .reasoning) == nil)
        }
    }

    @Test func effectiveTextFallsBackToDefaultText() {
        withCleanDefaults(presets: [.roleplay]) {
            #expect(PromptPresets.effectiveText(for: .roleplay)
                == PromptPreset.roleplay.defaultText)
            PromptPresets.setOverride("In-scene persona", for: .roleplay)
            #expect(PromptPresets.effectiveText(for: .roleplay) == "In-scene persona")
        }
    }

    @Test func selectedRoundTripsThroughDefaults() {
        withCleanDefaults(selection: true) {
            PromptPresets.selected = .coding
            #expect(PromptPresets.selected == .coding)
            PromptPresets.selected = .custom
            #expect(PromptPresets.selected == .custom)
        }
    }
}
