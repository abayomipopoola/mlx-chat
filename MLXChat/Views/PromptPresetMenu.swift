import SwiftUI

/// Toolbar prompt-preset picker: choose a system-prompt preset, or edit the
/// selected one. Toggles the root-hosted prompt dropdown (see WindowDropdown);
/// the panel lives at the window root because the header's `.safeAreaInset`
/// bar clips its content to the bar region.
///
/// The toolbar button shows the ACTIVE preset's own glyph (sparkles for General),
/// so the current selection is visible without opening the dropdown.
struct PromptPresetMenu: View {
    @AppStorage(Keys.promptPreset) private var selectedRaw = PromptPreset.general.rawValue
    @Environment(HeaderDropdown.self) private var dropdown

    private var current: PromptPreset {
        PromptPreset(rawValue: selectedRaw) ?? .general
    }

    /// Per-preset icon color (MLX Studio's palette).
    private func tint(for preset: PromptPreset) -> Color {
        switch preset {
        case .general: return .iconBlue
        case .creative: return .pink
        case .roleplay: return .iconPurple
        case .coding: return .brandGreen
        case .reasoning: return .iconOrange
        case .custom: return .secondary
        }
    }

    var body: some View {
        Button {
            dropdown.toggle(.promptPresets)
        } label: {
            Image(systemName: current.icon)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(tint(for: current))
                .frame(width: 34, height: 34)
                .glassEffect(.regular.interactive(), in: Circle())
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("promptPresetButton")
        .help("Prompt preset: \(current.displayName)")
        .dropdownAnchor(.promptPresets, in: dropdown)
        .onAppear {
            // UI-verification hooks: open the dropdown (or the prompt editor)
            // at launch for screenshots.
            let arguments = ProcessInfo.processInfo.arguments
            if arguments.contains("--open-prompt-dropdown") {
                dropdown.open = .promptPresets
            }
            if arguments.contains("--open-prompt-editor") {
                dropdown.editingPrompt = current
            }
        }
    }
}

/// The root-hosted dropdown panel: preset rows with a leading checkmark slot,
/// plus the editor entry point. The editor sheet is NOT hosted here — closing
/// the dropdown removes this panel, which would tear the sheet down with it;
/// it is presented from the window root via `HeaderDropdown.editingPrompt`.
struct PromptPresetPanel: View {
    @AppStorage(Keys.promptPreset) private var selectedRaw = PromptPreset.general.rawValue
    @Environment(HeaderDropdown.self) private var dropdown

    private var current: PromptPreset {
        PromptPreset(rawValue: selectedRaw) ?? .general
    }

    /// Per-preset icon color (MLX Studio's palette).
    private func tint(for preset: PromptPreset) -> Color {
        switch preset {
        case .general: return .iconBlue
        case .creative: return .pink
        case .roleplay: return .iconPurple
        case .coding: return .brandGreen
        case .reasoning: return .iconOrange
        case .custom: return .secondary
        }
    }

    private func choose(_ preset: PromptPreset) {
        selectedRaw = preset.rawValue
        UserDefaults.standard.set(preset.rawValue, forKey: Keys.promptPreset)
        UserDefaults.standard.synchronize()
        dropdown.open = nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            // Top title, matching the model picker's "Model" label.
            Text("Prompt")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .padding(.leading, 20)
                .padding(.bottom, 2)
            ForEach(PromptPreset.allCases) { preset in
                presetRow(preset)
            }
            Divider()
                .padding(.vertical, 2)
            Button {
                dropdown.editingPrompt = current
                dropdown.open = nil
            } label: {
                Label("Edit \(current.displayName) Prompt…", systemImage: "pencil")
                    .font(.system(size: 13))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(.vertical, 3)
        }
        .padding(8)
    }

    private func presetRow(_ preset: PromptPreset) -> some View {
        HStack(spacing: 6) {
            // Checkmark slot, reserved for every row so names stay aligned
            // (native-menu convention, same as the model picker).
            Image(systemName: "checkmark")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.primary)
                .opacity(current == preset ? 1 : 0)
                .frame(width: 14)

            Button {
                choose(preset)
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: preset.icon)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(tint(for: preset))
                        .frame(width: 16)
                    Text(preset.displayName)
                        .font(.system(size: 13))
                        .lineLimit(1)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
        .padding(.vertical, 3)
    }
}

/// MLX Studio-style prompt editor sheet.
struct EditPromptSheet: View {
    let preset: PromptPreset
    @Environment(\.dismiss) private var dismiss
    @State private var text: String
    /// Global sampling temperature — applies to every preset.
    @AppStorage(Keys.personalizationTemperature) private var temperature = Keys.Defaults.personalizationTemperature

    private static let temperaturePresets: [(label: String, value: Double)] = [
        ("Precise – 0.1", 0.1),
        ("Balanced – 0.4", 0.4),
        ("Creative – 0.7", 0.7),
        ("Wild – 1.0", 1.0),
    ]

    init(preset: PromptPreset) {
        self.preset = preset
        _text = State(initialValue: PromptPresets.effectiveText(for: preset))
    }

    var body: some View {
        VStack(spacing: 0) {
            Text("Edit \(preset.displayName) Prompt")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)

            Divider()

            VStack(alignment: .leading, spacing: 6) {
                Text("\(preset.displayName) Prompt")
                    .font(.title3.bold())
                Text(preset == .custom
                    ? "Write your own prompt. It is sent at the start of every conversation to set the AI's behavior and personality."
                    : "This prompt is sent at the start of every conversation to set the AI's behavior and personality.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color.sidebarBackground.opacity(0.65))

            TextEditor(text: $text)
                .font(.system(size: 13, design: .monospaced))
                .scrollContentBackground(.hidden)
                .padding(12)
                .frame(minHeight: 240)
                .background(Color.cardFill)

            Divider()

            HStack {
                Text("Temperature")
                    .font(.callout.weight(.semibold))
                Text("Applies to all prompts")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Picker("", selection: $temperature) {
                    ForEach(Self.temperaturePresets, id: \.value) { item in
                        Text(item.label).tag(item.value)
                    }
                }
                .labelsHidden()
                .fixedSize()
                .tint(Color.brandGreen)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()

            HStack {
                Text("\(text.count) characters")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Reset to Default") { text = preset.defaultText }
                    .buttonStyle(.bordered)
                    .tint(Color.brandGreen)
                    .disabled(text == preset.defaultText)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Save") {
                    PromptPresets.setOverride(text, for: preset)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
                .tint(Color.brandGreen)
            }
            .padding(14)
        }
        .frame(width: 640, height: 580)
    }
}
