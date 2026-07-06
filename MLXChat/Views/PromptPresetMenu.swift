import SwiftUI

/// Toolbar prompt-preset menu: choose a system-prompt preset, or edit the selected one.
///
/// The toolbar button shows the ACTIVE preset's own glyph (sparkles for General),
/// so the current selection is visible without opening the menu.
struct PromptPresetMenu: View {
    @AppStorage(Keys.promptPreset) private var selectedRaw = PromptPreset.general.rawValue
    /// `.sheet(item:)` gives the editor a fresh identity per preset — a plain
    /// isPresented sheet would reuse the first presentation's @State text.
    @State private var editingPreset: PromptPreset?

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
    }

    var body: some View {
        Menu {
            Section("Prompt") {
                ForEach(PromptPreset.allCases) { preset in
                    Toggle(isOn: Binding(
                        get: { current == preset },
                        set: { isOn in
                            if isOn { choose(preset) }
                        })) {
                            Label(preset.displayName, systemImage: preset.icon)
                        }
                }
            }
            Divider()
            Button {
                editingPreset = current
            } label: {
                Label("Edit \(current.displayName) Prompt…", systemImage: "pencil")
            }
        } label: {
            Image(systemName: current.icon)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(tint(for: current))
                .frame(width: 34, height: 34)
                .background(Circle().fill(Color.sidebarBackground))
                .contentShape(Circle())
        }
        .menuStyle(.button)
        .menuIndicator(.hidden)
        .buttonStyle(.plain)
        .help("Prompt preset: \(current.displayName)")
        .sheet(item: $editingPreset) { preset in
            EditPromptSheet(preset: preset)
        }
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
