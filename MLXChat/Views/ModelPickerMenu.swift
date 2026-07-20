import SwiftUI

/// In-content model picker capsule: Apple Intelligence + downloaded models only.
/// Toggles the root-hosted model dropdown (see WindowDropdown): the panel can
/// not live in a per-button overlay because the header's `.safeAreaInset` bar
/// clips its content to the bar region.
struct ModelPickerMenu: View {
    @Environment(ChatController.self) private var chat
    @Environment(HeaderDropdown.self) private var dropdown

    /// True while the currently selected model is being pulled into RAM.
    private var isLoadingSelected: Bool {
        switch chat.runtime.state {
        case .loading(let id, _), .downloading(let id):
            return id == chat.selectedModelID
        default:
            return false
        }
    }

    var body: some View {
        Button {
            dropdown.toggle(.modelPicker)
        } label: {
            capsuleLabel
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("modelPickerButton")
        .dropdownAnchor(.modelPicker, in: dropdown)
        .onAppear {
            // UI-verification hook: open the dropdown at launch for screenshots.
            if ProcessInfo.processInfo.arguments.contains("--open-model-dropdown") {
                dropdown.open = .modelPicker
            }
        }
    }

    // MARK: - Capsule

    private var capsuleLabel: some View {
        HStack(spacing: 6) {
            if isLoadingSelected {
                ProgressView()
                    .controlSize(.small)
                    .scaleEffect(0.7)
            }
            Text(chat.selectedModelDisplayName)
                .font(.system(size: 13, weight: .semibold))
            Image(systemName: "chevron.down")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
        }
        .opacity(isLoadingSelected ? 0.5 : 1)
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .glassEffect(.regular.interactive(), in: Capsule())
        .contentShape(Capsule())
        .animation(.easeOut(duration: 0.2), value: isLoadingSelected)
    }
}

/// The root-hosted dropdown panel: native-menu-styled model rows with a
/// leading checkmark slot and an eject button on the resident model.
struct ModelPickerPanel: View {
    @Environment(ChatController.self) private var chat
    @Environment(HeaderDropdown.self) private var dropdown
    /// Pushes the in-frame Models page onto the detail column.
    let onManageModels: () -> Void

    private var downloadedModels: [CatalogModel] {
        chat.modelStore.allModels.filter {
            chat.modelStore.isDownloaded($0.id)
                && chat.modelStore.unsupportedReason(for: $0.id) == nil
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            // Top title ("Select model"), in the same 11pt style as the
            // "Models" section header below.
            Text("Select model")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .padding(.leading, 20)
                .padding(.bottom, 2)
            if AppleIntelligenceEngine.isAvailable {
                modelRow(id: appleIntelligenceEngineID, name: "Apple Intelligence", resident: false)
            }

            if !downloadedModels.isEmpty {
                if AppleIntelligenceEngine.isAvailable {
                    // Separator under the Apple Intelligence row, as the old
                    // AppKit menu drew between the plain item and the section;
                    // inset so it starts at the row text.
                    Divider()
                        .padding(.vertical, 2)
                        .padding(.leading, 20)
                }
                // Native-menu-style section header: 11pt semibold gray (the
                // exact size of the old AppKit menu's "Model"/"Chat" labels,
                // measured from the 1.0.2 menu), aligned with the row text.
                Text("Models")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(.leading, 20)
                    .padding(.top, 4)
                ForEach(downloadedModels) { model in
                    modelRow(
                        id: model.id,
                        name: model.displayName,
                        resident: chat.runtime.loadedModelID == model.id)
                }
            }

            if let message = AppleIntelligenceEngine.availabilityMessage() {
                Divider()
                Text("Apple Intelligence — \(message)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Divider()
                .padding(.vertical, 2)
            Button {
                dropdown.open = nil
                onManageModels()
            } label: {
                Label("Download Models…", systemImage: "arrow.down.circle")
                    .font(.system(size: 13))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(.vertical, 3)
        }
        .padding(8)
    }

    private func modelRow(id: String, name: String, resident: Bool) -> some View {
        HStack(spacing: 6) {
            // Leading checkmark slot (native-menu convention); reserved for
            // every row so names stay aligned. 10pt matches the old AppKit
            // menu's check glyph, which is drawn in the label color.
            Image(systemName: "checkmark")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.primary)
                .opacity(chat.selectedModelID == id ? 1 : 0)
                .frame(width: 14)

            Button {
                chat.selectedModelID = id
                dropdown.open = nil
            } label: {
                Text(name)
                    .font(.system(size: 13))
                    .lineLimit(1)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("modelRow.\(id)")

            if resident {
                Button {
                    eject()
                } label: {
                    Image(systemName: "eject.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .frame(width: 20, height: 20)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(chat.isStreaming)
                .help("Eject from memory and switch to Apple Intelligence")
            }
        }
        .padding(.vertical, 3)
    }

    /// Frees the resident MLX model and falls back to Apple Intelligence.
    /// (The selection didSet already unloads via warmSwitchedModel;
    /// unloadCurrent is the idempotent guarantee.)
    private func eject() {
        guard !chat.isStreaming else { return }
        chat.selectedModelID = appleIntelligenceEngineID
        chat.runtime.unloadCurrent()
        dropdown.open = nil
    }
}
