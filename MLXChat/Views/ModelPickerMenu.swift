import SwiftUI

/// In-content model picker capsule: Apple Intelligence + downloaded models only.
struct ModelPickerMenu: View {
    @Environment(ChatController.self) private var chat
    /// Pushes the in-frame Models page onto the detail column.
    let onManageModels: () -> Void

    private var downloadedModels: [CatalogModel] {
        chat.modelStore.allModels.filter { chat.modelStore.isDownloaded($0.id) }
    }

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
        @Bindable var chat = chat
        Menu {
            Picker("Model", selection: $chat.selectedModelID) {
                if AppleIntelligenceEngine.isAvailable {
                    Text("Apple Intelligence").tag(appleIntelligenceEngineID)
                }
                Section("Chat") {
                    ForEach(downloadedModels) { model in
                        Text(model.displayName).tag(model.id)
                    }
                }
            }
            .pickerStyle(.inline)

            if let message = AppleIntelligenceEngine.availabilityMessage() {
                Divider()
                Text("Apple Intelligence — \(message)")
            }

            Divider()
            Button {
                onManageModels()
            } label: {
                Label("Download Models…", systemImage: "arrow.down.circle")
            }
        } label: {
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
            .background(Capsule().fill(Color.sidebarBackground))
            .contentShape(Capsule())
        }
        .menuStyle(.button)
        .menuIndicator(.hidden)
        .buttonStyle(.plain)
        .animation(.easeOut(duration: 0.2), value: isLoadingSelected)
    }
}
