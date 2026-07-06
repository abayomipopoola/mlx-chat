import SwiftUI

/// Bottom input area: engine status rows, error banner, growing text field, send/stop.
struct ComposerView: View {
    @Environment(ChatController.self) private var chat
    @Binding var draft: String
    @FocusState.Binding var focused: Bool
    let onSend: (String) -> Void

    private var canSend: Bool {
        !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !chat.isStreaming
    }

    var body: some View {
        VStack(spacing: 10) {
            statusRows

            HStack(alignment: .bottom, spacing: 10) {
                TextField("What can I help you with?", text: $draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...8)
                    .font(Studio.composerText)
                    .padding(.vertical, 11)
                    .focused($focused)
                    .onKeyPress(.return, phases: .down) { press in
                        if press.modifiers.contains(.shift) {
                            draft += "\n"
                            return .handled
                        }
                        submit()
                        return .handled
                    }

                Button {
                    if chat.isStreaming {
                        chat.stop()
                    } else {
                        submit()
                    }
                } label: {
                    Image(systemName: chat.isStreaming ? "stop.fill" : "arrow.up")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(chat.isStreaming || canSend ? .white : Color.secondary)
                        .frame(width: Studio.sendButtonSize, height: Studio.sendButtonSize)
                        .background(
                            Circle().fill(
                                chat.isStreaming || canSend
                                    ? Color.sendGreen
                                    : Color.cardStroke.opacity(0.6)))
                }
                .buttonStyle(.plain)
                .disabled(!chat.isStreaming && !canSend)
            }
            .padding(.leading, 24)
            .padding(.trailing, 6)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: Studio.radiusComposer, style: .continuous)
                    .fill(Color.fieldFill))

            Text("Runs 100% on-device • Private & offline")
                .font(.system(size: 11.5))
                .foregroundStyle(Color.subtitleGray)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 40)
        .padding(.bottom, 12)
        .padding(.top, 6)
    }

    private func submit() {
        guard canSend else { return }
        let text = draft
        draft = ""
        onSend(text)
    }

    // MARK: - Status

    @ViewBuilder
    private var statusRows: some View {
        switch chat.runtime.state {
        case .downloading(let id):
            let model = chat.modelStore.model(for: id)
            let fraction = chat.modelStore.downloadProgress[id] ?? 0
            statusRow {
                HStack(spacing: 10) {
                    ProgressView(value: fraction)
                        .frame(maxWidth: 220)
                    Text(downloadLabel(model: model, fraction: fraction))
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Cancel") {
                        chat.modelStore.cancelDownload(id)
                        chat.stop()
                    }
                    .controlSize(.small)
                    .tint(Color.brandGreen)
                }
            }
        case .loading(let id, let fraction):
            let name = chat.modelStore.model(for: id)?.displayName ?? id
            statusRow {
                HStack(spacing: 10) {
                    ProgressView(value: fraction)
                        .frame(maxWidth: 220)
                    Text("Loading \(name)… \(Int(fraction * 100))%")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
            }
        case .empty, .ready, .generating:
            EmptyView()
        }

        if let banner = chat.errorBanner {
            statusRow {
                HStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(banner)
                        .font(.callout)
                        .lineLimit(2)
                    Spacer()
                    if chat.canRetry {
                        Button("Retry") { chat.retry() }
                            .controlSize(.small)
                            .tint(Color.brandGreen)
                    }
                    Button {
                        chat.errorBanner = nil
                    } label: {
                        Image(systemName: "xmark")
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func downloadLabel(model: CatalogModel?, fraction: Double) -> String {
        let name = model?.displayName ?? "model"
        if let size = model?.sizeGB, size > 0 {
            return "Downloading \(name) (\(String(format: "%.1f", size)) GB)… \(Int(fraction * 100))%"
        }
        return "Downloading \(name)… \(Int(fraction * 100))%"
    }

    private func statusRow(@ViewBuilder content: () -> some View) -> some View {
        content()
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(Color.fieldFill)
            .clipShape(RoundedRectangle(cornerRadius: 14))
    }
}
