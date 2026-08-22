import AppKit
import SwiftUI

// MARK: - Pending attachments

struct PendingAttachmentStrip: View {
    let pendingImagePreview: NSImage?
    let pendingFileName: String?
    let onClearImage: () -> Void
    let onClearFile: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            if let preview = pendingImagePreview {
                ZStack(alignment: .topTrailing) {
                    Image(nsImage: preview)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 44, height: 44)
                        .clipShape(RoundedRectangle(cornerRadius: Studio.radiusField, style: .continuous))
                    clearChipButton { onClearImage() }
                        .offset(x: 6, y: -6)
                }
            }

            if let name = pendingFileName {
                HStack(spacing: 6) {
                    Image(systemName: "doc.fill")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(Color.iconBlue)
                    Text(name)
                        .font(Studio.sans(12.5))
                        .lineLimit(1)
                        .foregroundStyle(.primary)
                    clearChipButton { onClearFile() }
                }
                .padding(.leading, 10)
                .padding(.trailing, 6)
                .padding(.vertical, 6)
                .studioQuietFill(in: Capsule())
                .accessibilityIdentifier("composerAttachmentChip")
            }

            Spacer(minLength: 0)
        }
    }

    private func clearChipButton(action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: "xmark.circle.fill")
                .font(.system(size: 14))
                .symbolRenderingMode(.palette)
                .foregroundStyle(.white, Color.secondary.opacity(0.85))
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Status

struct ComposerStatusRows: View {
    @Environment(ChatController.self) private var chat

    var body: some View {
        switch chat.runtime.effectiveState {
        case .downloading(let id):
            DownloadStatusRow(modelID: id)
        case .loading(let id, let fraction):
            LoadingStatusRow(modelID: id, fraction: fraction)
        case .empty, .ready, .generating:
            EmptyView()
        }

        if let banner = chat.errorBanner {
            ErrorBannerRow(banner: banner)
        }
    }
}

/// Shared chrome for composer status strips.
private func statusRow<Content: View>(@ViewBuilder content: () -> Content) -> some View {
    content()
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .studioQuietFill(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
}

// MARK: - Download status

private struct DownloadStatusRow: View {
    @Environment(ChatController.self) private var chat
    let modelID: String

    var body: some View {
        let model = chat.modelStore.model(for: modelID)
        let fraction = chat.modelStore.downloadProgress[modelID] ?? 0
        statusRow {
            HStack(spacing: 10) {
                ProgressView(value: fraction)
                    .frame(maxWidth: 220)
                Text(downloadLabel(model: model, fraction: fraction))
                    .font(Studio.sans(12))
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Cancel") {
                    chat.modelStore.cancelDownload(modelID)
                    chat.stop()
                }
                .controlSize(.small)
                .tint(Color.brandGreen)
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
}

// MARK: - Loading status

private struct LoadingStatusRow: View {
    @Environment(ChatController.self) private var chat
    let modelID: String
    let fraction: Double

    var body: some View {
        let name = chat.modelStore.model(for: modelID)?.displayName ?? modelID
        statusRow {
            HStack(spacing: 10) {
                ProgressView(value: fraction)
                    .frame(maxWidth: 220)
                Text("Loading \(name)… \(Int(fraction * 100))%")
                    .font(Studio.sans(12))
                    .foregroundStyle(.secondary)
                Spacer()
            }
        }
    }
}

// MARK: - Error banner

private struct ErrorBannerRow: View {
    @Environment(ChatController.self) private var chat
    let banner: String

    var body: some View {
        statusRow {
            HStack(spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                Text(banner)
                    .font(Studio.sans(12))
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
