import AppKit
import SwiftUI

// MARK: - Pending attachments

struct PendingAttachmentStrip: View {
    let pendingImagePreview: NSImage?
    /// One chip per pending file, in attach order.
    let pendingFiles: [(name: String, text: String)]
    let onClearImage: () -> Void
    /// Called with the name of the chip that was cleared.
    let onClearFile: (String) -> Void

    var body: some View {
        // Wraps onto additional lines once several files are attached, rather
        // than letting the strip scroll off the edge of the composer.
        FlowLayout(spacing: 8) {
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

            ForEach(pendingFiles, id: \.name) { file in
                HStack(spacing: 6) {
                    Image(systemName: "doc.fill")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(Color.iconBlue)
                    Text(file.name)
                        .font(Studio.sans(12.5))
                        .lineLimit(1)
                        .foregroundStyle(.primary)
                    clearChipButton { onClearFile(file.name) }
                }
                .padding(.leading, 10)
                .padding(.trailing, 6)
                .padding(.vertical, 6)
                .studioQuietFill(in: Capsule())
                .accessibilityIdentifier("composerAttachmentChip")
            }
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

/// Minimal left-to-right wrapping layout for a row of chips that may overflow
/// onto more than one line (SwiftUI has no built-in flow layout pre-macOS 26 HIG
/// components; this keeps the strip simple and dependency-free).
private struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var origin = CGPoint.zero
        var lineHeight: CGFloat = 0
        var totalHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if origin.x + size.width > maxWidth, origin.x > 0 {
                origin.x = 0
                origin.y += lineHeight + spacing
                lineHeight = 0
            }
            origin.x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
            totalHeight = origin.y + lineHeight
        }
        return CGSize(width: maxWidth.isFinite ? maxWidth : origin.x, height: totalHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var origin = CGPoint(x: bounds.minX, y: bounds.minY)
        var lineHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if origin.x + size.width > bounds.maxX, origin.x > bounds.minX {
                origin.x = bounds.minX
                origin.y += lineHeight + spacing
                lineHeight = 0
            }
            subview.place(at: origin, proposal: ProposedViewSize(size))
            origin.x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
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
