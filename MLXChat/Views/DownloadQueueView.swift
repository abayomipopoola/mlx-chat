import SwiftUI

/// Active model downloads (MLX Studio download-queue card). Shown inside
/// Manage Models and as a floating card over the chat while downloads run.
struct DownloadQueueView: View {
    @Environment(ChatController.self) private var chat

    private var downloads: [(id: String, fraction: Double)] {
        chat.modelStore.downloadProgress
            .sorted { $0.key < $1.key }
            .map { (id: $0.key, fraction: $0.value) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionHeader("Downloads")
            ForEach(downloads, id: \.id) { download in
                DownloadItemRow(id: download.id, fraction: download.fraction)
            }
        }
        .studioCard()
    }
}

private struct DownloadItemRow: View {
    @Environment(ChatController.self) private var chat
    let id: String
    let fraction: Double

    private var model: CatalogModel? { chat.modelStore.model(for: id) }

    private var caption: String {
        let percent = "\(Int(fraction * 100))%"
        if let size = model?.sizeGB, size > 0 {
            return "\(percent) • \(String(format: "%.1f", size)) GB"
        }
        return percent
    }

    var body: some View {
        HStack(spacing: 10) {
            TintedCircleIcon(systemName: "arrow.down", size: 30)
            VStack(alignment: .leading, spacing: 4) {
                Text(model?.displayName ?? id)
                    .font(Studio.body14Semibold)
                    .lineLimit(1)
                    .truncationMode(.middle)
                ProgressView(value: fraction)
                    .tint(Color.brandGreen)
                Text(caption)
                    .font(Studio.caption13)
                    .foregroundStyle(Color.subtitleGray)
            }
            Button {
                chat.modelStore.cancelDownload(id)
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 16))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Cancel download")
        }
    }
}
