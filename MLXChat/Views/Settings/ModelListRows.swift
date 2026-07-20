import SwiftUI

// MARK: - Row

struct ModelRow: View {
    @Environment(ChatController.self) private var chat
    let model: CatalogModel
    let onDelete: () -> Void

    private var store: ModelStore { chat.modelStore }

    /// Non-nil when the on-disk model can't run in this build.
    private var unsupportedReason: String? {
        guard store.isDownloaded(model.id) else { return nil }
        return store.unsupportedReason(for: model.id)
    }

    var body: some View {
        // Snapshot per-model store values once so a progress tick for another
        // model does not re-read the shared dictionary inside the control.
        let isDownloaded = store.isDownloaded(model.id)
        let reason = unsupportedReason
        let progress = store.downloadProgress[model.id]

        HStack(spacing: 12) {
            TintedCircleIcon(
                systemName: model.category == .coding
                    ? "chevron.left.forwardslash.chevron.right"
                    : model.category == .reasoning ? "brain" : "message")

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(model.displayName).font(Studio.rowTitle)
                    if model.sizeGB > 0 {
                        Text(String(format: "%.1f GB", model.sizeGB))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    CapsuleBadge(text: model.quantLabel, tint: .iconBlue)
                    CapsuleBadge(text: "\(model.ctxTokens / 1024)K ctx", tint: .iconPurple)
                }
                Text(model.blurb)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                if let reason {
                    Text(reason)
                        .font(.caption)
                        .foregroundStyle(Color.iconOrange)
                        .lineLimit(2)
                }
                Link(destination: model.hugginFaceURL) {
                    Label("View on HuggingFace", systemImage: "link")
                        .font(.caption)
                }
                .foregroundStyle(Color.brandGreen)
            }

            Spacer()

            ModelDownloadControl(
                modelID: model.id,
                isDownloaded: isDownloaded,
                unsupportedReason: reason,
                progress: progress,
                onDelete: onDelete)
        }
        .padding(14)
        .background(Color.cardFill)
        .overlay(RoundedRectangle(cornerRadius: Studio.radiusCard).stroke(Color.cardStroke, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: Studio.radiusCard))
    }
}

/// Trailing download / progress / delete control for a model row.
/// Receives precomputed store values so it does not re-read the shared
/// `downloadProgress` dictionary on every body evaluation.
private struct ModelDownloadControl: View {
    @Environment(ChatController.self) private var chat
    let modelID: String
    let isDownloaded: Bool
    let unsupportedReason: String?
    let progress: Double?
    let onDelete: () -> Void

    private var store: ModelStore { chat.modelStore }

    var body: some View {
        if let fraction = progress {
            Button {
                store.cancelDownload(modelID)
            } label: {
                ZStack {
                    Circle()
                        .stroke(Color.brandGreen.opacity(0.25), lineWidth: 3)
                    Circle()
                        .trim(from: 0, to: max(0.02, fraction))
                        .stroke(Color.brandGreen, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                    Text("\(Int(fraction * 100))")
                        .font(.system(size: 9, weight: .bold))
                }
                .frame(width: 32, height: 32)
            }
            .buttonStyle(.plain)
            .help("Cancel download")
        } else if isDownloaded {
            HStack(spacing: 8) {
                if unsupportedReason != nil {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.title2)
                        .foregroundStyle(Color.iconOrange)
                        .help("This model can't run in this build")
                } else {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.title2)
                        .foregroundStyle(Color.brandGreen)
                }
                Button(role: .destructive, action: onDelete) {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Delete download")
            }
        } else {
            Button {
                store.download(modelID)
            } label: {
                Image(systemName: "arrow.down.circle")
                    .font(.title2)
                    .foregroundStyle(Color.brandGreen)
            }
            .buttonStyle(.plain)
            .help("Download")
        }
    }
}

// MARK: - Family card

struct FamilyCard: View {
    let family: ModelFamily
    let modelCount: Int
    let action: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                TintedCircleIcon(systemName: family.icon)
                VStack(alignment: .leading, spacing: 4) {
                    Text(family.name)
                        .font(Studio.rowTitle)
                    Text(family.blurb)
                        .font(Studio.body13)
                        .foregroundStyle(Color.subtitleGray)
                        .fixedSize(horizontal: false, vertical: true)
                    Text("\(modelCount) model\(modelCount == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if !family.badges.isEmpty {
                        HStack(spacing: 6) {
                            ForEach(family.badges, id: \.self) { badge in
                                FamilyBadgeChip(badge: badge)
                            }
                        }
                        .padding(.top, 2)
                    }
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.tertiary)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(hovering ? Color.fieldFill : Color.cardFill)
            .overlay(RoundedRectangle(cornerRadius: Studio.radiusCard).stroke(Color.cardStroke, lineWidth: 1))
            .clipShape(RoundedRectangle(cornerRadius: Studio.radiusCard))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
    }
}

struct FamilyBadgeChip: View {
    let badge: ModelFamily.Badge

    private var tint: Color {
        switch badge {
        case .new: return .iconOrange
        case .thinking: return .iconPurple
        case .coding: return .brandGreen
        case .reasoning: return .iconBlue
        case .recommended: return .statusGreen
        case .vision: return .iconBlue
        }
    }

    private var icon: String {
        switch badge {
        case .new: return "sparkle"
        case .thinking: return "lightbulb.fill"
        case .coding: return "chevron.left.forwardslash.chevron.right"
        case .reasoning: return "brain.head.profile"
        case .recommended: return "checkmark.seal.fill"
        case .vision: return "eye"
        }
    }

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 9, weight: .semibold))
            Text(badge.rawValue)
                .font(.system(size: 11, weight: .semibold))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(Capsule().fill(tint.opacity(0.14)))
    }
}
