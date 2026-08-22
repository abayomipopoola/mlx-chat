import SwiftUI

// MARK: - Row

struct ModelRow: View {
    @Environment(ChatController.self) private var chat
    let model: CatalogModel
    let onDelete: () -> Void
    let onDownload: (() -> Void)?
    let onSetupStreamed: (() -> Void)?

    init(
        model: CatalogModel,
        onDelete: @escaping () -> Void,
        onDownload: (() -> Void)? = nil,
        onSetupStreamed: (() -> Void)? = nil
    ) {
        self.model = model
        self.onDelete = onDelete
        self.onDownload = onDownload
        self.onSetupStreamed = onSetupStreamed
    }

    private var store: ModelStore { chat.modelStore }

    var body: some View {
        // Snapshot per-model store values once so a progress tick for another
        // model does not re-read the shared dictionary inside the control.
        let isAvailable = store.isAvailable(model.id)
        let reason = model.installRestriction
            ?? (isAvailable ? store.unsupportedReason(for: model.id) : nil)
        let progress = store.downloadProgress[model.id]
        let streamedRemovalKind = model.backend == .streamedMoE
            ? store.streamedModelRemovalKind(for: model.id)
            : nil

        HStack(spacing: 12) {
            TintedCircleIcon(
                systemName: model.backend == .streamedMoE
                    ? "externaldrive.fill"
                    : model.category == .coding
                    ? "chevron.left.forwardslash.chevron.right"
                    : model.category == .reasoning ? "brain" : "message")

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(model.displayName).font(Studio.rowTitle)
                    if model.sizeGB > 0 {
                        Text(String(format: "%.1f GB", model.sizeGB))
                            .font(Studio.sans(10))
                            .foregroundStyle(.secondary)
                    }
                    CapsuleBadge(text: model.quantLabel, tint: .iconBlue)
                    CapsuleBadge(text: "\(model.runtimeContextTokens / 1024)K ctx", tint: .iconPurple)
                }
                Text(model.blurb)
                    .font(Studio.sans(11))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                if let reason {
                    Text(reason)
                        .font(Studio.sans(10))
                        .foregroundStyle(Color.iconOrange)
                        .lineLimit(2)
                }
                Link(destination: model.backend == .streamedMoE
                    ? URL(string: "https://github.com/drumih/turbo-fieldfare#command-line-interface")!
                    : model.hugginFaceURL) {
                    Label(model.backend == .streamedMoE
                        ? "About TurboFieldfare" : "View on HuggingFace", systemImage: "link")
                        .font(Studio.sans(10))
                }
                .foregroundStyle(Color.brandGreen)
            }

            Spacer()

            ModelAvailabilityControl(
                model: model,
                isAvailable: isAvailable,
                unsupportedReason: reason,
                progress: progress,
                streamedRemovalKind: streamedRemovalKind,
                onDelete: onDelete,
                onDownload: onDownload,
                onSetupStreamed: onSetupStreamed)
        }
        .padding(14)
        .background(Color.cardFill)
        .overlay(RoundedRectangle(cornerRadius: Studio.radiusCard).stroke(Color.cardStroke, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: Studio.radiusCard))
        // Keep the row as an addressable container without replacing the
        // identifiers on its nested download and removal buttons.
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("modelRow.\(model.id)")
    }
}

/// Trailing download, registration, progress, and removal control for a model row.
/// Receives precomputed store values so it does not re-read the shared
/// `downloadProgress` dictionary on every body evaluation.
private struct ModelAvailabilityControl: View {
    @Environment(ChatController.self) private var chat
    let model: CatalogModel
    let isAvailable: Bool
    let unsupportedReason: String?
    let progress: Double?
    let streamedRemovalKind: StreamedModelRemovalKind?
    let onDelete: () -> Void
    let onDownload: (() -> Void)?
    let onSetupStreamed: (() -> Void)?

    private var store: ModelStore { chat.modelStore }
    private var modelID: String { model.id }

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
                        .font(Studio.sans(9, weight: .bold))
                }
                .frame(width: 32, height: 32)
            }
            .buttonStyle(.plain)
            .help("Cancel download")
            .accessibilityLabel("Cancel download for \(model.displayName)")
            .accessibilityValue("\(Int(fraction * 100)) percent")
        } else if isAvailable || streamedRemovalKind != nil {
            HStack(spacing: 8) {
                if !isAvailable, streamedRemovalKind == .managed {
                    Image(systemName: "externaldrive.fill.badge.exclamationmark")
                        .font(.title2)
                        .foregroundStyle(Color.iconOrange)
                        .help("Managed SSD files remain on disk but are not active")
                } else if unsupportedReason != nil {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.title2)
                        .foregroundStyle(Color.iconOrange)
                        .help("This model can't run in this build")
                } else {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.title2)
                        .foregroundStyle(Color.brandGreen)
                }
                if let onDownload,
                   StreamedMoEInstall.supportsSSDStreaming(sourceModelID: modelID) {
                    Button(action: onDownload) {
                        Image(systemName: "externaldrive.badge.plus")
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.iconBlue)
                    .help("Add SSD-streamed version")
                    .accessibilityLabel(
                        "Add SSD-streamed version of \(model.displayName)")
                    .accessibilityIdentifier("downloadModel.\(modelID)")
                }
                if model.backend == .streamedMoE, let onSetupStreamed {
                    Button(action: onSetupStreamed) {
                        Image(systemName: "slider.horizontal.3")
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .help("Manage streamed install")
                    .accessibilityLabel(
                        "Manage streamed install for \(model.displayName)")
                }
                Button(role: .destructive, action: onDelete) {
                    Image(systemName: streamedRemovalKind == .external
                        ? "xmark.circle" : "trash")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help(removalHelp)
                .accessibilityLabel(removalAccessibilityLabel)
                .accessibilityIdentifier("removeModel.\(modelID)")
            }
        } else if let unsupportedReason {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.title2)
                .foregroundStyle(Color.iconOrange)
                .help(unsupportedReason)
                .accessibilityLabel("Unavailable: \(unsupportedReason)")
        } else {
            Button(action: startDownload) {
                Image(systemName: model.backend == .streamedMoE
                    ? "externaldrive.badge.plus" : "arrow.down.circle")
                    .font(.title2)
                    .foregroundStyle(Color.brandGreen)
            }
            .buttonStyle(.plain)
            .help(model.backend == .streamedMoE ? "Set up SSD model" : "Download")
            .accessibilityLabel(model.backend == .streamedMoE
                ? "Set up SSD model for \(model.displayName)"
                : "Download \(model.displayName)")
            .accessibilityIdentifier("downloadModel.\(modelID)")
        }
    }

    private func startDownload() {
        if model.backend == .streamedMoE {
            if let onSetupStreamed {
                onSetupStreamed()
            } else {
                store.installStreamedModel(modelID)
            }
        } else if let onDownload {
            onDownload()
        } else {
            store.download(modelID)
        }
    }

    private var removalHelp: String {
        switch streamedRemovalKind {
        case .managed: return "Move managed install to Trash"
        case .external: return "Forget external install"
        case nil: return "Delete download"
        }
    }

    private var removalAccessibilityLabel: String {
        switch streamedRemovalKind {
        case .managed: return "Move \(model.displayName) to Trash"
        case .external: return "Forget external install for \(model.displayName)"
        case nil: return "Delete \(model.displayName) download"
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
                        .font(Studio.sans(10))
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
        .accessibilityIdentifier("familyCard.\(family.id)")
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
                .font(Studio.sans(11, weight: .semibold))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(Capsule().fill(tint.opacity(0.14)))
    }
}
