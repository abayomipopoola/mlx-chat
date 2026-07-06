import SwiftUI

struct ModelsPage: View {
    @Environment(ChatController.self) private var chat
    /// Pops back (custom routing, no NavigationStack).
    var onBack: () -> Void = {}

    @State private var searchText = ""
    @State private var showImportSheet = false
    @State private var deleteCandidate: CatalogModel?
    /// Non-nil while a family's model list is open (in-page drill-down).
    @State private var openFamilyID: String?

    private var isSearching: Bool {
        !searchText.trimmingCharacters(in: .whitespaces).isEmpty
    }

    private var searchResults: [CatalogModel] {
        let query = searchText.trimmingCharacters(in: .whitespaces)
        return chat.modelStore.allModels.filter {
            $0.displayName.localizedStandardContains(query) || $0.id.localizedStandardContains(query)
        }
    }

    private var downloaded: [CatalogModel] {
        chat.modelStore.allModels.filter { chat.modelStore.isDownloaded($0.id) }
    }

    private func members(of familyID: String) -> [CatalogModel] {
        chat.modelStore.allModels.filter { ModelCatalog.familyID(for: $0) == familyID }
    }

    /// Family cards worth showing: at least one member not yet downloaded.
    private var visibleFamilies: [ModelFamily] {
        ModelCatalog.families.filter { family in
            members(of: family.id).contains { !chat.modelStore.isDownloaded($0.id) }
        }
    }

    private var openFamily: ModelFamily? {
        openFamilyID.flatMap { ModelCatalog.family(for: $0) }
    }

    var body: some View {
        VStack(spacing: 0) {
            StudioPageHeader(
                title: openFamily?.name ?? "Manage Models",
                onBack: openFamilyID != nil ? { openFamilyID = nil } : onBack)

            if openFamilyID == nil {
                StudioSearchField(placeholder: "Search models", text: $searchText)
                    .frame(maxWidth: Studio.contentMaxWidth)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
            }

            ScrollView {
                LazyVStack(spacing: 10) {
                    if let family = openFamily {
                        familyDetail(family)
                    } else if isSearching {
                        ForEach(searchResults) { model in
                            ModelRow(model: model) { deleteCandidate = model }
                        }
                    } else {
                        overview
                    }
                }
                .frame(maxWidth: Studio.contentMaxWidth)
                .frame(maxWidth: .infinity)
                .padding(.horizontal, 16)
                .padding(.bottom, 16)
            }
        }
        .background(Color.detailBackground)
        .animation(.easeOut(duration: 0.18), value: openFamilyID)
        .sheet(isPresented: $showImportSheet) {
            ImportModelSheet()
        }
        .alert(
            "Delete \(deleteCandidate?.displayName ?? "model") download?",
            isPresented: Binding(
                get: { deleteCandidate != nil },
                set: { if !$0 { deleteCandidate = nil } })
        ) {
            Button("Delete", role: .destructive) {
                if let model = deleteCandidate { delete(model) }
                deleteCandidate = nil
            }
            Button("Cancel", role: .cancel) { deleteCandidate = nil }
        } message: {
            Text("The model files are removed from disk. You can download them again anytime.")
        }
    }

    // MARK: - Overview (Downloaded + family cards)

    @ViewBuilder
    private var overview: some View {
        if !chat.modelStore.downloadProgress.isEmpty {
            DownloadQueueView()
                .padding(.bottom, 6)
        }

        if !downloaded.isEmpty {
            SectionHeader("Downloaded")
                .padding(.top, 4)
            ForEach(downloaded) { model in
                ModelRow(model: model) { deleteCandidate = model }
            }
        }

        if !visibleFamilies.isEmpty {
            SectionHeader("Model Families")
                .padding(.top, downloaded.isEmpty ? 4 : 14)
            ForEach(visibleFamilies) { family in
                FamilyCard(family: family, modelCount: members(of: family.id).count) {
                    openFamilyID = family.id
                }
            }
        }

        importSection
            .padding(.top, 6)

        Link(destination: URL(string: "https://huggingface.co/mlx-community")!) {
            Label("Browse MLX models on Hugging Face", systemImage: "globe")
                .font(.callout)
        }
        .foregroundStyle(Color.brandGreen)
        .padding(.vertical, 10)
    }

    // MARK: - Family drill-down

    @ViewBuilder
    private func familyDetail(_ family: ModelFamily) -> some View {
        Text(family.blurb)
            .font(Studio.body13)
            .foregroundStyle(Color.subtitleGray)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 4)
            .padding(.bottom, 4)
        ForEach(members(of: family.id)) { model in
            ModelRow(model: model) { deleteCandidate = model }
        }
    }

    private func delete(_ model: CatalogModel) {
        chat.runtime.modelDeleted(model.id)
        if model.category == .imported {
            chat.modelStore.removeImported(model.id)
        } else {
            chat.modelStore.delete(model.id)
        }
    }

    /// Tappable row (MLX Studio pattern) that opens the import dialog.
    private var importSection: some View {
        Button {
            showImportSheet = true
        } label: {
            HStack(spacing: 12) {
                Image(systemName: "arrow.down")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 36, height: 36)
                    .background(Circle().fill(Color.brandGreen))
                VStack(alignment: .leading, spacing: 2) {
                    Text("Import MLX Model")
                        .font(Studio.rowTitle)
                    Text("Download existing MLX models")
                        .font(Studio.body13)
                        .foregroundStyle(Color.subtitleGray)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.tertiary)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.brandGreen.opacity(0.08))
            .overlay(RoundedRectangle(cornerRadius: Studio.radiusCard)
                .stroke(Color.brandGreen.opacity(0.35), lineWidth: 1))
            .clipShape(RoundedRectangle(cornerRadius: Studio.radiusCard))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Row

private struct ModelRow: View {
    @Environment(ChatController.self) private var chat
    let model: CatalogModel
    let onDelete: () -> Void

    private var store: ModelStore { chat.modelStore }

    var body: some View {
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
                Link(destination: model.hugginFaceURL) {
                    Label("View on HuggingFace", systemImage: "link")
                        .font(.caption)
                }
                .foregroundStyle(Color.brandGreen)
            }

            Spacer()

            trailingControl
        }
        .padding(14)
        .background(Color.cardFill)
        .overlay(RoundedRectangle(cornerRadius: Studio.radiusCard).stroke(Color.cardStroke, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: Studio.radiusCard))
    }

    @ViewBuilder
    private var trailingControl: some View {
        if let fraction = store.downloadProgress[model.id] {
            Button {
                store.cancelDownload(model.id)
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
        } else if store.isDownloaded(model.id) {
            HStack(spacing: 8) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.title2)
                    .foregroundStyle(Color.brandGreen)
                Button(role: .destructive, action: onDelete) {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Delete download")
            }
        } else {
            Button {
                store.download(model.id)
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

private struct FamilyCard: View {
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

private struct FamilyBadgeChip: View {
    let badge: ModelFamily.Badge

    private var tint: Color {
        switch badge {
        case .new: return .iconOrange
        case .thinking: return .iconPurple
        case .coding: return .brandGreen
        case .reasoning: return .iconBlue
        case .recommended: return .statusGreen
        }
    }

    private var icon: String {
        switch badge {
        case .new: return "sparkle"
        case .thinking: return "lightbulb.fill"
        case .coding: return "chevron.left.forwardslash.chevron.right"
        case .reasoning: return "brain.head.profile"
        case .recommended: return "checkmark.seal.fill"
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

// MARK: - Import dialog (compact MLX Studio pattern)

private struct ImportModelSheet: View {
    @Environment(ChatController.self) private var chat
    @Environment(\.dismiss) private var dismiss

    @State private var input = ""
    @State private var importError: String?
    @State private var importing = false

    /// Accepts a bare repo id or a full Hugging Face URL.
    private var normalizedRepoID: String {
        var text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        for prefix in ["https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"]
        where text.lowercased().hasPrefix(prefix) {
            text = String(text.dropFirst(prefix.count))
            break
        }
        let parts = text.split(separator: "/")
        if parts.count >= 2 { return "\(parts[0])/\(parts[1])" }
        return text
    }

    var body: some View {
        VStack(spacing: 0) {
            Text("Import Custom Model")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)

            Divider()

            ScrollView {
                VStack(spacing: 16) {
                    Image(systemName: "arrow.down")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(.white)
                        .frame(width: 44, height: 44)
                        .background(Circle().fill(Color.brandGreen))
                        .padding(.top, 18)

                    VStack(spacing: 4) {
                        Text("Import from HuggingFace")
                            .font(.system(size: 20, weight: .bold))
                        Text("Download MLX-compatible models directly from HuggingFace")
                            .font(Studio.body13)
                            .foregroundStyle(Color.subtitleGray)
                            .multilineTextAlignment(.center)
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Model ID or URL")
                            .font(.system(size: 13, weight: .semibold))
                        TextField("mlx-community/ModelName-4bit", text: $input)
                            .textFieldStyle(.plain)
                            .font(.system(size: 13).monospaced())
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: Studio.radiusField, style: .continuous)
                                    .fill(Color.fieldFill))
                            .onSubmit(runImport)
                        Text("Examples: mlx-community/Qwen2-VL-7B-4bit or a full HuggingFace URL")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        if let importError {
                            Text(importError)
                                .font(.caption)
                                .foregroundStyle(.red)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .studioCard()

                    VStack(alignment: .leading, spacing: 8) {
                        Label("At Your Own Risk", systemImage: "exclamationmark.triangle.fill")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundStyle(Color.iconOrange)
                        VStack(alignment: .leading, spacing: 4) {
                            bullet("Model compatibility is not guaranteed")
                            bullet("May consume significant storage space")
                            bullet("Could crash or perform poorly on your device")
                        }
                        Text("Only import models from trusted sources. Look for models in the **mlx-community** organization for best compatibility.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Link(destination: URL(string: "https://huggingface.co/mlx-community")!) {
                            Label("Browse MLX Models on HuggingFace", systemImage: "globe")
                                .font(.caption.weight(.semibold))
                        }
                        .foregroundStyle(Color.brandGreen)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(14)
                    .background(Color.iconOrange.opacity(0.08))
                    .overlay(RoundedRectangle(cornerRadius: Studio.radiusCard)
                        .stroke(Color.iconOrange.opacity(0.35), lineWidth: 1))
                    .clipShape(RoundedRectangle(cornerRadius: Studio.radiusCard))
                }
                .padding(.horizontal, 18)
                .padding(.bottom, 14)
            }

            Divider()

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button(importing ? "Importing…" : "Import") { runImport() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .tint(Color.brandGreen)
                    .disabled(importing || normalizedRepoID.isEmpty)
            }
            .padding(14)
        }
        .frame(width: 460, height: 560)
    }

    private func bullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Text("•").foregroundStyle(.secondary)
            Text(text)
                .font(Studio.caption13)
                .foregroundStyle(.secondary)
        }
    }

    private func runImport() {
        let repo = normalizedRepoID
        guard !repo.isEmpty, !importing else { return }
        importing = true
        importError = nil
        Task {
            let error = await chat.modelStore.importModel(repoID: repo)
            importing = false
            importError = error
            if error == nil { dismiss() }
        }
    }
}
