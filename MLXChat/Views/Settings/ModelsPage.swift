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

    private var openFamily: ModelFamily? {
        openFamilyID.flatMap { ModelCatalog.family(for: $0) }
    }

    private func members(of familyID: String) -> [CatalogModel] {
        chat.modelStore.allModels.filter { ModelCatalog.familyID(for: $0) == familyID }
    }

    var body: some View {
        VStack(spacing: 0) {
            StudioPageHeader(
                title: openFamily?.name ?? "Manage Models",
                onBack: openFamilyID != nil ? { openFamilyID = nil } : onBack)

            if openFamilyID == nil {
                StudioSearchField(
                    placeholder: "Search models", text: $searchText,
                    verticalPadding: 10, horizontalPadding: 14, ringColor: .brandGreen,
                    ringOnFocusOnly: true)
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
                        // Compute overview inputs once per body pass.
                        let allModels = chat.modelStore.allModels
                        let membersByFamily = Dictionary(grouping: allModels) {
                            ModelCatalog.familyID(for: $0)
                        }
                        let downloaded = allModels.filter { chat.modelStore.isDownloaded($0.id) }
                        let visibleFamilies = ModelCatalog.families.filter { family in
                            (membersByFamily[family.id] ?? []).contains {
                                !chat.modelStore.isDownloaded($0.id)
                            }
                        }
                        let memberCounts = membersByFamily.mapValues(\.count)

                        ModelsOverview(
                            showDownloadQueue: !chat.modelStore.downloadProgress.isEmpty,
                            downloaded: downloaded,
                            visibleFamilies: visibleFamilies,
                            memberCounts: memberCounts,
                            onImport: { showImportSheet = true },
                            onDelete: { deleteCandidate = $0 },
                            onOpenFamily: { openFamilyID = $0 })
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
            "Delete model download?",
            isPresented: Binding(
                get: { deleteCandidate != nil },
                set: { if !$0 { deleteCandidate = nil } }),
            presenting: deleteCandidate
        ) { model in
            Button("Delete", role: .destructive) {
                delete(model)
                deleteCandidate = nil
            }
            Button("Cancel", role: .cancel) { deleteCandidate = nil }
        } message: { model in
            // Title is fixed by the API; name comes from the stable presented model.
            Text(
                "Delete \(model.displayName) download? The model files are removed from disk. You can download them again anytime."
            )
        }
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
}

// MARK: - Overview (Downloaded + family cards)

/// Models home content: import row, downloaded list, and family cards.
/// Receives precomputed lists/counts so filters and groupings run once.
private struct ModelsOverview: View {
    let showDownloadQueue: Bool
    let downloaded: [CatalogModel]
    let visibleFamilies: [ModelFamily]
    let memberCounts: [String: Int]
    let onImport: () -> Void
    let onDelete: (CatalogModel) -> Void
    let onOpenFamily: (String) -> Void

    var body: some View {
        if showDownloadQueue {
            DownloadQueueView()
                .padding(.bottom, 6)
        }

        ImportModelRow(onTap: onImport)
            .padding(.top, 4)

        if !downloaded.isEmpty {
            SectionHeader("Downloaded")
                .padding(.top, 10)
            ForEach(downloaded) { model in
                ModelRow(model: model) { onDelete(model) }
            }
        }

        if !visibleFamilies.isEmpty {
            SectionHeader("Model Families")
                .padding(.top, downloaded.isEmpty ? 10 : 14)
            ForEach(visibleFamilies) { family in
                FamilyCard(family: family, modelCount: memberCounts[family.id] ?? 0) {
                    onOpenFamily(family.id)
                }
            }
        }

        Link(destination: URL(string: "https://huggingface.co/mlx-community")!) {
            Label("Browse MLX models on Hugging Face", systemImage: "globe")
                .font(.callout)
        }
        .foregroundStyle(Color.brandGreen)
        .padding(.vertical, 10)
    }
}

// MARK: - Import row

/// Tappable row (MLX Studio pattern) that opens the import dialog.
private struct ImportModelRow: View {
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
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
        .accessibilityIdentifier("importModelRow")
    }
}
