import SwiftUI

private enum ModelsSheetDestination: Identifiable {
    case importMLX
    case download(model: CatalogModel, initialMode: ModelDownloadMode)

    var id: String {
        switch self {
        case .importMLX: return "import-mlx"
        case .download(let model, let mode): return "download-\(model.id)-\(mode.rawValue)"
        }
    }
}

struct ModelsPage: View {
    @Environment(ChatController.self) private var chat
    /// Family drill-down is an AppRoute; nil shows the overview.
    var familyID: String? = nil
    var onOpenFamily: (String) -> Void = { _ in }

    @State private var searchText = ""
    @State private var presentedSheet: ModelsSheetDestination?
    @State private var deleteCandidate: CatalogModel?
    @State private var removalError: String?

    private var isSearching: Bool {
        !searchText.trimmingCharacters(in: .whitespaces).isEmpty
    }

    /// Drops a restricted MLX source row from search results when its
    /// SSD-streamed replacement is already visible, matching family drill-down.
    static func deduplicatedSearchResults(
        _ matches: [CatalogModel],
        visibleStreamedIDs: Set<String>
    ) -> [CatalogModel] {
        matches.filter { model in
            guard model.backend == .mlx,
                  model.installRestriction != nil,
                  let variant = StreamedMoEInstall.variant(forSourceModelID: model.id)
            else { return true }
            return !visibleStreamedIDs.contains(variant.modelID)
        }
    }

    private var searchResults: [CatalogModel] {
        let query = searchText.trimmingCharacters(in: .whitespaces)
        let visible = chat.modelStore.allModels.filter { isVisibleInModelManager($0) }
        let visibleStreamedIDs = Set(
            visible.filter { $0.backend.isSSDStreamed }.map(\.id))
        let matches = visible.filter {
            $0.displayName.localizedStandardContains(query)
                || $0.id.localizedStandardContains(query)
        }
        return Self.deduplicatedSearchResults(matches, visibleStreamedIDs: visibleStreamedIDs)
    }

    private var openFamily: ModelFamily? {
        familyID.flatMap { ModelCatalog.family(for: $0) }
    }

    private func members(of familyID: String) -> [CatalogModel] {
        let allModels = chat.modelStore.allModels
        return allModels
            .filter { $0.backend == .mlx && ModelCatalog.familyID(for: $0) == familyID }
            .map { model in
                guard model.installRestriction != nil,
                      let variant = StreamedMoEInstall.variant(forSourceModelID: model.id),
                      let streamed = allModels.first(where: { $0.id == variant.modelID }),
                      isVisibleInModelManager(streamed) else {
                    return model
                }
                return streamed
            }
    }

    private func isVisibleInModelManager(_ model: CatalogModel) -> Bool {
        model.backend == .mlx
            || chat.modelStore.isAvailable(model.id)
            || chat.modelStore.isDownloading(model.id)
            || chat.modelStore.streamedModelRemovalKind(for: model.id) != nil
    }

    var body: some View {
        VStack(spacing: 0) {
            StudioPageHeader(title: openFamily?.name ?? "Manage Models")

            if familyID == nil {
                StudioSearchField(
                    placeholder: "Search models", text: $searchText,
                    verticalPadding: 10, horizontalPadding: 14,
                    ringOnFocusOnly: true)
                    .frame(maxWidth: Studio.contentMaxWidth)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
            }

            if let family = openFamily {
                modelsScrollView {
                    familyDetail(family)
                }
                .id("models-family-\(family.id)")
            } else {
                modelsScrollView {
                    overviewContent
                }
                .id("models-overview")
            }
        }
        .background(Color.detailBackground)
        .sheet(item: $presentedSheet) { destination in
            switch destination {
            case .importMLX:
                ImportModelSheet()
            case .download(let model, let initialMode):
                ModelDownloadOptionsSheet(model: model, initialMode: initialMode)
            }
        }
        .alert(
            deleteCandidate.map(deleteAlertTitle) ?? "Remove model?",
            isPresented: Binding(
                get: { deleteCandidate != nil },
                set: { if !$0 { deleteCandidate = nil } }),
            presenting: deleteCandidate
        ) { model in
            Button(deleteActionTitle(for: model), role: .destructive) {
                delete(model)
                deleteCandidate = nil
            }
            Button("Cancel", role: .cancel) { deleteCandidate = nil }
        } message: { model in
            Text(deleteMessage(for: model))
        }
        .alert(
            "Couldn’t remove model",
            isPresented: Binding(
                get: { removalError != nil },
                set: { if !$0 { removalError = nil } })
        ) {
            Button("OK", role: .cancel) { removalError = nil }
        } message: {
            Text(removalError ?? "Unknown error")
        }
    }

    /// Keep overview and family routes in separate scroll-view identities.
    /// AppKit otherwise preserves the overview's bottom offset when the much
    /// shorter family content replaces it, leaving that content out of frame.
    private func modelsScrollView<Content: View>(
        @ViewBuilder content: () -> Content
    ) -> some View {
        ScrollView {
            LazyVStack(spacing: 10) {
                content()
            }
            .frame(maxWidth: Studio.contentMaxWidth)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 16)
            .padding(.bottom, 16)
        }
        .accessibilityIdentifier("modelsScrollView")
    }

    @ViewBuilder
    private var overviewContent: some View {
        if isSearching {
            if searchResults.isEmpty {
                ContentUnavailableView(
                    "No models found",
                    systemImage: "magnifyingglass")
                    .padding(.top, 40)
            } else {
                ForEach(searchResults) { model in
                    modelRow(model)
                }
            }
        } else {
            // Compute overview inputs once per body pass.
            let allModels = chat.modelStore.allModels
            let standardModels = allModels.filter { $0.backend == .mlx }
            let streamedModels = allModels.filter {
                $0.backend.isSSDStreamed && isVisibleInModelManager($0)
            }
            let membersByFamily = Dictionary(grouping: standardModels) {
                ModelCatalog.familyID(for: $0)
            }
            let localModels = standardModels.filter {
                chat.modelStore.isAvailable($0.id)
            }
            let visibleFamilies = ModelCatalog.families.filter { family in
                (membersByFamily[family.id] ?? []).contains { model in
                    if chat.modelStore.isAvailable(model.id) { return false }
                    if model.installRestriction != nil,
                       let variant = StreamedMoEInstall.variant(forSourceModelID: model.id),
                       chat.modelStore.isAvailable(variant.modelID) {
                        return false
                    }
                    return true
                }
            }
            let memberCounts = membersByFamily.mapValues(\.count)

            ModelsOverview(
                showDownloadQueue: !chat.modelStore.downloadProgress.isEmpty,
                streamedModels: streamedModels,
                localModels: localModels,
                visibleFamilies: visibleFamilies,
                memberCounts: memberCounts,
                onAdd: { presentedSheet = .importMLX },
                onSetupStreamed: presentStreamedSetup,
                onAddStreamed: { model in
                    presentedSheet = .download(model: model, initialMode: .ssdStreamed)
                },
                onDelete: { deleteCandidate = $0 },
                onOpenFamily: onOpenFamily)
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
            modelRow(model)
        }
    }

    private func modelRow(_ model: CatalogModel) -> some View {
        let supportsSSD = StreamedMoEInstall.supportsSSDStreaming(sourceModelID: model.id)
            || MLXStreamedInstall.supportsSSDStreaming(sourceModelID: model.id)
        return ModelRow(
            model: model,
            onDelete: { deleteCandidate = model },
            onDownload: downloadAction(for: model),
            onAddStreamed: supportsSSD
                ? { presentedSheet = .download(model: model, initialMode: .ssdStreamed) }
                : nil,
            onSetupStreamed: model.backend.isSSDStreamed
                ? {
                    presentStreamedSetup(for: model)
                }
                : nil)
    }

    private func downloadAction(for model: CatalogModel) -> (() -> Void)? {
        guard model.backend == .mlx else { return nil }
        if model.installRestriction == nil {
            return {
                presentedSheet = .download(model: model, initialMode: .mlx)
            }
        }
        guard StreamedMoEInstall.supportsSSDStreaming(sourceModelID: model.id)
            || MLXStreamedInstall.supportsSSDStreaming(sourceModelID: model.id) else {
            return nil
        }
        return {
            presentedSheet = .download(model: model, initialMode: .ssdStreamed)
        }
    }

    private func presentStreamedSetup(for streamedModel: CatalogModel) {
        let sourceID = StreamedMoEInstall.variant(
            forStreamedModelID: streamedModel.id)?.sourceRepoID
            ?? MLXStreamedInstall.variant(
                forStreamedModelID: streamedModel.id)?.sourceModelID
        guard let sourceID,
              let sourceModel = ModelCatalog.model(for: sourceID) else { return }
        presentedSheet = .download(model: sourceModel, initialMode: .ssdStreamed)
    }

    private func delete(_ model: CatalogModel) {
        if chat.isStreaming, chat.selectedModelID == model.id {
            removalError = "Stop the current response before removing \(model.displayName)."
            return
        }
        chat.runtime.modelDeleted(model.id)
        if model.category == .imported {
            removalError = chat.modelStore.removeImported(model.id)
        } else {
            removalError = chat.modelStore.delete(model.id)
        }
        if removalError == nil {
            chat.handleModelDeleted(model.id)
        }
    }

    private func deleteAlertTitle(for model: CatalogModel) -> String {
        if model.backend == .mlxStreamed {
            return "Disable SSD-streamed mode?"
        }
        guard model.backend == .streamedMoE else {
            return "Delete model download?"
        }
        switch chat.modelStore.streamedModelRemovalKind(for: model.id) {
        case .managed: return "Move model install to Trash?"
        case .external: return "Forget model install?"
        case nil: return "Remove model install?"
        }
    }

    private func deleteActionTitle(for model: CatalogModel) -> String {
        if model.backend == .mlxStreamed { return "Disable" }
        guard model.backend == .streamedMoE else { return "Delete" }
        return chat.modelStore.streamedModelRemovalKind(for: model.id) == .managed
            ? "Move to Trash" : "Forget"
    }

    private func deleteMessage(for model: CatalogModel) -> String {
        if model.backend == .mlxStreamed {
            return "Remove \(model.displayName) from SSD-Streamed Models? The shared MLX weights stay on disk."
        }
        if model.backend == .mlx,
           let dependent = MLXStreamedInstall.variant(forSourceModelID: model.id),
           chat.modelStore.isMLXStreamedEnabled(dependent.modelID) {
            return "Delete \(model.displayName) download? The shared model files are removed from disk, which also disables SSD-streamed mode."
        }
        guard model.backend == .streamedMoE else {
            return "Delete \(model.displayName) download? The model files are removed from disk. You can download them again anytime."
        }
        switch chat.modelStore.streamedModelRemovalKind(for: model.id) {
        case .managed:
            return "Move \(model.displayName)’s MLX Chat-managed .gturbo files to Trash? Empty Trash to reclaim their disk space."
        case .external:
            return "Forget \(model.displayName)? Its external .gturbo files stay on disk."
        case nil:
            return "Forget \(model.displayName)?"
        }
    }
}

// MARK: - Overview (local models + family cards)

/// Models home content: add-model row, available local models, and family cards.
/// Receives precomputed lists/counts so filters and groupings run once.
private struct ModelsOverview: View {
    let showDownloadQueue: Bool
    let streamedModels: [CatalogModel]
    let localModels: [CatalogModel]
    let visibleFamilies: [ModelFamily]
    let memberCounts: [String: Int]
    let onAdd: () -> Void
    let onSetupStreamed: (CatalogModel) -> Void
    let onAddStreamed: (CatalogModel) -> Void
    let onDelete: (CatalogModel) -> Void
    let onOpenFamily: (String) -> Void

    var body: some View {
        if showDownloadQueue {
            DownloadQueueView()
                .padding(.bottom, 6)
        }

        AddModelRow(onTap: onAdd)
            .padding(.top, 4)

        if !streamedModels.isEmpty {
            SectionHeader("SSD-Streamed Models")
                .padding(.top, 10)
            ForEach(streamedModels) { model in
                ModelRow(
                    model: model,
                    onDelete: { onDelete(model) },
                    onSetupStreamed: { onSetupStreamed(model) })
            }
        }

        if !localModels.isEmpty {
            SectionHeader("Local Models")
                .padding(.top, streamedModels.isEmpty ? 10 : 14)
            ForEach(localModels) { model in
                ModelRow(
                    model: model,
                    onDelete: { onDelete(model) },
                    onAddStreamed: (StreamedMoEInstall.supportsSSDStreaming(sourceModelID: model.id)
                        || MLXStreamedInstall.supportsSSDStreaming(sourceModelID: model.id))
                        ? { onAddStreamed(model) }
                        : nil)
            }
        }

        if !visibleFamilies.isEmpty {
            SectionHeader("Model Families")
                .padding(.top, localModels.isEmpty ? 10 : 14)
            ForEach(visibleFamilies) { family in
                FamilyCard(family: family, modelCount: memberCounts[family.id] ?? 0) {
                    onOpenFamily(family.id)
                }
            }
        }

        Link(destination: URL(string: "https://huggingface.co/models?library=mlx")!) {
            Label("Browse MLX models on Hugging Face", systemImage: "globe")
                .font(Studio.sans(12))
        }
        .foregroundStyle(Color.brandGreen)
        .padding(.vertical, 10)
    }
}

// MARK: - Add-model row

/// Tappable row for importing an additional Hugging Face MLX model.
private struct AddModelRow: View {
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
                    Text("Add Model")
                        .font(Studio.rowTitle)
                    Text("Import an MLX model from Hugging Face")
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
        .accessibilityIdentifier("addModelRow")
    }
}
