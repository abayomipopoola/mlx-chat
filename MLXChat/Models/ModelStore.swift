import Foundation
import HuggingFace
import MLXLMCommon
import Observation

enum StreamedModelRemovalKind: Equatable, Sendable {
    /// MLX Chat created the install under `~/Models`; removal moves all of its
    /// resumable artifacts to Trash.
    case managed
    /// The user registered a folder elsewhere; removal only forgets access.
    case external
}

enum StreamedManagedInstallState: Equatable, Sendable {
    case none
    case resumable
    case complete
}

/// Owns MLX downloads, external streamed-model registrations, storage usage,
/// and the imported-model list.
///
/// Storage root is `~/Models` (the REAL home — the sandbox remaps `NSHomeDirectory`,
/// so it is resolved via getpwuid; access is granted by the home-relative sandbox
/// exception for `/Models/`). Two on-disk layouts are recognized:
/// - plain `~/Models/<org>/<name>/…` (pre-existing / hand-placed models)
/// - HubClient cache `~/Models/hub/models--<org>--<name>/snapshots/<sha>/…` (our downloads)
@MainActor
@Observable
final class ModelStore {
    /// Holds one balanced security-scope lease for a registered external model.
    /// Replacing, forgetting, or releasing the store closes the lease in deinit.
    private final class StreamedModelRegistration {
        let directory: URL
        private let closesSecurityScope: Bool

        init(directory: URL, closesSecurityScope: Bool) {
            self.directory = directory
            self.closesSecurityScope = closesSecurityScope
        }

        deinit {
            if closesSecurityScope { directory.stopAccessingSecurityScopedResource() }
        }
    }

    /// `~/Models` in the user's real home directory.
    nonisolated static let modelsRoot: URL = {
        let home = getpwuid(getuid()).flatMap { String(validatingCString: $0.pointee.pw_dir) }
            ?? NSHomeDirectory()
        return URL(fileURLWithPath: home, isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }()

    /// HubClient downloads live under `~/Models/hub`.
    nonisolated static let hubCacheDirectory: URL =
        modelsRoot.appendingPathComponent("hub", isDirectory: true)

    nonisolated static let hubClient = HubClient(cache: HubCache(cacheDirectory: hubCacheDirectory))

    /// Mirrors MLXLMCommon's package-internal `modelDownloadPatterns`.
    nonisolated static let downloadPatterns = ["*.safetensors", "*.json", "*.jinja"]

    private(set) var downloadProgress: [String: Double] = [:]
    private(set) var customModels: [CatalogModel]
    var lastError: String?

    private var downloadTasks: [String: Task<Void, Never>] = [:]
    private var streamedInstallTasks: [String: Task<Void, Never>] = [:]
    private var streamedInstallers: [String: StreamedModelInstaller] = [:]
    private var streamedInstallProgressByModelID: [String: StreamedInstallProgress] = [:]
    private var streamedInstallErrors: [String: String] = [:]
    /// Invalidates filesystem-backed managed-install queries after a Trash move.
    private var streamedStorageRevision = 0
    /// Validated `.gturbo` directories chosen by the user. MLX snapshots keep
    /// using the existing `~/Models` layouts and never enter this map.
    private var streamedModelRegistrations: [String: StreamedModelRegistration] = [:]
    /// Cached `ModelSupport.unsupportedReason(modelID:)` results; entry
    /// invalidated when a download finishes or the model is deleted.
    private var unsupportedReasonCache: [String: String?] = [:]

    init() {
#if DEBUG
        let seededCustomModels: [CatalogModel]? =
            ProcessInfo.processInfo.arguments.contains("--seed-imported-models")
            ? [
                ModelCatalog.importedModel(
                    repoID: "mlx-community/Imported-Regression-4bit")
            ]
            : nil
#else
        let seededCustomModels: [CatalogModel]? = nil
#endif
        if let seededCustomModels {
            // UI regression data stays in memory and never replaces the user's imports.
            customModels = seededCustomModels
        } else if let data = UserDefaults.standard.data(forKey: Keys.customModels),
           let models = try? JSONDecoder().decode([CatalogModel].self, from: data) {
            let upgraded = models
                .map(ModelCatalog.applyingKnownCapabilities)
                .filter { ModelCatalog.model(for: $0.id) == nil }
            customModels = upgraded
            if upgraded != models, let encoded = try? JSONEncoder().encode(upgraded) {
                UserDefaults.standard.set(encoded, forKey: Keys.customModels)
            }
        } else {
            customModels = []
        }
        restoreStreamedModelDirectories()
        restoreManagedStreamedModelDirectory()
        try? FileManager.default.createDirectory(at: Self.hubCacheDirectory, withIntermediateDirectories: true)
    }

    /// Catalog plus imported entries, in display order.
    var allModels: [CatalogModel] { ModelCatalog.models + customModels }

    func model(for id: String) -> CatalogModel? {
        allModels.first { $0.id == id }
    }

    // MARK: - Local availability

    /// `~/Models/<org>/<name>` for a repo id, when the id has the org/name shape.
    nonisolated static func plainDirectory(for id: String) -> URL? {
        let parts = id.split(separator: "/")
        guard parts.count == 2 else { return nil }
        return modelsRoot
            .appendingPathComponent(String(parts[0]), isDirectory: true)
            .appendingPathComponent(String(parts[1]), isDirectory: true)
    }

    nonisolated static func directoryHasWeights(_ url: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.appendingPathComponent("config.json").path) else { return false }
        let contents = (try? fm.contentsOfDirectory(atPath: url.path)) ?? []
        return contents.contains { $0.hasSuffix(".safetensors") }
    }

    /// Loadable MLX directory: plain layout wins, hub snapshot second.
    nonisolated static func localMLXModelDirectory(for id: String) -> URL? {
        if let plain = plainDirectory(for: id), directoryHasWeights(plain) {
            return plain
        }
        if let snapshot = newestSnapshot(for: id), directoryHasWeights(snapshot) {
            return snapshot
        }
        return nil
    }

    func availableDirectory(for id: String) -> URL? {
        if StreamedMoEInstall.variant(forStreamedModelID: id) != nil {
            return streamedModelRegistrations[id]?.directory
        }
        return Self.localMLXModelDirectory(for: id)
    }

    func isAvailable(_ id: String) -> Bool {
        availableDirectory(for: id) != nil
    }

    func hasManagedStreamedModelFiles(_ id: String) -> Bool {
        managedStreamedInstallState(for: id) != .none
    }

    func managedStreamedInstallState(for id: String) -> StreamedManagedInstallState {
        _ = streamedStorageRevision
        guard let variant = StreamedMoEInstall.variant(
            forStreamedModelID: id) else { return .none }
        let paths = variant.managedInstallPaths
        let fileManager = FileManager.default
        return Self.managedStreamedInstallState(
            finalDirectoryExists: fileManager.fileExists(
                atPath: paths.finalDirectory.path),
            partialDirectoryExists: fileManager.fileExists(
                atPath: paths.partialDirectory.path),
            checkpointExists: fileManager.fileExists(
                atPath: paths.checkpointFile.path))
    }

    nonisolated static func managedStreamedInstallState(
        finalDirectoryExists: Bool,
        partialDirectoryExists: Bool,
        checkpointExists: Bool
    ) -> StreamedManagedInstallState {
        if finalDirectoryExists { return .complete }
        return partialDirectoryExists || checkpointExists ? .resumable : .none
    }

    func streamedModelRemovalKind(for id: String) -> StreamedModelRemovalKind? {
        guard let variant = StreamedMoEInstall.variant(
            forStreamedModelID: id) else { return nil }
        return Self.streamedModelRemovalKind(
            registeredDirectory: streamedModelRegistrations[id]?.directory,
            managedDirectory: variant.managedDirectory,
            managedArtifactsExist: hasManagedStreamedModelFiles(id))
    }

    nonisolated static func streamedModelRemovalKind(
        registeredDirectory: URL?,
        managedDirectory: URL,
        managedArtifactsExist: Bool
    ) -> StreamedModelRemovalKind? {
        if let registeredDirectory {
            return registeredDirectory.standardizedFileURL
                == managedDirectory.standardizedFileURL
                ? .managed : .external
        }
        return managedArtifactsExist ? .managed : nil
    }

    func isUsableLocalSelection(_ id: String) -> Bool {
        guard isAvailable(id) else { return false }
        return StreamedMoEInstall.variant(forStreamedModelID: id) != nil
            || unsupportedReason(for: id) == nil
    }

    /// Why this on-disk model can't run in this build, or nil if it can / isn't local.
    func unsupportedReason(for id: String) -> String? {
        if StreamedMoEInstall.variant(forStreamedModelID: id) != nil { return nil }
        // [String: String?] — use updateValue so a nil reason (supported model)
        // is stored; plain assignment of nil would remove the key instead.
        if unsupportedReasonCache.index(forKey: id) != nil {
            return unsupportedReasonCache[id]!
        }
        let reason = ModelSupport.unsupportedReason(modelID: id)
        unsupportedReasonCache.updateValue(reason, forKey: id)
        return reason
    }

    var localModelCount: Int { allModels.filter { isAvailable($0.id) }.count }

    nonisolated static func hubRepoDirectory(for id: String) -> URL? {
        guard let repoID = HuggingFace.Repo.ID(rawValue: id) else { return nil }
        return HubCache(cacheDirectory: hubCacheDirectory).repoDirectory(repo: repoID, kind: .model)
    }

    nonisolated static func snapshotsDirectory(for id: String) -> URL? {
        guard let repoID = HuggingFace.Repo.ID(rawValue: id) else { return nil }
        return HubCache(cacheDirectory: hubCacheDirectory).snapshotsDirectory(repo: repoID, kind: .model)
    }

    /// Newest snapshot subdirectory of the hub layout, nil when none exists.
    nonisolated static func newestSnapshot(for id: String) -> URL? {
        guard let dir = snapshotsDirectory(for: id) else { return nil }
        let fm = FileManager.default
        let subdirs = (try? fm.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey]))?
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
        return subdirs?.max {
            let a = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let b = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return a < b
        }
    }

    // MARK: - Download / cancel / delete

    func isDownloading(_ id: String) -> Bool { downloadProgress[id] != nil }

    func streamedInstallProgress(for modelID: String) -> StreamedInstallProgress? {
        streamedInstallProgressByModelID[modelID]
    }

    func streamedInstallError(for modelID: String) -> String? {
        streamedInstallErrors[modelID]
    }

    func download(_ id: String) {
        guard downloadTasks[id] == nil else { return }
        if StreamedMoEInstall.variant(forStreamedModelID: id) != nil {
            installStreamedModel(id)
            return
        }
        if let restriction = model(for: id)?.installRestriction {
            lastError = restriction
            return
        }
        // Downloads land in the plain `~/Models/<org>/<name>` layout, which
        // `localMLXModelDirectory` resolves first, so the load path finds them offline
        // and never falls back to the (broken) library `.id` downloader. See
        // `SnapshotDownloader` for why we don't use the vendored `downloadSnapshot`.
        guard let destination = Self.plainDirectory(for: id) else {
            lastError = "Invalid model id: \(id)"
            return
        }
        lastError = nil
        downloadProgress[id] = 0
        let sourceModel = model(for: id)
        let sourceSubfolder = sourceModel?.sourceSubfolder
        let sourceRepoID = sourceModel?.huggingFaceRepoID ?? id
        let task = Task { [weak self] in
            do {
                try await SnapshotDownloader.download(
                    repoID: sourceRepoID, subfolder: sourceSubfolder, destination: destination,
                    onProgress: { fraction in
                        Task { @MainActor in
                            self?.downloadProgress[id] = fraction
                        }
                    })
            } catch is CancellationError {
                // Cancelled: the `.partial` staging dir is left for resume.
            } catch {
                await MainActor.run { [weak self] in
                    self?.lastError = "Download failed: \(error.localizedDescription)"
                }
            }
            await MainActor.run { [weak self] in
                self?.downloadProgress[id] = nil
                self?.downloadTasks[id] = nil
                self?.unsupportedReasonCache.removeValue(forKey: id)
            }
        }
        downloadTasks[id] = task
    }

    /// Awaitable variant used by the lazy first-send path.
    func downloadAndWait(_ id: String) async throws {
        if isAvailable(id) { return }
        download(id)
        if let task = downloadTasks[id] { await task.value }
        guard isAvailable(id) else {
            throw EngineError.generationFailed(lastError ?? "Download failed.")
        }
    }

    func cancelDownload(_ id: String) {
        if StreamedMoEInstall.variant(forStreamedModelID: id) != nil {
            streamedInstallTasks[id]?.cancel()
            streamedInstallers[id]?.cancel()
            streamedInstallProgressByModelID[id] = nil
        }
        downloadTasks[id]?.cancel()
        downloadTasks[id] = nil
        downloadProgress[id] = nil
    }

    /// Removes an app-owned install or forgets an external registration.
    /// Managed `.gturbo` artifacts go to macOS Trash so the operation remains
    /// recoverable; externally selected folders are never modified.
    @discardableResult
    func delete(_ id: String) -> String? {
        if let variant = StreamedMoEInstall.variant(forStreamedModelID: id) {
            return removeStreamedModel(id, variant: variant)
        }
        if let plain = Self.plainDirectory(for: id) {
            try? FileManager.default.removeItem(at: plain)
            let staging = plain.deletingLastPathComponent()
                .appendingPathComponent(plain.lastPathComponent + ".partial", isDirectory: true)
            try? FileManager.default.removeItem(at: staging)
        }
        if let hub = Self.hubRepoDirectory(for: id) {
            try? FileManager.default.removeItem(at: hub)
        }
        unsupportedReasonCache.removeValue(forKey: id)
        return nil
    }

    private func removeStreamedModel(
        _ id: String,
        variant: StreamedMoEVariant
    ) -> String? {
        switch streamedModelRemovalKind(for: id) {
        case .external:
            unregisterStreamedModel(id)
            lastError = nil
            return nil
        case .managed:
            guard streamedInstallTasks[id] == nil else {
                return "Wait for the SSD model download to stop, then remove it again."
            }
            let fileManager = FileManager.default
            do {
                try Self.moveExistingManagedArtifactsToTrash(
                    variant.managedInstallArtifactURLs,
                    fileExists: { fileManager.fileExists(atPath: $0) },
                    moveToTrash: { url in
                        try fileManager.trashItem(at: url, resultingItemURL: nil)
                    })
                setManagedStreamedModelEnabled(false, for: variant)
                unregisterStreamedModel(id)
                streamedInstallProgressByModelID[id] = nil
                streamedInstallErrors[id] = nil
                unsupportedReasonCache.removeValue(forKey: id)
                streamedStorageRevision &+= 1
                lastError = nil
                return nil
            } catch {
                streamedStorageRevision &+= 1
                let message = "Could not move \(variant.displayName) to Trash: \(error.localizedDescription)"
                lastError = message
                return message
            }
        case nil:
            unregisterStreamedModel(id)
            lastError = nil
            return nil
        }
    }

    nonisolated static func moveExistingManagedArtifactsToTrash(
        _ artifacts: [URL],
        fileExists: (String) -> Bool,
        moveToTrash: (URL) throws -> Void
    ) rethrows {
        for artifact in artifacts where fileExists(artifact.path) {
            try moveToTrash(artifact)
        }
    }

    // MARK: - Import

    /// Returns an error message, or nil on success (download started).
    func importModel(repoID: String) async -> String? {
        let trimmed = repoID.trimmingCharacters(in: .whitespaces)
        let pattern = /^[\w.\-]+\/[\w.\-]+$/
        guard trimmed.wholeMatch(of: pattern) != nil, HuggingFace.Repo.ID(rawValue: trimmed) != nil else {
            return "Enter a repo id like ModelAuthor/ModelName-MLX."
        }
        guard allModels.first(where: { $0.id == trimmed }) == nil else {
            return "This model is already in the list."
        }
        do {
            let url = URL(string: "https://huggingface.co/api/models/\(trimmed)")!
            let (_, response) = try await URLSession.shared.data(from: url)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                return "Repository not found on Hugging Face."
            }
        } catch {
            return "Could not reach Hugging Face: \(error.localizedDescription)"
        }
        // Compatibility preflight: fetch config.json and reject what this build
        // can't run (architecture / quantization) BEFORE pulling gigabytes of
        // weights — the load path enforces the same gate via ModelSupport.
        // Also sniffs vision capability so imported VLMs persist supportsVision.
        let supportsVision: Bool
        do {
            let url = URL(string: "https://huggingface.co/\(trimmed)/raw/main/config.json")!
            let (data, response) = try await URLSession.shared.data(from: url)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                return "Repository has no readable config.json — it doesn't look like an MLX model."
            }
            if let reason = ModelSupport.unsupportedReason(configJSON: data) {
                return reason
            }
            supportsVision = ModelSupport.visionCapable(configJSON: data)
        } catch {
            return "Could not verify the model config: \(error.localizedDescription)"
        }
        customModels.append(ModelCatalog.importedModel(repoID: trimmed, supportsVision: supportsVision))
        persistCustomModels()
        download(trimmed)
        return nil
    }

    func removeImported(_ id: String) {
        delete(id)
        customModels.removeAll { $0.id == id }
        persistCustomModels()
    }

    private func persistCustomModels() {
        if let data = try? JSONEncoder().encode(customModels) {
            UserDefaults.standard.set(data, forKey: Keys.customModels)
        }
    }

    // MARK: - Streamed MoE install registration

    /// Starts the bundled, resumable TurboFieldfare installer. It downloads
    /// directly into the managed `.gturbo` layout and registers it on success.
    func installStreamedModel(_ modelID: String) {
        guard let variant = StreamedMoEInstall.variant(
            forStreamedModelID: modelID),
              streamedInstallTasks[modelID] == nil else { return }
        if isAvailable(modelID) {
            streamedInstallProgressByModelID[modelID] = StreamedInstallProgress(
                phase: .ready,
                fraction: 1,
                downloadedBytes: variant.approximateDownloadBytes,
                totalBytes: variant.approximateDownloadBytes)
            return
        }
        if StreamedMoEInstall.validationError(
            at: variant.managedDirectory,
            for: variant) == nil {
            _ = registerManagedStreamedModelDirectory(
                variant.managedDirectory,
                variant: variant)
            return
        }

        streamedInstallErrors[modelID] = nil
        lastError = nil
        downloadProgress[modelID] = 0.01
        streamedInstallProgressByModelID[modelID] = StreamedInstallProgress(
            phase: .preparing,
            fraction: nil,
            downloadedBytes: nil,
            totalBytes: nil)

        let outputDirectory = variant.managedDirectory
        let installer = StreamedModelInstaller()
        streamedInstallers[modelID] = installer
        let task = Task { [weak self] in
            guard let self else { return }
            do {
                try await installer.install(
                    variant: variant,
                    at: outputDirectory) { [weak self] progress in
                    Task { @MainActor [weak self] in
                        self?.streamedInstallProgressByModelID[modelID] = progress
                        self?.downloadProgress[modelID] = progress.fraction ?? 0.01
                    }
                }
                try Task.checkCancellation()
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    if let error = registerManagedStreamedModelDirectory(
                        outputDirectory,
                        variant: variant,
                        requiresValidation: false) {
                        streamedInstallErrors[modelID] = error
                        lastError = error
                    }
                }
            } catch is CancellationError {
                // The helper keeps its verified checkpoint so the next click resumes.
            } catch {
                await MainActor.run { [weak self] in
                    let message = error.localizedDescription
                    self?.streamedInstallErrors[modelID] = message
                    self?.lastError = message
                }
            }
            await MainActor.run { [weak self] in
                self?.downloadProgress[modelID] = nil
                self?.streamedInstallTasks[modelID] = nil
                self?.streamedInstallers[modelID] = nil
            }
        }
        streamedInstallTasks[modelID] = task
    }

    private func registerManagedStreamedModelDirectory(
        _ directory: URL,
        variant: StreamedMoEVariant,
        requiresValidation: Bool = true
    ) -> String? {
        let normalized = directory.standardizedFileURL
        if requiresValidation, let error = StreamedMoEInstall.validationError(
            at: normalized,
            for: variant) {
            return error
        }
        streamedModelRegistrations[variant.modelID] = StreamedModelRegistration(
            directory: normalized,
            closesSecurityScope: false)
        setManagedStreamedModelEnabled(true, for: variant)
        streamedInstallProgressByModelID[variant.modelID] = StreamedInstallProgress(
            phase: .ready,
            fraction: 1,
            downloadedBytes: variant.approximateDownloadBytes,
            totalBytes: variant.approximateDownloadBytes)
        streamedInstallErrors[variant.modelID] = nil
        unsupportedReasonCache.removeValue(forKey: variant.modelID)
        return nil
    }

    /// Registers an existing, completed TurboFieldfare install without copying
    /// its multi-gigabyte payload. Returns an error message, or nil on success.
    func registerStreamedModelDirectory(modelID: String, directory: URL) -> String? {
        guard let variant = StreamedMoEInstall.variant(
            forStreamedModelID: modelID) else {
            return "This model does not use an SSD-streamed install."
        }
        let normalized = directory.standardizedFileURL
        if normalized == variant.managedDirectory.standardizedFileURL {
            return registerManagedStreamedModelDirectory(
                normalized,
                variant: variant)
        }
        let openedSecurityScope = normalized.startAccessingSecurityScopedResource()
        if let error = StreamedMoEInstall.validationError(
            at: normalized,
            for: variant) {
            if openedSecurityScope { normalized.stopAccessingSecurityScopedResource() }
            return error
        }

        do {
            let bookmark = try normalized.bookmarkData(
                options: [.withSecurityScope],
                includingResourceValuesForKeys: nil,
                relativeTo: nil)
            streamedModelRegistrations[modelID] = StreamedModelRegistration(
                directory: normalized,
                closesSecurityScope: openedSecurityScope)
            var bookmarks = Self.persistedStreamedModelBookmarks()
            bookmarks[modelID] = bookmark
            Self.persistStreamedModelBookmarks(bookmarks)
            unsupportedReasonCache.removeValue(forKey: modelID)
            lastError = nil
            streamedInstallErrors[modelID] = nil
            return nil
        } catch {
            if openedSecurityScope { normalized.stopAccessingSecurityScopedResource() }
            return "Could not save access to this model directory: \(error.localizedDescription)"
        }
    }

    /// Forgets the security-scoped location only. The external `.gturbo`
    /// install stays on disk and can be registered again later.
    func unregisterStreamedModel(_ id: String) {
        if let variant = StreamedMoEInstall.variant(forStreamedModelID: id),
           streamedModelRegistrations[id]?.directory.standardizedFileURL
            == variant.managedDirectory.standardizedFileURL {
            setManagedStreamedModelEnabled(false, for: variant)
        }
        streamedModelRegistrations.removeValue(forKey: id)
        var bookmarks = Self.persistedStreamedModelBookmarks()
        bookmarks.removeValue(forKey: id)
        Self.persistStreamedModelBookmarks(bookmarks)
        unsupportedReasonCache.removeValue(forKey: id)
    }

    private func restoreStreamedModelDirectories() {
        var bookmarks = Self.persistedStreamedModelBookmarks()
        if bookmarks[StreamedMoEInstall.standard.modelID] == nil,
           let legacyBookmark = bookmarks.removeValue(forKey: StreamedMoEInstall.legacyModelID) {
            bookmarks[StreamedMoEInstall.standard.modelID] = legacyBookmark
            Self.persistStreamedModelBookmarks(bookmarks)
        }
        for (id, bookmark) in bookmarks {
            guard let variant = StreamedMoEInstall.variant(
                forStreamedModelID: id) else { continue }
            guard let directory = Self.resolveStreamedModelBookmark(bookmark) else { continue }
            let openedSecurityScope = directory.startAccessingSecurityScopedResource()
            guard StreamedMoEInstall.validationError(
                at: directory,
                for: variant) == nil else {
                if openedSecurityScope { directory.stopAccessingSecurityScopedResource() }
                continue
            }
            streamedModelRegistrations[id] = StreamedModelRegistration(
                directory: directory,
                closesSecurityScope: openedSecurityScope)
        }
    }

    private func restoreManagedStreamedModelDirectory() {
        migrateLegacyManagedStreamedModelPreference()
        let preferences = Self.persistedManagedStreamedModelPreferences()
        for variant in StreamedMoEInstall.variants {
            guard streamedModelRegistrations[variant.modelID] == nil,
                  preferences[variant.modelID] != false,
                  StreamedMoEInstall.validationError(
                    at: variant.managedDirectory,
                    for: variant) == nil else { continue }
            streamedModelRegistrations[variant.modelID] = StreamedModelRegistration(
                directory: variant.managedDirectory,
                closesSecurityScope: false)
            setManagedStreamedModelEnabled(true, for: variant)
        }
    }

    private func migrateLegacyManagedStreamedModelPreference() {
        var preferences = Self.persistedManagedStreamedModelPreferences()
        let modelID = StreamedMoEInstall.standard.modelID
        guard preferences[modelID] == nil,
              let legacy = UserDefaults.standard.object(
                forKey: Keys.managedStreamedModelEnabled) as? Bool else { return }
        preferences[modelID] = legacy
        Self.persistManagedStreamedModelPreferences(preferences)
    }

    private func setManagedStreamedModelEnabled(
        _ enabled: Bool,
        for variant: StreamedMoEVariant
    ) {
        var preferences = Self.persistedManagedStreamedModelPreferences()
        preferences[variant.modelID] = enabled
        Self.persistManagedStreamedModelPreferences(preferences)
    }

    private nonisolated static func resolveStreamedModelBookmark(_ data: Data) -> URL? {
        var stale = false
        return try? URL(
            resolvingBookmarkData: data,
            options: [.withSecurityScope, .withoutUI],
            relativeTo: nil,
            bookmarkDataIsStale: &stale)
    }

    private nonisolated static func persistedStreamedModelBookmarks() -> [String: Data] {
        guard let data = UserDefaults.standard.data(forKey: Keys.streamedModelBookmarks) else {
            return [:]
        }
        return (try? JSONDecoder().decode([String: Data].self, from: data)) ?? [:]
    }

    private nonisolated static func persistStreamedModelBookmarks(_ bookmarks: [String: Data]) {
        guard let data = try? JSONEncoder().encode(bookmarks) else { return }
        UserDefaults.standard.set(data, forKey: Keys.streamedModelBookmarks)
    }

    private nonisolated static func persistedManagedStreamedModelPreferences()
        -> [String: Bool] {
        guard let data = UserDefaults.standard.data(
            forKey: Keys.managedStreamedModelPreferences) else { return [:] }
        return (try? JSONDecoder().decode([String: Bool].self, from: data)) ?? [:]
    }

    private nonisolated static func persistManagedStreamedModelPreferences(
        _ preferences: [String: Bool]
    ) {
        guard let data = try? JSONEncoder().encode(preferences) else { return }
        UserDefaults.standard.set(data, forKey: Keys.managedStreamedModelPreferences)
    }

    // MARK: - Storage stats

    /// Sum of regular-file sizes under `~/Models`, skipping symlinks
    /// (hub snapshots symlink into blobs; counting both would double the total).
    nonisolated static func diskUsageBytes() -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: modelsRoot,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey],
            options: []) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in enumerator {
            guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]),
                  values.isRegularFile == true, values.isSymbolicLink != true
            else { continue }
            total += Int64(values.fileSize ?? 0)
        }
        return total
    }

    nonisolated static func availableBytes() -> Int64 {
        let values = try? modelsRoot.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage ?? 0
    }
}
