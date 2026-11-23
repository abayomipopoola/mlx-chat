import AppKit
import Foundation
import HuggingFace
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
    /// Retaining the original scoped URL ensures `stopAccessingSecurityScopedResource`
    /// balances the exact URL instance opened before standardization.
    private final class StreamedModelRegistration {
        let directory: URL
        private let securityScopedURL: URL?

        init(directory: URL, securityScopedURL: URL?) {
            self.directory = directory.standardizedFileURL
            self.securityScopedURL = securityScopedURL
        }

        deinit {
            securityScopedURL?.stopAccessingSecurityScopedResource()
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
    private(set) var downloadErrors: [String: String] = [:]
    private(set) var customModels: [CatalogModel]
    private(set) var enabledMLXStreamedModelIDs: Set<String> = []
    var lastError: String?

#if DEBUG
    /// In-memory availability for native release-video captures. This never
    /// creates files, registers bookmarks, or changes a normal app launch.
    private let usesSeededDemoAvailability: Bool
    private let seededDemoModelIDs: Set<String>
    private let extraSeededAvailableIDs: Set<String>
#endif

    private var downloadTasks: [String: Task<Void, Never>] = [:]
    private var streamedInstallTasks: [String: Task<Void, Never>] = [:]
    private var streamedInstallers: [String: StreamedModelInstaller] = [:]
    private var streamedInstallProgressByModelID: [String: StreamedInstallProgress] = [:]
    private var streamedInstallErrors: [String: String] = [:]
    @ObservationIgnored
    private var nextDownloadToken: UInt64 = 0
    @ObservationIgnored
    private var activeDownloadTokens: [String: UInt64] = [:]
    @ObservationIgnored
    private var dyingDownloadTasks: [String: Task<Void, Never>] = [:]
    @ObservationIgnored
    private var dyingDownloadTokens: [String: UInt64] = [:]
    /// Invalidates filesystem-backed managed-install queries after a Trash move or install.
    private(set) var streamedStorageRevision = 0
    /// Validated `.gturbo` directories chosen by the user. MLX snapshots keep
    /// using the existing `~/Models` layouts and never enter this map.
    private var streamedModelRegistrations: [String: StreamedModelRegistration] = [:]
    /// Cached `ModelSupport.unsupportedReason(modelID:)` results; entry
    /// invalidated when a download finishes or the model is deleted.
    private var unsupportedReasonCache: [String: String?] = [:]
    /// Disk-probe cache for the local MLX layout. Hits and misses are stored.
    /// Bumping `availabilityRevision` makes `@Observable` re-render views.
    private let availabilityCache = ModelAvailabilityCache()
    private var availabilityRevision = 0
    @ObservationIgnored
    private var customModelsByID: [String: CatalogModel] = [:]
    @ObservationIgnored
    private var inFlightImportRepoIDs: Set<String> = []
    @ObservationIgnored
    private var visionPackPresenceCache: [String: Bool] = [:]
    @ObservationIgnored
    private nonisolated(unsafe) var becomeActiveObserver: NSObjectProtocol?
    @ObservationIgnored
    private nonisolated(unsafe) var becomeActiveDebounceTask: Task<Void, Never>?
    @ObservationIgnored
    private nonisolated(unsafe) var storageStatsTask: Task<Void, Never>?
    @ObservationIgnored
    private var storageStatsRevision: UInt64 = 0
    @ObservationIgnored
    var mlxDownloadHandler: (@Sendable (
        String, String?, URL, @escaping @Sendable (Double) -> Void
    ) async throws -> Void)?
    @ObservationIgnored
    var streamedInstallHandler: (@Sendable (
        StreamedModelInstaller, StreamedMoEVariant, URL, Bool,
        @escaping @Sendable (StreamedInstallProgress) -> Void
    ) async throws -> Void)?
    @ObservationIgnored
    var importFetchHandler: (@Sendable (URL) async throws -> (Data, URLResponse))?
    @ObservationIgnored
    var isAvailableOverride: ((String) -> Bool)?
    /// Last completed storage-stat snapshot. Refreshed off-main.
    private(set) var usedBytes: Int64 = 0
    private(set) var availableBytes: Int64 = 0
    private(set) var totalBytes: Int64 = 0

    var cachedUnsupportedReasonCount: Int { unsupportedReasonCache.count }

    init() {
#if DEBUG
        let arguments = ProcessInfo.processInfo.arguments
        // Exactly the four models the release-video dropdown shows: one dense
        // Liquid model, one dense Gemma, and the two SSD-streamed installs.
        let seedsPickerDemo = arguments.contains("--seed-picker-demo")
        usesSeededDemoAvailability =
            seedsPickerDemo || arguments.contains("--seed-streamed-demo-models")
        if seedsPickerDemo {
            seededDemoModelIDs = [
                "LiquidAI/LFM2.5-2.6B-MLX@bf16",
                StreamedMoEInstall.ornith15.modelID,
                "mlx-community/gemma-4-12B-it-4bit",
                StreamedMoEInstall.standard.modelID,
            ]
        } else {
            seededDemoModelIDs = usesSeededDemoAvailability
                ? Set(StreamedMoEInstall.variants.map(\.modelID))
                : []
        }
        extraSeededAvailableIDs = arguments.contains("--seed-ornith-demo-model")
            ? ["ornith-ai/Ornith-1.5-9B-MLX-4bit"]
            : []
        let seededCustomModels: [CatalogModel]? =
            arguments.contains("--seed-imported-models")
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
                .map { ModelCatalog.applyingKnownCapabilities(to: $0) }
                .filter { ModelCatalog.model(for: $0.id) == nil }
            customModels = upgraded
            if upgraded != models, let encoded = try? JSONEncoder().encode(upgraded) {
                UserDefaults.standard.set(encoded, forKey: Keys.customModels)
            }
        } else {
            customModels = []
        }
        customModelsByID = Dictionary(
            customModels.map { ($0.id, $0) },
            uniquingKeysWith: { first, _ in first })
        if let enabled = UserDefaults.standard.array(forKey: Keys.enabledMLXStreamedModelIDs) as? [String] {
            enabledMLXStreamedModelIDs = Set(enabled.filter {
                MLXStreamedInstall.variant(forStreamedModelID: $0) != nil
            })
        } else {
            enabledMLXStreamedModelIDs = []
        }
        restoreStreamedModelDirectories()
        restoreManagedStreamedModelDirectory()
        try? FileManager.default.createDirectory(at: Self.hubCacheDirectory, withIntermediateDirectories: true)
        refreshStorageStats()
        observeApplicationActivation()
    }

    deinit {
        if let observer = becomeActiveObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        becomeActiveDebounceTask?.cancel()
        storageStatsTask?.cancel()
    }

    /// Catalog plus imported entries, in display order.
    var allModels: [CatalogModel] { ModelCatalog.models + customModels }

    /// O(1) lookup across curated catalog and imported models.
    nonisolated static func lookupModel(
        id: String,
        customModelsByID: [String: CatalogModel]
    ) -> CatalogModel? {
        ModelCatalog.model(for: id) ?? customModelsByID[id]
    }

    func model(for id: String) -> CatalogModel? {
        guard let base = Self.lookupModel(id: id, customModelsByID: customModelsByID) else {
            return nil
        }
        let enriched = ModelCatalog.applyingKnownCapabilities(to: base)
        guard let variant = StreamedMoEInstall.variant(forStreamedModelID: id),
              variant.architecture == .gemma4,
              let directory = availableDirectory(for: id)
        else { return enriched }
        return cachedVisionPackPresent(beside: directory)
            ? enriched.withSupportsVision(true)
            : enriched
    }

    func gemmaVisionPackMissing(for streamedModelID: String) -> (missing: Bool, isManaged: Bool) {
        guard let variant = StreamedMoEInstall.variant(forStreamedModelID: streamedModelID),
              variant.architecture == .gemma4
        else { return (false, false) }
        let isManaged: Bool
        let directory: URL
        if let registered = availableDirectory(for: streamedModelID) {
            directory = registered
            isManaged = registered.standardizedFileURL == variant.managedDirectory.standardizedFileURL
        } else if managedStreamedInstallState(for: streamedModelID) == .complete {
            directory = variant.managedDirectory
            isManaged = true
        } else {
            return (false, false)
        }
        return (!cachedVisionPackPresent(beside: directory), isManaged)
    }

    private func cachedVisionPackPresent(beside directory: URL) -> Bool {
        let key = directory.standardizedFileURL.path
        if let cached = visionPackPresenceCache[key] {
            return cached
        }
        let present = StreamedMoEInstall.visionCompanionLooksPresent(beside: directory)
        visionPackPresenceCache[key] = present
        return present
    }

    // MARK: - Local availability

    /// `~/Models/<org>/<name>` for a repo id, when the id has the org/name shape.
    nonisolated static func plainDirectory(for id: String) -> URL? {
        let parts = id.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2 else { return nil }
        let org = String(parts[0])
        let name = String(parts[1])
        guard !org.isEmpty, !name.isEmpty,
              org != ".", org != "..",
              name != ".", name != "..",
              !org.contains("\0"), !name.contains("\0")
        else { return nil }
        return modelsRoot
            .appendingPathComponent(org, isDirectory: true)
            .appendingPathComponent(name, isDirectory: true)
    }

    nonisolated static func directoryHasWeights(_ url: URL) -> Bool {
        directoryHasWeights(url, hasPositiveWeightFile: regularFileHasPositiveSize)
    }

    /// Verifies that a model directory has `config.json`, no sibling `.partial`,
    /// and every unique weight shard listed in `model.safetensors.index.json`
    /// (checked once per distinct shard file, not once per tensor entry).
    nonisolated static func directoryHasWeights(
        _ url: URL,
        hasPositiveWeightFile: (URL) -> Bool
    ) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.appendingPathComponent("config.json").path) else { return false }
        let siblingPartial = url.deletingLastPathComponent()
            .appendingPathComponent(url.lastPathComponent + ".partial", isDirectory: true)
        if fm.fileExists(atPath: siblingPartial.path) { return false }

        let indexURL = url.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: indexURL.path) {
            guard let data = try? Data(contentsOf: indexURL),
                  let index = try? JSONDecoder().decode(SafetensorsWeightIndex.self, from: data)
            else { return false }
            let uniqueShards = Set(index.weightMap.values)
            guard !uniqueShards.isEmpty else { return false }
            return uniqueShards.allSatisfy { shard in
                hasPositiveWeightFile(url.appendingPathComponent(shard))
            }
        }

        let contents = (try? fm.contentsOfDirectory(atPath: url.path)) ?? []
        return contents.contains { name in
            name.hasSuffix(".safetensors")
                && hasPositiveWeightFile(url.appendingPathComponent(name))
        }
    }

    nonisolated static func regularFileHasPositiveSize(_ url: URL) -> Bool {
        let resolved = url.resolvingSymlinksInPath()
        guard let values = try? resolved.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
              values.isRegularFile == true,
              let size = values.fileSize
        else { return false }
        return size > 0
    }

    private struct SafetensorsWeightIndex: Decodable {
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
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
        _ = availabilityRevision
        if StreamedMoEInstall.variant(forStreamedModelID: id) != nil {
            return streamedModelRegistrations[id]?.directory
        }
        if let variant = MLXStreamedInstall.variant(forStreamedModelID: id) {
            guard enabledMLXStreamedModelIDs.contains(id) else { return nil }
            return availabilityCache.directory(for: variant.sourceModelID)
        }
        return availabilityCache.directory(for: id)
    }

    func isAvailable(_ id: String) -> Bool {
        if let isAvailableOverride {
            return isAvailableOverride(id)
        }
#if DEBUG
        if extraSeededAvailableIDs.contains(id) { return true }
        if usesSeededDemoAvailability {
            return seededDemoModelIDs.contains(id)
        }
#endif
        return availableDirectory(for: id) != nil
    }

    func isMLXStreamedEnabled(_ id: String) -> Bool {
        enabledMLXStreamedModelIDs.contains(id)
    }

    func enableMLXStreamedModel(_ id: String) {
        guard MLXStreamedInstall.variant(forStreamedModelID: id) != nil else { return }
        if enabledMLXStreamedModelIDs.insert(id).inserted {
            persistEnabledMLXStreamedModelIDs()
        }
        invalidateModelLookupCaches(for: id)
    }

    func disableMLXStreamedModel(_ id: String) {
        if enabledMLXStreamedModelIDs.remove(id) != nil {
            persistEnabledMLXStreamedModelIDs()
        }
        invalidateModelLookupCaches(for: id)
    }

    func enableAndDownloadMLXStreamedModel(_ id: String) {
        guard let variant = MLXStreamedInstall.variant(forStreamedModelID: id) else { return }
        enableMLXStreamedModel(id)
        if availabilityCache.directory(for: variant.sourceModelID) == nil,
           downloadTasks[variant.sourceModelID] == nil {
            download(variant.sourceModelID)
        }
    }

    private func persistEnabledMLXStreamedModelIDs() {
        UserDefaults.standard.set(
            enabledMLXStreamedModelIDs.sorted(),
            forKey: Keys.enabledMLXStreamedModelIDs)
    }

    private func invalidateAvailability(for id: String) {
        availabilityCache.invalidate(id)
        visionPackPresenceCache.removeAll()
        availabilityRevision &+= 1
    }

    private func invalidateModelLookupCaches(for id: String) {
        unsupportedReasonCache.removeValue(forKey: id)
        if let dependent = MLXStreamedInstall.variant(forSourceModelID: id) {
            unsupportedReasonCache.removeValue(forKey: dependent.modelID)
            availabilityCache.invalidate(dependent.modelID)
        }
        invalidateAvailability(for: id)
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
        if let variant = MLXStreamedInstall.variant(forStreamedModelID: id) {
            return unsupportedReason(for: variant.sourceModelID)
        }
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

    func isDownloading(_ id: String) -> Bool {
        if let variant = MLXStreamedInstall.variant(forStreamedModelID: id) {
            return enabledMLXStreamedModelIDs.contains(id)
                && downloadProgress[variant.sourceModelID] != nil
        }
        return downloadProgress[id] != nil
    }

    func streamedInstallProgress(for modelID: String) -> StreamedInstallProgress? {
        streamedInstallProgressByModelID[modelID]
    }

    func streamedInstallError(for modelID: String) -> String? {
        streamedInstallErrors[modelID]
    }

    func downloadError(for modelID: String) -> String? {
        if let variant = MLXStreamedInstall.variant(forStreamedModelID: modelID) {
            return downloadErrors[variant.sourceModelID]
        }
        return streamedInstallErrors[modelID] ?? downloadErrors[modelID]
    }

    private func issueDownloadToken(for id: String) -> UInt64 {
        nextDownloadToken &+= 1
        activeDownloadTokens[id] = nextDownloadToken
        return nextDownloadToken
    }

    private func isCurrentDownload(_ id: String, token: UInt64) -> Bool {
        activeDownloadTokens[id] == token
    }

    func download(_ id: String) {
        if MLXStreamedInstall.variant(forStreamedModelID: id) != nil {
            enableAndDownloadMLXStreamedModel(id)
            return
        }
        guard downloadTasks[id] == nil else { return }
        if StreamedMoEInstall.variant(forStreamedModelID: id) != nil {
            installStreamedModel(id)
            return
        }
        if let restriction = model(for: id)?.installRestriction {
            downloadErrors[id] = restriction
            lastError = restriction
            return
        }
        // Downloads land in the plain `~/Models/<org>/<name>` layout, which
        // `localMLXModelDirectory` resolves first, so the load path finds them offline
        // and never falls back to the (broken) library `.id` downloader. See
        // `SnapshotDownloader` for why we don't use the vendored `downloadSnapshot`.
        guard let destination = Self.plainDirectory(for: id) else {
            let message = "Invalid model id: \(id)"
            downloadErrors[id] = message
            lastError = message
            return
        }
        downloadErrors[id] = nil
        lastError = nil
        downloadProgress[id] = 0
        let sourceModel = model(for: id)
        let sourceSubfolder = sourceModel?.sourceSubfolder
        let sourceRepoID = sourceModel?.huggingFaceRepoID ?? id
        let token = issueDownloadToken(for: id)
        let priorTask = dyingDownloadTasks[id]
        let customHandler = mlxDownloadHandler
        let task = Task { [weak self] in
            if let priorTask {
                await priorTask.value
            }
            do {
                try Task.checkCancellation()
                let progressHandler: @Sendable (Double) -> Void = { fraction in
                    Task { @MainActor [weak self] in
                        guard let self, self.isCurrentDownload(id, token: token) else { return }
                        self.downloadProgress[id] = fraction
                    }
                }
                if let customHandler {
                    try await customHandler(sourceRepoID, sourceSubfolder, destination, progressHandler)
                } else {
                    try await SnapshotDownloader.download(
                        repoID: sourceRepoID, subfolder: sourceSubfolder, destination: destination,
                        onProgress: progressHandler)
                }
            } catch is CancellationError {
                // Cancelled: the `.partial` staging dir is left for resume.
            } catch {
                await MainActor.run { [weak self] in
                    guard let self, self.isCurrentDownload(id, token: token) else { return }
                    let message = "Download failed: \(error.localizedDescription)"
                    self.downloadErrors[id] = message
                    self.lastError = message
                }
            }
            await MainActor.run { [weak self] in
                guard let self else { return }
                if self.dyingDownloadTokens[id] == token {
                    self.dyingDownloadTokens[id] = nil
                    self.dyingDownloadTasks[id] = nil
                }
                if self.isCurrentDownload(id, token: token) {
                    self.activeDownloadTokens[id] = nil
                    self.downloadProgress[id] = nil
                    self.downloadTasks[id] = nil
                    self.invalidateModelLookupCaches(for: id)
                    self.refreshStorageStats()
                }
            }
        }
        downloadTasks[id] = task
    }

    /// Awaitable variant used by the lazy first-send path.
    func downloadAndWait(_ id: String) async throws {
        if isAvailable(id) { return }
        download(id)
        let trackedDownloadID = MLXStreamedInstall.variant(forStreamedModelID: id)?.sourceModelID ?? id
        if let task = downloadTasks[trackedDownloadID] ?? streamedInstallTasks[id] {
            await task.value
        }
        guard isAvailable(id) else {
            let message = downloadError(for: id) ?? lastError ?? "Download failed."
            throw EngineError.generationFailed(message)
        }
    }

    func cancelDownload(_ id: String) {
        let resolvedID = MLXStreamedInstall.variant(forStreamedModelID: id)?.sourceModelID ?? id
        let token = activeDownloadTokens.removeValue(forKey: resolvedID)
        if StreamedMoEInstall.variant(forStreamedModelID: id) != nil {
            if let task = streamedInstallTasks.removeValue(forKey: id) {
                task.cancel()
                if let token {
                    dyingDownloadTasks[id] = task
                    dyingDownloadTokens[id] = token
                }
            }
            streamedInstallers.removeValue(forKey: id)?.cancel()
            streamedInstallProgressByModelID[id] = nil
        }
        if let task = downloadTasks.removeValue(forKey: resolvedID) {
            task.cancel()
            if let token {
                dyingDownloadTasks[resolvedID] = task
                dyingDownloadTokens[resolvedID] = token
            }
        }
        downloadProgress[resolvedID] = nil
    }

    /// Removes an app-owned install, forgets an external registration, or
    /// disables an opt-in MLX streamed entry while keeping shared weights.
    @discardableResult
    func delete(_ id: String) -> String? {
        if MLXStreamedInstall.variant(forStreamedModelID: id) != nil {
            disableMLXStreamedModel(id)
            lastError = nil
            refreshStorageStats()
            return nil
        }
        if let variant = StreamedMoEInstall.variant(forStreamedModelID: id) {
            let message = removeStreamedModel(id, variant: variant)
            refreshStorageStats()
            return message
        }
        if downloadTasks[id] != nil {
            cancelDownload(id)
        }
        var artifacts: [URL] = []
        if let plain = Self.plainDirectory(for: id) {
            artifacts.append(plain)
            artifacts.append(
                plain.deletingLastPathComponent()
                    .appendingPathComponent(plain.lastPathComponent + ".partial", isDirectory: true))
        }
        if let hub = Self.hubRepoDirectory(for: id) {
            artifacts.append(hub)
        }
        let fileManager = FileManager.default
        do {
            try Self.moveExistingManagedArtifactsToTrash(
                artifacts,
                fileExists: { fileManager.fileExists(atPath: $0) },
                moveToTrash: { url in
                    try fileManager.trashItem(at: url, resultingItemURL: nil)
                })
            if let dependent = MLXStreamedInstall.variant(forSourceModelID: id) {
                disableMLXStreamedModel(dependent.modelID)
            }
            invalidateModelLookupCaches(for: id)
            refreshStorageStats()
            downloadErrors[id] = nil
            lastError = nil
            return nil
        } catch {
            invalidateModelLookupCaches(for: id)
            refreshStorageStats()
            let name = model(for: id)?.displayName ?? id
            let message = "Could not move \(name) to Trash: \(error.localizedDescription)"
            lastError = message
            return message
        }
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
            guard streamedInstallTasks[id] == nil && dyingDownloadTasks[id] == nil else {
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
                invalidateModelLookupCaches(for: id)
                streamedStorageRevision &+= 1
                lastError = nil
                return nil
            } catch {
                invalidateAvailability(for: id)
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

    nonisolated static func validateImportRepoID(_ raw: String) -> String? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = trimmed.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2 else { return nil }
        let org = String(parts[0])
        let name = String(parts[1])
        guard !org.isEmpty, !name.isEmpty,
              org != ".", org != "..",
              name != ".", name != ".."
        else { return nil }
        let isAllowedASCII: (UnicodeScalar) -> Bool = { scalar in
            (scalar >= "A" && scalar <= "Z")
                || (scalar >= "a" && scalar <= "z")
                || (scalar >= "0" && scalar <= "9")
                || scalar == "." || scalar == "_" || scalar == "-"
        }
        guard org.unicodeScalars.allSatisfy(isAllowedASCII),
              name.unicodeScalars.allSatisfy(isAllowedASCII),
              HuggingFace.Repo.ID(rawValue: trimmed) != nil
        else { return nil }
        return trimmed
    }

    nonisolated static func huggingFaceMetadataURL(repoID: String) -> URL? {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/api/models/\(repoID)"
        return components.url
    }

    nonisolated static func huggingFaceConfigURL(repoID: String) -> URL? {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/\(repoID)/raw/main/config.json"
        return components.url
    }

    func containsModelCaseInsensitive(_ id: String) -> Bool {
        let needle = id.lowercased()
        return allModels.contains { $0.id.lowercased() == needle }
    }

    /// Returns an error message, or nil on success (download started).
    func importModel(repoID: String) async -> String? {
        guard let trimmed = Self.validateImportRepoID(repoID) else {
            return "Enter a repo id like ModelAuthor/ModelName-MLX."
        }
        let key = trimmed.lowercased()
        guard !containsModelCaseInsensitive(trimmed), !inFlightImportRepoIDs.contains(key) else {
            return "This model is already in the list."
        }
        guard let metadataURL = Self.huggingFaceMetadataURL(repoID: trimmed),
              let configURL = Self.huggingFaceConfigURL(repoID: trimmed) else {
            return "Enter a repo id like ModelAuthor/ModelName-MLX."
        }
        inFlightImportRepoIDs.insert(key)
        defer { inFlightImportRepoIDs.remove(key) }

        let fetcher = importFetchHandler ?? { url in
            try await URLSession.shared.data(from: url)
        }
        do {
            let (_, response) = try await fetcher(metadataURL)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                return "Repository not found on Hugging Face."
            }
        } catch is CancellationError {
            return nil
        } catch let error as URLError where error.code == .cancelled {
            return nil
        } catch {
            return "Could not reach Hugging Face: \(error.localizedDescription)"
        }
        // Compatibility preflight: fetch config.json and reject what this build
        // can't run (architecture / quantization) BEFORE pulling gigabytes of
        // weights — the load path enforces the same gate via ModelSupport.
        // Also sniffs vision capability so imported VLMs persist supportsVision.
        let supportsVision: Bool
        let configJSON: Data
        do {
            let (data, response) = try await fetcher(configURL)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                return "Repository has no readable config.json — it doesn't look like an MLX model."
            }
            if let reason = ModelSupport.unsupportedReason(configJSON: data) {
                return reason
            }
            supportsVision = ModelSupport.visionCapable(configJSON: data)
            configJSON = data
        } catch is CancellationError {
            return nil
        } catch let error as URLError where error.code == .cancelled {
            return nil
        } catch {
            return "Could not verify the model config: \(error.localizedDescription)"
        }
        guard !Task.isCancelled else { return nil }
        guard !containsModelCaseInsensitive(trimmed) else {
            return "This model is already in the list."
        }
        let imported = ModelCatalog.importedModel(
            repoID: trimmed, supportsVision: supportsVision, configJSON: configJSON)
        customModels.append(imported)
        customModelsByID[imported.id] = imported
        persistCustomModels()
        download(trimmed)
        return nil
    }

    @discardableResult
    func removeImported(_ id: String) -> String? {
        removeImported(id) { self.delete($0) }
    }

    @discardableResult
    func removeImported(_ id: String, deleteAction: (String) -> String?) -> String? {
        if let error = deleteAction(id) {
            return error
        }
        customModels.removeAll { $0.id == id }
        customModelsByID.removeValue(forKey: id)
        persistCustomModels()
        return nil
    }

    func registerCustomModelForTesting(_ model: CatalogModel) {
        customModels.append(model)
        customModelsByID[model.id] = model
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
        let textIsValid = StreamedMoEInstall.validationError(
            at: variant.managedDirectory,
            for: variant) == nil
        let companionOnly = variant.architecture == .gemma4
            && textIsValid
            && variant.needsVisionCompanion
        if isAvailable(modelID), !companionOnly {
            streamedInstallProgressByModelID[modelID] = StreamedInstallProgress(
                phase: .ready,
                fraction: 1,
                downloadedBytes: variant.approximateDownloadBytes,
                totalBytes: variant.approximateDownloadBytes)
            return
        }
        if textIsValid, !companionOnly {
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
            totalBytes: companionOnly
                ? StreamedMoEInstall.approximateVisionCompanionBytes
                : nil,
            isVision: companionOnly)

        let outputDirectory = variant.managedDirectory
        let installer = StreamedModelInstaller()
        streamedInstallers[modelID] = installer
        let token = issueDownloadToken(for: modelID)
        let priorTask = dyingDownloadTasks[modelID]
        let customHandler = streamedInstallHandler
        let task = Task { [weak self] in
            guard let self else { return }
            if let priorTask {
                await priorTask.value
            }
            do {
                try Task.checkCancellation()
                let progressHandler: @Sendable (StreamedInstallProgress) -> Void = { [weak self] progress in
                    Task { @MainActor [weak self] in
                        guard let self, self.isCurrentDownload(modelID, token: token) else { return }
                        self.streamedInstallProgressByModelID[modelID] = progress
                        self.downloadProgress[modelID] = progress.fraction ?? 0.01
                    }
                }
                if let customHandler {
                    try await customHandler(
                        installer, variant, outputDirectory, companionOnly, progressHandler)
                } else {
                    try await installer.install(
                        variant: variant,
                        at: outputDirectory,
                        companionOnly: companionOnly,
                        onProgress: progressHandler)
                }
                try Task.checkCancellation()
                await MainActor.run { [weak self] in
                    guard let self, self.isCurrentDownload(modelID, token: token) else { return }
                    if let error = registerManagedStreamedModelDirectory(
                        outputDirectory,
                        variant: variant,
                        requiresValidation: false) {
                        streamedInstallErrors[modelID] = error
                        lastError = error
                    } else if companionOnly {
                        streamedInstallProgressByModelID[modelID] = StreamedInstallProgress(
                            phase: .ready,
                            fraction: 1,
                            downloadedBytes:
                                StreamedMoEInstall.approximateVisionCompanionBytes,
                            totalBytes:
                                StreamedMoEInstall.approximateVisionCompanionBytes,
                            isVision: true)
                    }
                }
            } catch is CancellationError {
                // The helper keeps its verified checkpoint so the next click resumes.
            } catch {
                await MainActor.run { [weak self] in
                    guard let self, self.isCurrentDownload(modelID, token: token) else { return }
                    let message = error.localizedDescription
                    self.streamedInstallErrors[modelID] = message
                    self.lastError = message
                }
            }
            await MainActor.run { [weak self] in
                guard let self else { return }
                if self.dyingDownloadTokens[modelID] == token {
                    self.dyingDownloadTokens[modelID] = nil
                    self.dyingDownloadTasks[modelID] = nil
                }
                if self.isCurrentDownload(modelID, token: token) {
                    self.activeDownloadTokens[modelID] = nil
                    self.downloadProgress[modelID] = nil
                    self.streamedInstallTasks[modelID] = nil
                    self.streamedInstallers[modelID] = nil
                }
                self.streamedStorageRevision &+= 1
                self.refreshStorageStats()
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
            securityScopedURL: nil)
        // Clear any stale external bookmark so it cannot shadow the managed install on next launch.
        Self.dropPersistedStreamedModelBookmark(variant.modelID)
        setManagedStreamedModelEnabled(true, for: variant)
        streamedInstallProgressByModelID[variant.modelID] = StreamedInstallProgress(
            phase: .ready,
            fraction: 1,
            downloadedBytes: variant.approximateDownloadBytes,
            totalBytes: variant.approximateDownloadBytes)
        streamedInstallErrors[variant.modelID] = nil
        downloadErrors[variant.modelID] = nil
        invalidateModelLookupCaches(for: variant.modelID)
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
        let openedSecurityScope = directory.startAccessingSecurityScopedResource()
        if let error = StreamedMoEInstall.validationError(
            at: normalized,
            for: variant) {
            if openedSecurityScope { directory.stopAccessingSecurityScopedResource() }
            return error
        }

        do {
            let bookmark = try directory.bookmarkData(
                options: [.withSecurityScope],
                includingResourceValuesForKeys: nil,
                relativeTo: nil)
            streamedModelRegistrations[modelID] = StreamedModelRegistration(
                directory: normalized,
                securityScopedURL: openedSecurityScope ? directory : nil)
            var bookmarks = Self.persistedStreamedModelBookmarks()
            bookmarks[modelID] = bookmark
            Self.persistStreamedModelBookmarks(bookmarks)
            invalidateModelLookupCaches(for: modelID)
            lastError = nil
            streamedInstallErrors[modelID] = nil
            downloadErrors[modelID] = nil
            return nil
        } catch {
            if openedSecurityScope { directory.stopAccessingSecurityScopedResource() }
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
        invalidateModelLookupCaches(for: id)
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
            guard let directory = Self.resolveStreamedModelBookmark(bookmark, id: id) else { continue }
            let openedSecurityScope = directory.startAccessingSecurityScopedResource()
            guard StreamedMoEInstall.validationError(
                at: directory,
                for: variant) == nil else {
                if openedSecurityScope { directory.stopAccessingSecurityScopedResource() }
                Self.dropPersistedStreamedModelBookmark(id)
                continue
            }
            streamedModelRegistrations[id] = StreamedModelRegistration(
                directory: directory,
                securityScopedURL: openedSecurityScope ? directory : nil)
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
                securityScopedURL: nil)
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

    private nonisolated static func resolveStreamedModelBookmark(_ data: Data, id: String) -> URL? {
        var stale = false
        let url: URL
        do {
            url = try URL(
                resolvingBookmarkData: data,
                options: [.withSecurityScope, .withoutUI],
                relativeTo: nil,
                bookmarkDataIsStale: &stale)
        } catch {
            dropPersistedStreamedModelBookmark(id)
            return nil
        }
        if stale {
            refreshStaleStreamedModelBookmark(url, id: id)
        }
        return url
    }

    private nonisolated static func dropPersistedStreamedModelBookmark(_ id: String) {
        var bookmarks = persistedStreamedModelBookmarks()
        bookmarks.removeValue(forKey: id)
        persistStreamedModelBookmarks(bookmarks)
    }

    private nonisolated static func refreshStaleStreamedModelBookmark(_ url: URL, id: String) {
        let accessed = url.startAccessingSecurityScopedResource()
        defer {
            if accessed { url.stopAccessingSecurityScopedResource() }
        }
        guard let replacement = try? url.bookmarkData(
            options: .withSecurityScope,
            includingResourceValuesForKeys: nil,
            relativeTo: nil)
        else { return }
        var bookmarks = persistedStreamedModelBookmarks()
        bookmarks[id] = replacement
        persistStreamedModelBookmarks(bookmarks)
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

    // MARK: - External ~/Models refresh

    /// Re-probe `~/Models` after the user may have added or removed files
    /// while the app was in the background. No filesystem watcher.
    func noteExternalModelDirectoryMayHaveChanged() {
        availabilityCache.invalidateAll()
        visionPackPresenceCache.removeAll()
        unsupportedReasonCache.removeAll()
        availabilityRevision &+= 1
        streamedStorageRevision &+= 1
        refreshStorageStats()
    }

    private func observeApplicationActivation() {
        becomeActiveObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.scheduleExternalModelDirectoryRefresh()
            }
        }
    }

    private func scheduleExternalModelDirectoryRefresh() {
        becomeActiveDebounceTask?.cancel()
        becomeActiveDebounceTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            self?.noteExternalModelDirectoryMayHaveChanged()
        }
    }

    // MARK: - Storage stats

    func refreshStorageStats() {
        refreshStorageStats {
            await Task.detached(priority: .utility) {
                (Self.diskUsageBytes(), Self.availableBytes(), Self.volumeCapacityBytes())
            }.value
        }
    }

    func refreshStorageStats(
        compute: @escaping @Sendable () async -> (used: Int64, available: Int64, total: Int64)
    ) {
        storageStatsTask?.cancel()
        storageStatsRevision &+= 1
        let revision = storageStatsRevision
        storageStatsTask = Task { [weak self] in
            let stats = await compute()
            guard let self, !Task.isCancelled, self.storageStatsRevision == revision else { return }
            self.usedBytes = stats.used
            self.availableBytes = stats.available
            self.totalBytes = stats.total
        }
    }

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

    nonisolated static func volumeCapacityBytes() -> Int64 {
        let values = try? modelsRoot.resourceValues(forKeys: [.volumeTotalCapacityKey])
        return Int64(values?.volumeTotalCapacity ?? 0)
    }
}
