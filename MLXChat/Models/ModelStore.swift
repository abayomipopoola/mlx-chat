import Foundation
import HuggingFace
import MLXLMCommon
import Observation

/// Owns model files on disk: downloads, deletion, usage, and the imported-model list.
///
/// Storage root is `~/Models` (the REAL home — the sandbox remaps `NSHomeDirectory`,
/// so it is resolved via getpwuid; access is granted by the home-relative sandbox
/// exception for `/Models/`). Two on-disk layouts are recognized:
/// - plain `~/Models/<org>/<name>/…` (pre-existing / hand-placed models)
/// - HubClient cache `~/Models/hub/models--<org>--<name>/snapshots/<sha>/…` (our downloads)
@MainActor
@Observable
final class ModelStore {
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

    init() {
        if let data = UserDefaults.standard.data(forKey: Keys.customModels),
           let models = try? JSONDecoder().decode([CatalogModel].self, from: data) {
            customModels = models
        } else {
            customModels = []
        }
        try? FileManager.default.createDirectory(at: Self.hubCacheDirectory, withIntermediateDirectories: true)
    }

    /// Catalog plus imported entries, in display order.
    var allModels: [CatalogModel] { ModelCatalog.models + customModels }

    func model(for id: String) -> CatalogModel? {
        allModels.first { $0.id == id }
    }

    // MARK: - On-disk state (purely filesystem-based; self-healing)

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

    /// Loadable local directory for a model: plain layout wins, hub snapshot second.
    nonisolated static func localModelDirectory(for id: String) -> URL? {
        if let plain = plainDirectory(for: id), directoryHasWeights(plain) {
            return plain
        }
        if let snapshot = newestSnapshot(for: id), directoryHasWeights(snapshot) {
            return snapshot
        }
        return nil
    }

    func isDownloaded(_ id: String) -> Bool {
        Self.localModelDirectory(for: id) != nil
    }

    var downloadedCount: Int { allModels.filter { isDownloaded($0.id) }.count }

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

    func download(_ id: String) {
        guard downloadTasks[id] == nil else { return }
        lastError = nil
        downloadProgress[id] = 0
        let task = Task { [weak self] in
            do {
                let downloader = HubDownloader(Self.hubClient)
                _ = try await downloader.download(
                    id: id, revision: "main",
                    matching: Self.downloadPatterns, useLatest: false,
                    progressHandler: { progress in
                        Task { @MainActor [weak self] in
                            self?.downloadProgress[id] = progress.fractionCompleted
                        }
                    })
            } catch is CancellationError {
                // Cancelled: partial blobs are resumable; leave them.
            } catch {
                await MainActor.run { [weak self] in
                    self?.lastError = "Download failed: \(error.localizedDescription)"
                }
            }
            await MainActor.run { [weak self] in
                self?.downloadProgress[id] = nil
                self?.downloadTasks[id] = nil
            }
        }
        downloadTasks[id] = task
    }

    /// Awaitable variant used by the lazy first-send path.
    func downloadAndWait(_ id: String) async throws {
        if isDownloaded(id) { return }
        download(id)
        if let task = downloadTasks[id] { await task.value }
        guard isDownloaded(id) else {
            throw EngineError.generationFailed(lastError ?? "Download failed.")
        }
    }

    func cancelDownload(_ id: String) {
        downloadTasks[id]?.cancel()
        downloadTasks[id] = nil
        downloadProgress[id] = nil
    }

    /// Removes BOTH layouts for the model.
    func delete(_ id: String) {
        if let plain = Self.plainDirectory(for: id) {
            try? FileManager.default.removeItem(at: plain)
        }
        if let hub = Self.hubRepoDirectory(for: id) {
            try? FileManager.default.removeItem(at: hub)
        }
    }

    // MARK: - Import

    /// Returns an error message, or nil on success (download started).
    func importModel(repoID: String) async -> String? {
        let trimmed = repoID.trimmingCharacters(in: .whitespaces)
        let pattern = /^[\w.\-]+\/[\w.\-]+$/
        guard trimmed.wholeMatch(of: pattern) != nil, HuggingFace.Repo.ID(rawValue: trimmed) != nil else {
            return "Enter a repo id like mlx-community/ModelName-4bit."
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
        customModels.append(ModelCatalog.importedModel(repoID: trimmed))
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
