import Foundation

/// Downloads a Hugging Face model snapshot into the plain `~/Models/<org>/<name>/`
/// layout using the *classic* `URLSessionDownloadTask` completion-handler API.
///
/// Why not `swift-huggingface`'s `downloadSnapshot`? Its `downloadFile` uses the
/// async `URLSession.download(for:delegate:)` variant, which hangs indefinitely on
/// Hugging Face's cross-host `302` redirect to the Xet CDN (`us.aws.cdn.hf.co`) for
/// large LFS weight files. Small files served inline download fine; the multi-GB
/// `.safetensors` shards stall at zero bytes forever (no error, no progress), which
/// is the "stuck at 0%" download failure. The classic completion-handler task follows
/// the identical redirect at full speed, so we drive the download ourselves and skip
/// the library's download path entirely.
enum SnapshotDownloader {
    struct RemoteFile: Sendable {
        let path: String
        let size: Int64
    }

    enum DownloadError: LocalizedError {
        case listFailed(String)
        case fileFailed(path: String, detail: String)
        case noFiles

        var errorDescription: String? {
            switch self {
            case .listFailed(let detail): return "Couldn't list model files: \(detail)"
            case .fileFailed(let path, let detail): return "Failed downloading \(path): \(detail)"
            case .noFiles: return "No downloadable model files were found in the repository."
            }
        }
    }

    /// Files an MLX load needs: weights, configs, tokenizer, and chat template.
    /// Mirrors `ModelStore.downloadPatterns` (`*.safetensors`, `*.json`, `*.jinja`).
    static func isWanted(_ path: String) -> Bool {
        let name = (path as NSString).lastPathComponent
        return name.hasSuffix(".safetensors") || name.hasSuffix(".json") || name.hasSuffix(".jinja")
    }

    /// Lists the repo's files at `main` via the public tree API. The `size` field
    /// carries the resolved LFS size, so it doubles as the progress weight.
    static func listFiles(repoID: String) async throws -> [RemoteFile] {
        guard var components = URLComponents(
            string: "https://huggingface.co/api/models/\(repoID)/tree/main") else {
            throw DownloadError.listFailed("bad repository id")
        }
        components.queryItems = [URLQueryItem(name: "recursive", value: "true")]
        guard let url = components.url else { throw DownloadError.listFailed("bad repository id") }

        let (data, response) = try await URLSession.shared.data(from: url)
        let status = (response as? HTTPURLResponse)?.statusCode ?? -1
        guard status == 200 else { throw DownloadError.listFailed("HTTP \(status)") }

        struct Entry: Decodable { let path: String; let type: String; let size: Int64? }
        let entries = try JSONDecoder().decode([Entry].self, from: data)
        return entries
            .filter { $0.type == "file" && isWanted($0.path) }
            .map { RemoteFile(path: $0.path, size: $0.size ?? 0) }
    }

    /// Downloads all wanted files into `destination` (a plain `~/Models/<org>/<name>`
    /// directory), reporting aggregate `0…1` progress. Files land in a `.partial`
    /// sibling first and are moved into place only once every file is present, so
    /// `destination` never appears as a half-complete (loadable-looking) model.
    ///
    /// Sequential by design: one big stream saturates the link, whereas parallel
    /// streams to the throttled public Xet bridge starve each other (and several
    /// time out entirely).
    static func download(
        repoID: String,
        destination: URL,
        onProgress: @escaping @Sendable (Double) -> Void
    ) async throws {
        let files = try await listFiles(repoID: repoID)
        guard !files.isEmpty else { throw DownloadError.noFiles }
        let totalBytes = max(files.reduce(Int64(0)) { $0 + $1.size }, 1)

        let staging = destination.deletingLastPathComponent()
            .appendingPathComponent(destination.lastPathComponent + ".partial", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)

        let session = URLSession(configuration: .default)
        defer { session.finishTasksAndInvalidate() }

        var completedBytes: Int64 = 0
        for file in files {
            try Task.checkCancellation()
            let fileURL = staging.appendingPathComponent(file.path)

            // Resume across restarts / re-clicks: a fully staged file is reused.
            if file.size > 0,
               let attrs = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
               (attrs[.size] as? Int64) == file.size {
                completedBytes += file.size
                onProgress(Double(completedBytes) / Double(totalBytes))
                continue
            }

            guard let src = URL(
                string: "https://huggingface.co/\(repoID)/resolve/main/\(file.path)") else {
                throw DownloadError.fileFailed(path: file.path, detail: "bad file url")
            }
            let base = completedBytes
            let fileSize = file.size
            try await downloadFile(from: src, to: fileURL, session: session) { fraction in
                let aggregate = Double(base) + fraction * Double(fileSize)
                onProgress(aggregate / Double(totalBytes))
            }
            completedBytes += file.size
            onProgress(Double(completedBytes) / Double(totalBytes))
        }

        // Publish atomically: the plain directory appears only when fully staged.
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: staging, to: destination)
    }

    /// Downloads one file with the classic completion-handler task (which follows the
    /// Xet redirect correctly) and reports its `0…1` progress from `task.progress`.
    private static func downloadFile(
        from url: URL,
        to destination: URL,
        session: URLSession,
        onFraction: @escaping @Sendable (Double) -> Void
    ) async throws {
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

        let box = TaskBox()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                let filename = url.lastPathComponent
                let task = session.downloadTask(with: url) { tempURL, response, error in
                    box.poller?.cancel()
                    if let error {
                        continuation.resume(throwing: DownloadError.fileFailed(
                            path: filename, detail: error.localizedDescription))
                        return
                    }
                    let status = (response as? HTTPURLResponse)?.statusCode ?? -1
                    guard (200..<300).contains(status) else {
                        continuation.resume(throwing: DownloadError.fileFailed(
                            path: filename, detail: "HTTP \(status)"))
                        return
                    }
                    guard let tempURL else {
                        continuation.resume(throwing: DownloadError.fileFailed(
                            path: filename, detail: "no downloaded file"))
                        return
                    }
                    // Must move synchronously: URLSession deletes tempURL after this returns.
                    do {
                        try? FileManager.default.removeItem(at: destination)
                        try FileManager.default.moveItem(at: tempURL, to: destination)
                        continuation.resume()
                    } catch {
                        continuation.resume(throwing: DownloadError.fileFailed(
                            path: filename, detail: error.localizedDescription))
                    }
                }
                box.task = task
                box.poller = Task {
                    while !Task.isCancelled {
                        onFraction(task.progress.fractionCompleted)
                        try? await Task.sleep(nanoseconds: 200_000_000)
                    }
                }
                task.resume()
            }
        } onCancel: {
            box.task?.cancel()
            box.poller?.cancel()
        }
    }

    /// Holds the in-flight task + progress poller so cancellation can reach them.
    private final class TaskBox: @unchecked Sendable {
        var task: URLSessionDownloadTask?
        var poller: Task<Void, Never>?
    }
}
