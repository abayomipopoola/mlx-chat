import CryptoKit
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
        let remotePath: String
        let localPath: String
        let size: Int64
        let sha256: String?
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

    /// Lists the repo's files at `main` via the public tree API. Follows RFC 8288
    /// `Link: rel="next"` until that relation is absent. The `size` field carries
    /// the resolved LFS size, so it doubles as the progress weight.
    static func listFiles(repoID: String, subfolder: String? = nil) async throws -> [RemoteFile] {
        guard let start = treeListURL(repoID: repoID) else {
            throw DownloadError.listFailed("bad repository id")
        }

        var collected: [RemoteFile] = []
        var seen = Set<URL>()
        var current: URL? = start
        while let url = current, seen.insert(url.absoluteURL).inserted {
            let (data, response) = try await URLSession.shared.data(from: url)
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            guard status == 200 else { throw DownloadError.listFailed("HTTP \(status)") }
            collected.append(contentsOf: try remoteFiles(fromTreeJSON: data, subfolder: subfolder))
            if let http = response as? HTTPURLResponse,
               let header = http.value(forHTTPHeaderField: "Link"),
               let next = nextLink(from: header) {
                current = URL(string: next, relativeTo: url)?.absoluteURL
            } else {
                current = nil
            }
        }
        return collected
    }

    /// RFC 8288 `Link` header: the URI-Reference of the first `rel="next"` link.
    static func nextLink(from linkHeader: String) -> String? {
        var rest = Substring(linkHeader)
        while !rest.isEmpty {
            rest = rest.drop(while: { $0.isWhitespace })
            guard rest.first == "<", let close = rest.firstIndex(of: ">") else { return nil }
            let uri = String(rest[rest.index(after: rest.startIndex)..<close])
            rest = rest[rest.index(after: close)...]

            var isNext = false
            while !rest.isEmpty {
                rest = rest.drop(while: { $0.isWhitespace })
                if rest.first == "," {
                    rest = rest.dropFirst()
                    break
                }
                guard rest.first == ";" else { return isNext ? uri : nil }
                rest = rest.dropFirst()
                rest = rest.drop(while: { $0.isWhitespace })
                let nameEnd = rest.firstIndex(where: { $0 == "=" || $0 == ";" || $0 == "," || $0.isWhitespace })
                    ?? rest.endIndex
                let name = rest[..<nameEnd].lowercased()
                rest = rest[nameEnd...]
                rest = rest.drop(while: { $0.isWhitespace })
                var value = ""
                if rest.first == "=" {
                    rest = rest.dropFirst()
                    rest = rest.drop(while: { $0.isWhitespace })
                    if rest.first == "\"" {
                        rest = rest.dropFirst()
                        if let endQuote = rest.firstIndex(of: "\"") {
                            value = String(rest[..<endQuote])
                            rest = rest[rest.index(after: endQuote)...]
                        }
                    } else {
                        let valueEnd = rest.firstIndex(where: { $0 == ";" || $0 == "," }) ?? rest.endIndex
                        value = rest[..<valueEnd].trimmingCharacters(in: .whitespaces)
                        rest = rest[valueEnd...]
                    }
                }
                if name == "rel" {
                    let rels = value.lowercased().split(whereSeparator: { $0.isWhitespace }).map(String.init)
                    if rels.contains("next") { isNext = true }
                }
            }
            if isNext { return uri }
        }
        return nil
    }

    /// True when the file's SHA-256 hex digest equals `expected` (optional `sha256:` prefix ignored).
    static func fileSHA256(_ url: URL, matches expected: String) -> Bool {
        guard let want = normalizedSHA256(expected),
              let actual = try? sha256Hex(ofFileAt: url)
        else { return false }
        return actual.caseInsensitiveCompare(want) == .orderedSame
    }

    static func canReuseStagedFile(at url: URL, size: Int64, sha256: String?) -> Bool {
        guard size > 0,
              let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              ((attrs[.size] as? NSNumber)?.int64Value ?? -1) == size
        else { return false }
        guard let sha256 else { return true }
        return fileSHA256(url, matches: sha256)
    }

    /// Verifies a freshly downloaded temporary file against the repository's
    /// expected byte count and SHA-256 digest before moving it into staging.
    static func verifyDownloadedFile(
        at url: URL,
        expectedSize: Int64,
        expectedSHA256: String?,
        remotePath: String
    ) throws {
        if expectedSize > 0 {
            guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
                  let actualSize = (attrs[.size] as? NSNumber)?.int64Value
            else {
                throw DownloadError.fileFailed(
                    path: remotePath,
                    detail: "couldn't read downloaded file size")
            }
            guard actualSize == expectedSize else {
                throw DownloadError.fileFailed(
                    path: remotePath,
                    detail: "size mismatch (got \(actualSize) bytes, expected \(expectedSize))")
            }
        }
        if let expectedSHA256 {
            guard fileSHA256(url, matches: expectedSHA256) else {
                throw DownloadError.fileFailed(
                    path: remotePath,
                    detail: "SHA-256 checksum mismatch")
            }
        }
    }

    private static func remoteFiles(fromTreeJSON data: Data, subfolder: String?) throws -> [RemoteFile] {
        struct LFS: Decodable { let oid: String? }
        struct Entry: Decodable {
            let path: String
            let type: String
            let size: Int64?
            let lfs: LFS?
        }
        let entries = try JSONDecoder().decode([Entry].self, from: data)
        return entries.compactMap { entry -> RemoteFile? in
            guard entry.type == "file",
                  let localPath = localPath(for: entry.path, subfolder: subfolder) else {
                return nil
            }
            return RemoteFile(
                remotePath: entry.path,
                localPath: localPath,
                size: entry.size ?? 0,
                sha256: normalizedSHA256(entry.lfs?.oid))
        }
    }

    static func normalizedSHA256(_ oid: String?) -> String? {
        guard var value = oid?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else {
            return nil
        }
        if value.lowercased().hasPrefix("sha256:") {
            value = String(value.dropFirst(7))
        }
        return value.isEmpty ? nil : value
    }

    static func treeListURL(repoID: String) -> URL? {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/api/models/\(repoID)/tree/main"
        components.queryItems = [URLQueryItem(name: "recursive", value: "true")]
        return components.url
    }

    static func resolveFileURL(repoID: String, remotePath: String) -> URL? {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/\(repoID)/resolve/main/\(remotePath)"
        return components.url
    }

    static func sha256Hex(ofFileAt url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        return try sha256Hex {
            try handle.read(upToCount: 1024 * 1024)
        }
    }

    static func sha256Hex(readChunk: () throws -> Data?) throws -> String {
        var hasher = SHA256()
        while let chunk = try readChunk(), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Selects one repository folder and removes that prefix locally. This lets
    /// repositories such as LiquidAI/LFM2.5-2.6B-MLX expose several precision
    /// folders while the MLX loader still sees config and weights at its root.
    ///
    /// Weights below the model root are skipped: MLX merges every `.safetensors` it finds
    /// under the directory into one strict load, so an optional draft head such as
    /// Bonsai 2's `mtp/model.safetensors` would fail it.
    static func localPath(for remotePath: String, subfolder: String?) -> String? {
        guard isWanted(remotePath) else { return nil }
        let localPath: String
        if let subfolder, !subfolder.isEmpty {
            let prefix = subfolder.hasSuffix("/") ? subfolder : subfolder + "/"
            guard remotePath.hasPrefix(prefix) else { return nil }
            localPath = String(remotePath.dropFirst(prefix.count))
        } else {
            localPath = remotePath
        }
        guard !localPath.isEmpty,
              !localPath.hasPrefix("/"),
              !localPath.contains("\0")
        else { return nil }
        let segments = localPath.split(separator: "/", omittingEmptySubsequences: false)
        if segments.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }) {
            return nil
        }
        if localPath.hasSuffix(".safetensors") {
            if localPath.contains("/") || localPath == "mtp.safetensors" { return nil }
        }
        return localPath
    }

    static func safeStagedFileURL(staging: URL, localPath: String) -> URL? {
        guard !localPath.isEmpty,
              !localPath.hasPrefix("/"),
              !localPath.contains("\0")
        else { return nil }
        let segments = localPath.split(separator: "/", omittingEmptySubsequences: false)
        if segments.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }) {
            return nil
        }
        let base = staging.standardizedFileURL
        let resolved = base.appendingPathComponent(localPath).standardizedFileURL
        let prefix = base.path.hasSuffix("/") ? base.path : base.path + "/"
        guard resolved.path.hasPrefix(prefix) else { return nil }
        return resolved
    }

    static func publishStagedDirectory(staging: URL, to destination: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fm.fileExists(atPath: destination.path) {
            _ = try fm.replaceItemAt(destination, withItemAt: staging)
        } else {
            try fm.moveItem(at: staging, to: destination)
        }
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
        subfolder: String? = nil,
        destination: URL,
        onProgress: @escaping @Sendable (Double) -> Void
    ) async throws {
        let files = try await listFiles(repoID: repoID, subfolder: subfolder)
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
            guard let fileURL = safeStagedFileURL(staging: staging, localPath: file.localPath) else {
                throw DownloadError.fileFailed(path: file.remotePath, detail: "unsafe file path")
            }

            // Resume across restarts / re-clicks: reuse only when size (and sha, if known) match.
            if canReuseStagedFile(at: fileURL, size: file.size, sha256: file.sha256) {
                completedBytes += file.size
                onProgress(Double(completedBytes) / Double(totalBytes))
                continue
            }

            guard let src = resolveFileURL(repoID: repoID, remotePath: file.remotePath) else {
                throw DownloadError.fileFailed(path: file.remotePath, detail: "bad file url")
            }
            let base = completedBytes
            let fileSize = file.size
            try await downloadFile(
                from: src,
                to: fileURL,
                expectedSize: file.size,
                expectedSHA256: file.sha256,
                remotePath: file.remotePath,
                session: session
            ) { fraction in
                let aggregate = Double(base) + fraction * Double(fileSize)
                onProgress(aggregate / Double(totalBytes))
            }
            completedBytes += file.size
            onProgress(Double(completedBytes) / Double(totalBytes))
        }

        // Publish atomically: the plain directory appears only when fully staged.
        try publishStagedDirectory(staging: staging, to: destination)
    }

    static func mapDownloadFailure(
        _ error: Error,
        filename: String,
        isCancelled: Bool
    ) -> Error {
        if isCancelled || Task.isCancelled {
            return CancellationError()
        }
        if let urlError = error as? URLError, urlError.code == .cancelled {
            return CancellationError()
        }
        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain, nsError.code == NSURLErrorCancelled {
            return CancellationError()
        }
        return DownloadError.fileFailed(path: filename, detail: error.localizedDescription)
    }

    /// Downloads one file with the classic completion-handler task (which follows the
    /// Xet redirect correctly) and reports its `0…1` progress from `task.progress`.
    private static func downloadFile(
        from url: URL,
        to destination: URL,
        expectedSize: Int64,
        expectedSHA256: String?,
        remotePath: String,
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
                    box.stopPoller()
                    if let error {
                        continuation.resume(throwing: mapDownloadFailure(
                            error, filename: filename, isCancelled: box.isCancelled))
                        return
                    }
                    if box.isCancelled || Task.isCancelled {
                        continuation.resume(throwing: CancellationError())
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
                    // Verify on URLSession's temp file before moving into staging:
                    // on mismatch, throwing leaves staging untouched and URLSession
                    // cleans up tempURL automatically.
                    do {
                        try verifyDownloadedFile(
                            at: tempURL,
                            expectedSize: expectedSize,
                            expectedSHA256: expectedSHA256,
                            remotePath: remotePath)
                        if FileManager.default.fileExists(atPath: destination.path) {
                            _ = try FileManager.default.replaceItemAt(destination, withItemAt: tempURL)
                        } else {
                            try FileManager.default.moveItem(at: tempURL, to: destination)
                        }
                        continuation.resume()
                    } catch let error as DownloadError {
                        continuation.resume(throwing: error)
                    } catch {
                        continuation.resume(throwing: DownloadError.fileFailed(
                            path: filename, detail: error.localizedDescription))
                    }
                }
                let started = box.start(
                    task: task,
                    makePoller: {
                        Task {
                            while !Task.isCancelled {
                                onFraction(task.progress.fractionCompleted)
                                try? await Task.sleep(nanoseconds: 200_000_000)
                            }
                        }
                    },
                    resume: { task.resume() })
                if !started {
                    continuation.resume(throwing: CancellationError())
                }
            }
        } onCancel: {
            box.cancel()
        }
    }

    /// Holds the in-flight task + progress poller under a lock so cancellation
    /// racing with task startup never calls `resume()` after cancel.
    final class TaskBox: @unchecked Sendable {
        private let lock = NSLock()
        private var task: URLSessionDownloadTask?
        private var poller: Task<Void, Never>?
        private var cancelled = false

        var isCancelled: Bool {
            lock.withLock { cancelled }
        }

        func start(
            task: URLSessionDownloadTask?,
            makePoller: () -> Task<Void, Never>?,
            resume: () -> Void
        ) -> Bool {
            lock.withLock {
                if cancelled || Task.isCancelled {
                    return false
                }
                self.task = task
                self.poller = makePoller()
                resume()
                return true
            }
        }

        func stopPoller() {
            let pollerToCancel: Task<Void, Never>? = lock.withLock {
                let current = poller
                poller = nil
                return current
            }
            pollerToCancel?.cancel()
        }

        func cancel() {
            let (taskToCancel, pollerToCancel): (URLSessionDownloadTask?, Task<Void, Never>?) = lock.withLock {
                cancelled = true
                let currentTask = task
                let currentPoller = poller
                poller = nil
                return (currentTask, currentPoller)
            }
            taskToCancel?.cancel()
            pollerToCancel?.cancel()
        }
    }
}
