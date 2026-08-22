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
        guard var components = URLComponents(
            string: "https://huggingface.co/api/models/\(repoID)/tree/main") else {
            throw DownloadError.listFailed("bad repository id")
        }
        components.queryItems = [URLQueryItem(name: "recursive", value: "true")]
        guard let start = components.url else { throw DownloadError.listFailed("bad repository id") }

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
              let actual = sha256Hex(ofFileAt: url)
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

    private static func sha256Hex(ofFileAt url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk = (try? handle.read(upToCount: 1024 * 1024)) ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Selects one repository folder and removes that prefix locally. This lets
    /// repositories such as LiquidAI/LFM2.5-2.6B-MLX expose several precision
    /// folders while the MLX loader still sees config and weights at its root.
    static func localPath(for remotePath: String, subfolder: String?) -> String? {
        guard isWanted(remotePath) else { return nil }
        guard let subfolder, !subfolder.isEmpty else { return remotePath }
        let prefix = subfolder.hasSuffix("/") ? subfolder : subfolder + "/"
        guard remotePath.hasPrefix(prefix) else { return nil }
        let localPath = String(remotePath.dropFirst(prefix.count))
        return localPath.isEmpty ? nil : localPath
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
            let fileURL = staging.appendingPathComponent(file.localPath)

            // Resume across restarts / re-clicks: reuse only when size (and sha, if known) match.
            if canReuseStagedFile(at: fileURL, size: file.size, sha256: file.sha256) {
                completedBytes += file.size
                onProgress(Double(completedBytes) / Double(totalBytes))
                continue
            }

            guard let src = URL(
                string: "https://huggingface.co/\(repoID)/resolve/main/\(file.remotePath)") else {
                throw DownloadError.fileFailed(path: file.remotePath, detail: "bad file url")
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
