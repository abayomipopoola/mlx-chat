import Foundation

struct RemoteSnapshot {
    let metadata: IndexLoader.SourceMetadata
    let arch: ArchInfo
    let shardHeaders: [Safetensors.Header]
    let remoteFiles: [String: RemoteFileInfo]
    let resolvedCommit: String
    let metadataDirectory: String
}

enum RemoteSnapshotLoader {
    static func load(remote: HuggingFaceRemoteSource,
                     requireKnownSource: Bool,
                     metadataDirectory: String,
                     audit: RepackAudit? = nil) async throws -> RemoteSnapshot {
        try Posix.mkdirP(metadataDirectory)

        let indexInfo: RemoteFileInfo
        do {
            indexInfo = try await remote.resolveFileInfo(filename: "model.safetensors.index.json",
                                                         audit: audit)
        } catch {
            guard isRemoteNotFound(error) else { throw error }
            return try await loadWithoutPublishedIndex(remote: remote,
                                                       requireKnownSource: requireKnownSource,
                                                       metadataDirectory: metadataDirectory,
                                                       audit: audit)
        }
        let pinned = remote.pinned(commit: indexInfo.resolvedCommit)
        let configInfo = try await pinned.resolveFileInfo(filename: "config.json",
                                                          audit: audit)
        guard configInfo.resolvedCommit == indexInfo.resolvedCommit else {
            throw RepackError.remoteProtocolInvalid(detail: "config commit differs from index commit")
        }

        try await pinned.fetchSmallFile(filename: "model.safetensors.index.json",
                                        info: indexInfo,
                                        capBytes: 4 * 1024 * 1024,
                                        outputPath: (metadataDirectory as NSString)
                                            .appendingPathComponent("model.safetensors.index.json"),
                                        audit: audit)
        try await pinned.fetchSmallFile(filename: "config.json",
                                        info: configInfo,
                                        capBytes: 1024 * 1024,
                                        outputPath: (metadataDirectory as NSString)
                                            .appendingPathComponent("config.json"),
                                        audit: audit)

        let metadata = try IndexLoader.load(snapshotDir: metadataDirectory)
        if requireKnownSource && SourceFingerprint.modelID(forIndexSha256: metadata.indexSha256Hex) == nil {
            throw RepackError.sourceFingerprintRejected(path: metadata.indexPath,
                                                        sha256: metadata.indexSha256Hex)
        }
        let arch = try ArchInfo.load(configPath: metadata.configPath)

        var files: [String: RemoteFileInfo] = [
            indexInfo.filename: indexInfo,
            configInfo.filename: configInfo,
        ]
        var headers: [Safetensors.Header] = []
        headers.reserveCapacity(metadata.shardFilenames.count)
        for shard in metadata.shardFilenames {
            let info = try await pinned.resolveFileInfo(filename: shard, audit: audit)
            guard info.resolvedCommit == indexInfo.resolvedCommit else {
                throw RepackError.remoteProtocolInvalid(detail: "shard \(shard) commit differs from index commit")
            }
            guard info.acceptsRanges else {
                throw RepackError.remoteProtocolInvalid(detail: "shard \(shard) does not advertise byte ranges")
            }
            files[shard] = info
            let prefix = try await pinned.downloadRangeToTempFile(filename: shard,
                                                                 info: info,
                                                                 offset: 0,
                                                                 length: 8,
                                                                 audit: audit)
            defer { try? FileManager.default.removeItem(atPath: prefix.path) }
            let prefixData = try Data(contentsOf: URL(fileURLWithPath: prefix.path))
            guard prefixData.count == 8 else {
                throw RepackError.safetensorsHeaderInvalid(path: shard, detail: "short header prefix")
            }
            let headerSize = prefixData.withUnsafeBytes { raw -> UInt64 in
                var value: UInt64 = 0
                for i in 0..<8 {
                    value |= UInt64(raw[i]) << UInt64(i * 8)
                }
                return value
            }
            if headerSize > Safetensors.maxHeaderBytes || headerSize > info.size - 8 {
                throw RepackError.safetensorsHeaderTooLarge(path: shard, size: headerSize)
            }
            let headerFile = try await pinned.downloadRangeToTempFile(filename: shard,
                                                                     info: info,
                                                                     offset: 8,
                                                                     length: Int(headerSize),
                                                                     audit: audit)
            defer { try? FileManager.default.removeItem(atPath: headerFile.path) }
            let headerData = try Data(contentsOf: URL(fileURLWithPath: headerFile.path))
            headers.append(try Safetensors.parseHeaderBytes(path: shard,
                                                            fileSize: info.size,
                                                            headerBytes: headerData))
        }

        return RemoteSnapshot(metadata: metadata,
                              arch: arch,
                              shardHeaders: headers,
                              remoteFiles: files,
                              resolvedCommit: indexInfo.resolvedCommit,
                              metadataDirectory: metadataDirectory)
    }

    /// Flint publishes shard files and no `model.safetensors.index.json`.
    /// Build the canonical index from the shard headers so the rest of the
    /// installer, including the pinned SHA-256, stays on one path.
    private static func loadWithoutPublishedIndex(remote: HuggingFaceRemoteSource,
                                                 requireKnownSource: Bool,
                                                 metadataDirectory: String,
                                                 audit: RepackAudit?) async throws -> RemoteSnapshot {
        let configInfo = try await remote.resolveFileInfo(filename: "config.json", audit: audit)
        let pinned = remote.pinned(commit: configInfo.resolvedCommit)
        try await pinned.fetchSmallFile(filename: "config.json",
                                        info: configInfo,
                                        capBytes: 1024 * 1024,
                                        outputPath: (metadataDirectory as NSString)
                                            .appendingPathComponent("config.json"),
                                        audit: audit)

        var shardInfos: [RemoteFileInfo] = []
        for index in 0..<32 {
            let name = String(format: "model-%05d.safetensors", index)
            do {
                let info = try await pinned.resolveFileInfo(filename: name, audit: audit)
                guard info.resolvedCommit == configInfo.resolvedCommit else {
                    throw RepackError.remoteProtocolInvalid(
                        detail: "shard \(name) commit differs from config commit")
                }
                guard info.acceptsRanges else {
                    throw RepackError.remoteProtocolInvalid(
                        detail: "shard \(name) does not advertise byte ranges")
                }
                shardInfos.append(info)
            } catch {
                guard isRemoteNotFound(error) else { throw error }
                if index == 0 {
                    throw RepackError.remoteProtocolInvalid(detail: "checkpoint has no weight shards")
                }
                break
            }
        }

        var weightMap: [String: String] = [:]
        var totalSize: UInt64 = 0
        var headers: [Safetensors.Header] = []
        var files: [String: RemoteFileInfo] = [configInfo.filename: configInfo]
        for info in shardInfos {
            let header = try await fetchShardHeader(pinned: pinned, info: info, audit: audit)
            for tensor in header.tensors {
                if weightMap[tensor.name] != nil {
                    throw RepackError.configurationInvalid(
                        detail: "duplicate tensor \(tensor.name)")
                }
                weightMap[tensor.name] = info.filename
                totalSize += tensor.sizeBytes
            }
            headers.append(header)
            files[info.filename] = info
        }
        let indexData = IndexLoader.canonicalIndexJSON(weightMap: weightMap, totalSize: totalSize)
        let indexPath = (metadataDirectory as NSString)
            .appendingPathComponent("model.safetensors.index.json")
        try indexData.write(to: URL(fileURLWithPath: indexPath))

        let metadata = try IndexLoader.load(snapshotDir: metadataDirectory)
        if requireKnownSource && SourceFingerprint.modelID(forIndexSha256: metadata.indexSha256Hex) == nil {
            throw RepackError.sourceFingerprintRejected(path: metadata.indexPath,
                                                        sha256: metadata.indexSha256Hex)
        }
        let arch = try ArchInfo.load(configPath: metadata.configPath)
        return RemoteSnapshot(metadata: metadata,
                              arch: arch,
                              shardHeaders: headers,
                              remoteFiles: files,
                              resolvedCommit: configInfo.resolvedCommit,
                              metadataDirectory: metadataDirectory)
    }

    private static func fetchShardHeader(pinned: HuggingFaceRemoteSource,
                                        info: RemoteFileInfo,
                                        audit: RepackAudit?) async throws -> Safetensors.Header {
        let prefix = try await pinned.downloadRangeToTempFile(filename: info.filename,
                                                             info: info,
                                                             offset: 0,
                                                             length: 8,
                                                             audit: audit)
        defer { try? FileManager.default.removeItem(atPath: prefix.path) }
        let prefixData = try Data(contentsOf: URL(fileURLWithPath: prefix.path))
        guard prefixData.count == 8 else {
            throw RepackError.safetensorsHeaderInvalid(path: info.filename, detail: "short header prefix")
        }
        let headerSize = prefixData.withUnsafeBytes { raw -> UInt64 in
            var value: UInt64 = 0
            for i in 0..<8 {
                value |= UInt64(raw[i]) << UInt64(i * 8)
            }
            return value
        }
        if headerSize > Safetensors.maxHeaderBytes || headerSize > info.size - 8 {
            throw RepackError.safetensorsHeaderTooLarge(path: info.filename, size: headerSize)
        }
        let headerFile = try await pinned.downloadRangeToTempFile(filename: info.filename,
                                                                 info: info,
                                                                 offset: 8,
                                                                 length: Int(headerSize),
                                                                 audit: audit)
        defer { try? FileManager.default.removeItem(atPath: headerFile.path) }
        let headerData = try Data(contentsOf: URL(fileURLWithPath: headerFile.path))
        return try Safetensors.parseHeaderBytes(path: info.filename,
                                                fileSize: info.size,
                                                headerBytes: headerData)
    }

    private static func isRemoteNotFound(_ error: Error) -> Bool {
        if case RepackError.remoteHTTPStatus(_, 404) = error { return true }
        if case RepackError.remoteHTTPResponse(_, 404, _) = error { return true }
        return false
    }
}
