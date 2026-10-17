import Foundation
import TurboFieldfareFormat

public struct RemoteVisionCompanionOptions: Sendable {
    public let repoID: String
    public let revision: String
    public let textModelDirectory: String
    public let outputDirectory: String
    public let token: String?
    public let rangeChunkBytes: Int
    public let writeTileBytes: Int
    public let minFreeReserveBytes: UInt64
    public let overwrite: Bool
    public let resume: Bool
    public let downloadSession: RemoteDownloadSession
    public let baseURL: URL
    public let rangeRetryAttempts: Int
    public let retryBaseDelayNs: UInt64

    public init(repoID: String,
                revision: String,
                textModelDirectory: String,
                outputDirectory: String,
                token: String? = nil,
                rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
                writeTileBytes: Int = WriterCore.tileBytes,
                minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024,
                overwrite: Bool = false,
                resume: Bool = false,
                downloadSession: RemoteDownloadSession = RemoteDownloadSession(),
                baseURL: URL = URL(string: "https://huggingface.co")!,
                rangeRetryAttempts: Int = 4,
                retryBaseDelayNs: UInt64 = 1_000_000_000) {
        self.repoID = repoID
        self.revision = revision
        self.textModelDirectory = textModelDirectory
        self.outputDirectory = outputDirectory
        self.token = token
        self.rangeChunkBytes = rangeChunkBytes
        self.writeTileBytes = writeTileBytes
        self.minFreeReserveBytes = minFreeReserveBytes
        self.overwrite = overwrite
        self.resume = resume
        self.downloadSession = downloadSession
        self.baseURL = baseURL
        self.rangeRetryAttempts = rangeRetryAttempts
        self.retryBaseDelayNs = retryBaseDelayNs
    }
}

public struct RemoteVisionCompanionResult: Sendable {
    public let outputDir: String
    public let resolvedCommit: String
    public let tensorCount: Int
    public let sourcePayloadBytes: UInt64
    public let weightsFileSize: UInt64
    public let manifestSha256: String
    public let rangeRequestCount: Int
    public let remoteBytesToDownload: UInt64
    public let reusedBytes: UInt64
    public let downloadedThisRunBytes: UInt64
}

public final class RemoteVisionCompanionInstaller {
    private let options: RemoteVisionCompanionOptions
    private let audit: RepackAudit
    private let sourceProvider: SourceByteProvider?
    private let snapshotOverride: RemoteSnapshot?
    private let processorOverride: Data?

    public init(options: RemoteVisionCompanionOptions,
                audit: RepackAudit = RepackAudit()) {
        self.options = options
        self.audit = audit
        self.sourceProvider = nil
        self.snapshotOverride = nil
        self.processorOverride = nil
    }

    init(options: RemoteVisionCompanionOptions,
         audit: RepackAudit = RepackAudit(),
         sourceProvider: SourceByteProvider?,
         snapshot: RemoteSnapshot?,
         processorData: Data?) {
        self.options = options
        self.audit = audit
        self.sourceProvider = sourceProvider
        self.snapshotOverride = snapshot
        self.processorOverride = processorData
    }

    public func run(progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }) async throws
        -> RemoteVisionCompanionResult {
        try validateOptions()
        _ = try VisionPackWriter.loadTextBinding(
            textModelDirectory: options.textModelDirectory)
        let installLock = try InstallLock.acquire(outputDirectory: options.outputDirectory)
        defer { withExtendedLifetime(installLock) {} }
        let paths = installLock.paths
        if try Posix.entryKind(paths.finalDirectory) == .directory, !options.overwrite {
            throw RepackError.configurationInvalid(detail:
                "output directory already exists: \(paths.finalDirectory)")
        }
        let hasPartial = try Posix.entryKind(paths.partialDirectory) == .directory
        let hasCheckpoint = try Posix.entryKind(paths.checkpointFile) == .regular
        guard hasPartial == hasCheckpoint else {
            throw RepackError.installStateCorrupt(
                path: paths.partialDirectory,
                detail: "partial directory and checkpoint must exist together")
        }
        if options.resume {
            guard hasPartial else {
                throw RepackError.installStateMissing(path: paths.checkpointFile)
            }
        } else if hasPartial {
            throw RepackError.installStateIncompatible(
                detail: "saved download exists; resume or discard it")
        }
        do {
            return try await runPrepared(paths: paths, progress: progress)
        } catch {
            if !hasCheckpoint,
               (try? Posix.entryKind(paths.checkpointFile)) != .regular {
                try? FileManager.default.removeItem(atPath: paths.partialDirectory)
            }
            throw error
        }
    }

    private func runPrepared(paths: RemoteInstallPaths,
                             progress: @escaping @Sendable (ModelInstallProgress) -> Void) async throws
        -> RemoteVisionCompanionResult {
        try Task.checkCancellation()
        let saved = options.resume
            ? try RemoteInstallCheckpoint.load(from: paths.checkpointFile)
            : nil
        if let saved {
            guard saved.repoID == options.repoID,
                  saved.requestedRevision == options.revision else {
                throw RepackError.installStateIncompatible(
                    detail: "saved download belongs to a different source")
            }
        }
        let retryPolicy = RemoteRetryPolicy(attempts: options.rangeRetryAttempts,
                                            baseDelayNs: options.retryBaseDelayNs)
        let remote = HuggingFaceRemoteSource(repoID: options.repoID,
                                             requestedRevision: options.revision,
                                             resolvedCommit: saved?.resolvedCommit,
                                             token: options.token,
                                             downloadSession: options.downloadSession,
                                             baseURL: options.baseURL,
                                             tempDirectory: paths.partialDirectory,
                                             retryPolicy: retryPolicy)
        progress(.downloadingMetadata)
        let snapshot: RemoteSnapshot
        if let snapshotOverride {
            snapshot = snapshotOverride
        } else {
            snapshot = try await RemoteSnapshotLoader.load(
                remote: remote,
                requireKnownSource: true,
                metadataDirectory: paths.metadataDirectory,
                audit: audit)
        }
        try Task.checkCancellation()
        let modelID = try VisionPackWriter.acceptedSourceModelID(
            indexSha256: snapshot.metadata.indexSha256Hex)
        let plan = try RepackPlanner.planVisionCompanion(
            meta: snapshot.metadata, shardHeaders: snapshot.shardHeaders)
        let weightsPath = (paths.partialDirectory as NSString)
            .appendingPathComponent(GTurboVisionFormatV1.weightsFile)
        let rangePlan = try RangeCopyPlanner.plan(
            vision: plan,
            weightsPath: weightsPath,
            rangeChunkBytes: options.rangeChunkBytes)
        var checkpoint = saved ?? RemoteInstallCheckpoint(
            repoID: options.repoID,
            requestedRevision: options.revision,
            resolvedCommit: snapshot.resolvedCommit,
            sourceIndexSHA256: snapshot.metadata.indexSha256Hex,
            planFingerprint: rangePlan.canonicalFingerprint,
            totalSourceBytes: rangePlan.remoteBytesToDownload)
        if saved != nil {
            guard checkpoint.resolvedCommit == snapshot.resolvedCommit,
                  checkpoint.totalSourceBytes == rangePlan.remoteBytesToDownload,
                  checkpoint.matches(
                      repoID: options.repoID,
                      requestedRevision: options.revision,
                      sourceIndexSHA256: snapshot.metadata.indexSha256Hex,
                      planFingerprint: rangePlan.canonicalFingerprint) else {
                throw RepackError.installStateIncompatible(
                    detail: "saved download source or copy plan changed")
            }
            if try outputFilesMatch(plan: plan, rangePlan: rangePlan, weightsPath: weightsPath) {
                checkpoint.completedRanges = try RemoteStreamingRepacker.validatedCompletedRanges(
                    checkpoint.completedRanges,
                    copies: rangePlan.coalescedCopies,
                    partialDirectory: paths.partialDirectory)
            } else {
                checkpoint.completedRanges = []
                try FileManager.default.removeItem(atPath: paths.partialDirectory)
                try Posix.mkdirP(paths.partialDirectory)
                try createWeightsFile(plan: plan, weightsPath: weightsPath)
            }
            try checkpoint.write(
                to: paths.checkpointFile,
                parentDirectory: paths.parentDirectory)
        }

        progress(.planning(downloadBytes: rangePlan.remoteBytesToDownload,
                           outputBytes: plan.weightsFileSize))
        let reusedDestinationBytes = checkpoint.completedRanges.reduce(UInt64(0)) {
            $0 + $1.destinationBytes
        }
        let remainingOutputBytes = plan.weightsFileSize > reusedDestinationBytes
            ? plan.weightsFileSize - reusedDestinationBytes
            : 0
        let diskRequirement = try DiskSpaceChecker.requireAvailable(
            path: paths.parentDirectory,
            bytes: remainingOutputBytes + UInt64(options.rangeChunkBytes),
            reserveBytes: options.minFreeReserveBytes)
        progress(.checkingDisk(diskRequirement))
        try Task.checkCancellation()

        if saved == nil {
            progress(.reservingOutput(bytes: plan.weightsFileSize))
            try createWeightsFile(plan: plan, weightsPath: weightsPath)
            try checkpoint.write(
                to: paths.checkpointFile,
                parentDirectory: paths.parentDirectory)
        }

        let provider: SourceByteProvider
        if let sourceProvider {
            provider = sourceProvider
        } else {
            provider = HTTPRangeSourceByteProvider(
                remote: remote.pinned(commit: snapshot.resolvedCommit),
                files: snapshot.remoteFiles,
                writeTileBytes: options.writeTileBytes)
        }
        let reusedBytes = checkpoint.completedRanges.reduce(UInt64(0)) {
            $0 + $1.sourceBytes
        }
        let payloadDownloadStart = audit.remoteBytesDownloaded
        progress(.copyingPayload(
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: 0,
            totalBytes: rangePlan.remoteBytesToDownload))
        try await provider.copyBatch(
            rangePlan.coalescedCopies,
            completedRangeIDs: Set(checkpoint.completedRanges.map(\.id)),
            partialDirectory: paths.partialDirectory,
            temporaryPath: paths.rangeTemporaryFile,
            audit: audit,
            progress: { downloadedBytes in
                progress(.copyingPayload(
                    reusedBytes: reusedBytes,
                    downloadedThisRunBytes: downloadedBytes,
                    totalBytes: rangePlan.remoteBytesToDownload))
            },
            commit: { completed in
                checkpoint.completedRanges.removeAll { $0.id == completed.id }
                checkpoint.completedRanges.append(completed)
                checkpoint.completedRanges.sort { $0.id < $1.id }
                try checkpoint.write(
                    to: paths.checkpointFile,
                    parentDirectory: paths.parentDirectory)
            })

        progress(.hashingOutput(GTurboVisionFormatV1.weightsFile))
        let processorData: Data
        if let processorOverride {
            processorData = processorOverride
        } else {
            processorData = try await fetchProcessorConfig(
                remote: remote.pinned(commit: snapshot.resolvedCommit),
                outputPath: (paths.metadataDirectory as NSString)
                    .appendingPathComponent(GTurboVisionFormatV1.processorFile))
        }
        try? FileManager.default.removeItem(atPath: paths.rangeTemporaryFile)
        try? FileManager.default.removeItem(atPath: paths.metadataDirectory)
        progress(.finalizing)
        try Task.checkCancellation()
        let binding = try VisionPackWriter.loadTextBinding(
            textModelDirectory: options.textModelDirectory)
        let write = try VisionPackWriter.writeBoundSidecars(
            plan: plan,
            modelID: modelID,
            sourceRevision: snapshot.resolvedCommit,
            sourceIndexSha256: snapshot.metadata.indexSha256Hex,
            processorData: processorData,
            weightsPath: weightsPath,
            partialDirectory: paths.partialDirectory,
            finalURL: URL(fileURLWithPath: paths.finalDirectory, isDirectory: true),
            binding: binding,
            audit: audit)
        try? FileManager.default.removeItem(atPath: paths.checkpointFile)

        return RemoteVisionCompanionResult(
            outputDir: options.outputDirectory,
            resolvedCommit: snapshot.resolvedCommit,
            tensorCount: write.tensorCount,
            sourcePayloadBytes: write.sourcePayloadBytes,
            weightsFileSize: write.weightsFileSize,
            manifestSha256: write.manifestSha256,
            rangeRequestCount: rangePlan.coalescedCopies.count,
            remoteBytesToDownload: rangePlan.remoteBytesToDownload,
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: audit.remoteBytesDownloaded - payloadDownloadStart)
    }

    private func validateOptions() throws {
        guard options.rangeChunkBytes > 0,
              options.rangeChunkBytes <= RemoteChunkPolicy.maxBytes else {
            throw RepackError.configurationInvalid(detail: "bad range chunk bytes \(options.rangeChunkBytes)")
        }
        guard options.writeTileBytes > 0,
              options.writeTileBytes <= BoundedScratch.defaultLimitBytes else {
            throw RepackError.configurationInvalid(detail: "bad write tile bytes \(options.writeTileBytes)")
        }
        guard options.rangeRetryAttempts >= 0 else {
            throw RepackError.configurationInvalid(detail:
                "bad range retry attempts \(options.rangeRetryAttempts)")
        }
    }

    private func createWeightsFile(plan: VisionPackPlan, weightsPath: String) throws {
        try Posix.mkdirP((weightsPath as NSString).deletingLastPathComponent)
        let descriptor = try Posix.openCreateRW(weightsPath)
        try Posix.ftruncate(descriptor, path: weightsPath, size: plan.weightsFileSize)
        try Posix.fsync(descriptor, path: weightsPath)
        close(descriptor)
    }

    private func outputFilesMatch(plan: VisionPackPlan,
                                  rangePlan: RangeCopyPlan,
                                  weightsPath: String) throws -> Bool {
        guard try Posix.entryKind(weightsPath) == .regular else { return false }
        let descriptor = try Posix.openReadNoFollow(weightsPath)
        defer { close(descriptor) }
        return try Posix.fileSize(fd: descriptor, path: weightsPath) == plan.weightsFileSize
            && rangePlan.expectedOutputs.contains(where: {
                $0.relativePath == GTurboVisionFormatV1.weightsFile
                    && $0.size == plan.weightsFileSize
            })
    }

    private func fetchProcessorConfig(remote: HuggingFaceRemoteSource,
                                      outputPath: String) async throws -> Data {
        let info = try await remote.resolveFileInfo(
            filename: GTurboVisionFormatV1.processorFile, audit: audit)
        try await remote.fetchSmallFile(
            filename: GTurboVisionFormatV1.processorFile,
            info: info,
            capBytes: GTurboVisionFormatV1.metadataMaxBytes,
            outputPath: outputPath,
            audit: audit)
        return try Posix.readBoundedData(
            outputPath,
            maximumBytes: GTurboVisionFormatV1.metadataMaxBytes,
            followCallerSymlink: true)
    }
}

extension SupportedModelSource {
    public var supportsVisionCompanion: Bool {
        self == .gemma4 || self == .gemma4QAT
    }

    public func visionCompanionOptions(
        textModelDirectory: URL,
        outputDirectory: URL,
        overwrite: Bool,
        token: String?,
        resume: Bool = false
    ) throws -> RemoteVisionCompanionOptions {
        guard supportsVisionCompanion else {
            throw RepackError.configurationInvalid(
                detail: "vision-write accepts only gemma4 or gemma4-qat")
        }
        return RemoteVisionCompanionOptions(
            repoID: repoID,
            revision: revision,
            textModelDirectory: textModelDirectory.path,
            outputDirectory: outputDirectory.path,
            token: token,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume)
    }
}
