import Foundation
import TurboFieldfare

struct StreamedInstallProgress: Equatable, Sendable {
    enum Phase: Equatable, Sendable {
        case preparing
        case downloading
        case verifying
        case ready
    }

    let phase: Phase
    let fraction: Double?
    let downloadedBytes: UInt64?
    let totalBytes: UInt64?
    let isVision: Bool

    init(
        phase: Phase,
        fraction: Double?,
        downloadedBytes: UInt64?,
        totalBytes: UInt64?,
        isVision: Bool = false
    ) {
        self.phase = phase
        self.fraction = fraction
        self.downloadedBytes = downloadedBytes
        self.totalBytes = totalBytes
        self.isVision = isVision
    }

    var title: String {
        switch phase {
        case .preparing:
            return isVision ? "Preparing vision pack" : "Preparing download"
        case .downloading:
            return isVision
                ? "Downloading vision pack"
                : "Downloading and preparing for SSD"
        case .verifying:
            return isVision ? "Verifying vision pack" : "Verifying model"
        case .ready:
            return isVision ? "Vision pack ready" : "SSD model ready"
        }
    }
}

enum StreamedModelInstallerError: LocalizedError {
    case helperMissing
    case helperFailed(String)
    case invalidInstall(String)

    var errorDescription: String? {
        switch self {
        case .helperMissing:
            return "The SSD model installer is missing from this build."
        case .helperFailed(let detail):
            return detail.isEmpty ? "The SSD model download failed." : detail
        case .invalidInstall(let detail):
            return detail
        }
    }
}

/// Runs the pinned TurboFieldfare installer bundled inside the app. The helper
/// streams and repacks the source checkpoint directly into `.gturbo`; the app
/// never launches Terminal and never materializes a second safetensors snapshot.
final class StreamedModelInstaller: @unchecked Sendable {
    private final class ErrorTail: @unchecked Sendable {
        private let lock = NSLock()
        private var data = Data()
        private let limit = 64 * 1_024

        func append(_ newData: Data) {
            guard !newData.isEmpty else { return }
            lock.withLock {
                data.append(newData)
                if data.count > limit {
                    data.removeFirst(data.count - limit)
                }
            }
        }

        func text() -> String {
            lock.withLock {
                String(decoding: data, as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
    }

    private struct Checkpoint: Decodable {
        struct CompletedRange: Decodable {
            let sourceBytes: UInt64
        }

        let totalSourceBytes: UInt64
        let completedRanges: [CompletedRange]
    }

    static let helperExecutableName = "TurboFieldfareRepack"

    private let helperURL: URL?
    private let processLock = NSLock()
    private var runningProcess: Process?

    init(helperURL: URL? = StreamedModelInstaller.bundledHelperURL()) {
        self.helperURL = helperURL
    }

    static func bundledHelperURL(in bundle: Bundle = .main) -> URL? {
        guard let path = bundle.path(forAuxiliaryExecutable: helperExecutableName) else {
            return nil
        }
        return URL(fileURLWithPath: path)
    }

    func install(
        variant: StreamedMoEVariant,
        at outputDirectory: URL,
        companionOnly: Bool = false,
        onProgress: @escaping @Sendable (StreamedInstallProgress) -> Void
    ) async throws {
        guard let helperURL,
              FileManager.default.isExecutableFile(atPath: helperURL.path) else {
            throw StreamedModelInstallerError.helperMissing
        }

        try FileManager.default.createDirectory(
            at: outputDirectory.deletingLastPathComponent(),
            withIntermediateDirectories: true)

        if !companionOnly {
            try await runHelper(
                helperURL: helperURL,
                arguments: Self.textInstallArguments(
                    variant: variant, outputDirectory: outputDirectory),
                checkpointURL: StreamedModelInstallPaths(
                    finalDirectory: outputDirectory).checkpointFile,
                isVision: false,
                onProgress: onProgress)
            try Task.checkCancellation()
            onProgress(StreamedInstallProgress(
                phase: .verifying,
                fraction: 0.99,
                downloadedBytes: nil,
                totalBytes: nil))
            if let error = StreamedMoEInstall.validationError(
                at: outputDirectory,
                for: variant) {
                throw StreamedModelInstallerError.invalidInstall(error)
            }
        }

        if variant.architecture == .gemma4 {
            try await installVisionCompanion(
                helperURL: helperURL,
                variant: variant,
                textDirectory: outputDirectory,
                onProgress: onProgress)
        }

        let readyBytes = companionOnly
            ? StreamedMoEInstall.approximateVisionCompanionBytes
            : variant.approximateDownloadBytes
        onProgress(StreamedInstallProgress(
            phase: .ready,
            fraction: 1,
            downloadedBytes: readyBytes,
            totalBytes: readyBytes,
            isVision: companionOnly))
    }

    private func installVisionCompanion(
        helperURL: URL,
        variant: StreamedMoEVariant,
        textDirectory: URL,
        onProgress: @escaping @Sendable (StreamedInstallProgress) -> Void
    ) async throws {
        let companion = try VisionPackLocation.companionURL(forTextModel: textDirectory)
        try FileManager.default.createDirectory(
            at: companion.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try await runHelper(
            helperURL: helperURL,
            arguments: Self.visionInstallArguments(
                variant: variant,
                textDirectory: textDirectory,
                companionDirectory: companion),
            checkpointURL: StreamedModelInstallPaths(
                finalDirectory: companion).checkpointFile,
            isVision: true,
            onProgress: onProgress)
        try Task.checkCancellation()
        onProgress(StreamedInstallProgress(
            phase: .verifying,
            fraction: 0.99,
            downloadedBytes: nil,
            totalBytes: StreamedMoEInstall.approximateVisionCompanionBytes,
            isVision: true))
        guard StreamedMoEInstall.visionCompanionLooksPresent(beside: textDirectory) else {
            throw StreamedModelInstallerError.invalidInstall(
                "The vision companion pack was not written beside the SSD model.")
        }
    }

    static func textInstallArguments(
        variant: StreamedMoEVariant,
        outputDirectory: URL
    ) -> [String] {
        let paths = StreamedModelInstallPaths(finalDirectory: outputDirectory)
        var arguments = [
            "--model", variant.helperModelArgument,
            "--output", outputDirectory.path,
            "--overwrite",
        ]
        if FileManager.default.fileExists(atPath: paths.checkpointFile.path)
            && FileManager.default.fileExists(atPath: paths.partialDirectory.path) {
            arguments.append("--resume")
        }
        return arguments
    }

    static func visionInstallArguments(
        variant: StreamedMoEVariant,
        textDirectory: URL,
        companionDirectory: URL
    ) -> [String] {
        let paths = StreamedModelInstallPaths(finalDirectory: companionDirectory)
        var arguments = [
            "--write-vision",
            "--model", variant.helperModelArgument,
            "--text-model", textDirectory.path,
            "--output", companionDirectory.path,
            "--overwrite",
        ]
        if FileManager.default.fileExists(atPath: paths.checkpointFile.path)
            && FileManager.default.fileExists(atPath: paths.partialDirectory.path) {
            arguments.append("--resume")
        }
        return arguments
    }

    private func runHelper(
        helperURL: URL,
        arguments: [String],
        checkpointURL: URL,
        isVision: Bool,
        onProgress: @escaping @Sendable (StreamedInstallProgress) -> Void
    ) async throws {
        onProgress(StreamedInstallProgress(
            phase: .preparing,
            fraction: nil,
            downloadedBytes: nil,
            totalBytes: isVision
                ? StreamedMoEInstall.approximateVisionCompanionBytes
                : nil,
            isVision: isVision))

        let monitor = Task { [weak self] in
            while !Task.isCancelled {
                if let progress = self?.checkpointProgress(
                    at: checkpointURL, isVision: isVision) {
                    onProgress(progress)
                }
                try? await Task.sleep(for: .milliseconds(500))
            }
        }
        defer { monitor.cancel() }
        try await execute(helperURL: helperURL, arguments: arguments)
    }

    func cancel() {
        processLock.lock()
        let process = runningProcess
        processLock.unlock()
        if process?.isRunning == true { process?.terminate() }
    }

    func checkpointProgress(
        at checkpointURL: URL,
        isVision: Bool = false
    ) -> StreamedInstallProgress? {
        guard let data = try? Data(contentsOf: checkpointURL) else { return nil }
        return Self.checkpointProgress(data: data, isVision: isVision)
    }

    static func checkpointProgress(
        data: Data,
        isVision: Bool = false
    ) -> StreamedInstallProgress? {
        guard let checkpoint = try? JSONDecoder().decode(Checkpoint.self, from: data),
              checkpoint.totalSourceBytes > 0 else {
            return nil
        }
        let downloaded = checkpoint.completedRanges.reduce(UInt64(0)) { partial, range in
            let sum = partial.addingReportingOverflow(range.sourceBytes)
            return sum.overflow ? checkpoint.totalSourceBytes : sum.partialValue
        }
        let fraction = min(Double(downloaded) / Double(checkpoint.totalSourceBytes), 0.98)
        return StreamedInstallProgress(
            phase: .downloading,
            fraction: fraction,
            downloadedBytes: downloaded,
            totalBytes: checkpoint.totalSourceBytes,
            isVision: isVision)
    }

    private func execute(helperURL: URL, arguments: [String]) async throws {
        let process = Process()
        let standardError = Pipe()
        let errorTail = ErrorTail()
        process.executableURL = helperURL
        process.arguments = arguments
        process.standardOutput = FileHandle.nullDevice
        process.standardError = standardError
        standardError.fileHandleForReading.readabilityHandler = { handle in
            errorTail.append(handle.availableData)
        }

        processLock.withLock {
            runningProcess = process
        }

        defer {
            processLock.withLock {
                if runningProcess === process { runningProcess = nil }
            }
        }

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                process.terminationHandler = { completed in
                    standardError.fileHandleForReading.readabilityHandler = nil
                    errorTail.append(
                        standardError.fileHandleForReading.readDataToEndOfFile())
                    let errorText = errorTail.text()
                    if completed.terminationStatus == 0 {
                        continuation.resume()
                    } else {
                        continuation.resume(throwing:
                            StreamedModelInstallerError.helperFailed(errorText))
                    }
                }
                do {
                    try process.run()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
            try Task.checkCancellation()
        } onCancel: { [weak self] in
            self?.cancel()
        }
    }

}
