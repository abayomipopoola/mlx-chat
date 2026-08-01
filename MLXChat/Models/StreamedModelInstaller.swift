import Foundation

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

    var title: String {
        switch phase {
        case .preparing: return "Preparing download"
        case .downloading: return "Downloading and preparing for SSD"
        case .verifying: return "Verifying model"
        case .ready: return "SSD model ready"
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
        onProgress: @escaping @Sendable (StreamedInstallProgress) -> Void
    ) async throws {
        guard let helperURL,
              FileManager.default.isExecutableFile(atPath: helperURL.path) else {
            throw StreamedModelInstallerError.helperMissing
        }

        try FileManager.default.createDirectory(
            at: outputDirectory.deletingLastPathComponent(),
            withIntermediateDirectories: true)

        let paths = StreamedModelInstallPaths(finalDirectory: outputDirectory)
        let canResume = FileManager.default.fileExists(atPath: paths.checkpointFile.path)
            && FileManager.default.fileExists(atPath: paths.partialDirectory.path)

        var arguments = [
            "--model", variant.helperModelArgument,
            "--output", outputDirectory.path,
            "--overwrite",
        ]
        if canResume { arguments.append("--resume") }

        onProgress(StreamedInstallProgress(
            phase: .preparing,
            fraction: nil,
            downloadedBytes: nil,
            totalBytes: nil))

        let monitor = Task { [weak self] in
            while !Task.isCancelled {
                if let progress = self?.checkpointProgress(at: paths.checkpointFile) {
                    onProgress(progress)
                }
                try? await Task.sleep(for: .milliseconds(500))
            }
        }
        defer { monitor.cancel() }

        try await execute(helperURL: helperURL, arguments: arguments)
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
        onProgress(StreamedInstallProgress(
            phase: .ready,
            fraction: 1,
            downloadedBytes: variant.approximateDownloadBytes,
            totalBytes: variant.approximateDownloadBytes))
    }

    func cancel() {
        processLock.lock()
        let process = runningProcess
        processLock.unlock()
        if process?.isRunning == true { process?.terminate() }
    }

    func checkpointProgress(at checkpointURL: URL) -> StreamedInstallProgress? {
        guard let data = try? Data(contentsOf: checkpointURL) else { return nil }
        return Self.checkpointProgress(data: data)
    }

    static func checkpointProgress(data: Data) -> StreamedInstallProgress? {
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
            totalBytes: checkpoint.totalSourceBytes)
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
