import Darwin
import Foundation
import TurboFieldfareFormat

public struct VisionPackVerification: Sendable, Equatable {
    public let manifestSha256: String
    public let weightsBytes: UInt64
    public let sourceRevision: String
    public let sourceIndexSha256: String
    /// The text manifest this pack is bound to. Carried out so a successful
    /// verification can say *what it verified against*: two installs of the
    /// same revision are byte-identical, and without this the success line
    /// cannot distinguish which binding was checked.
    public let compatibleTextManifestSha256: String
}

public enum VisionPackVerifier {
    public static func verify(
        directory: URL,
        installedDirectory: URL? = nil,
        textModelDirectory: URL,
        verifyWeights: Bool = true,
        onHashProgress: ((UInt64, UInt64) throws -> Void)? = nil
    ) throws -> VisionPackVerification {
        let directory = directory.standardizedFileURL
        let installedDirectory = (installedDirectory ?? directory).standardizedFileURL
        let access = try GTurboDirectoryAccess(rootPath: directory.path)
        let expectedEntries = Set([
            GTurboVisionFormatV1.manifestFile,
            GTurboVisionFormatV1.weightsFile,
            GTurboVisionFormatV1.processorFile,
            GTurboVisionFormatV1.receiptFile,
        ])
        // Room for the four payload files plus every sidecar a round-trip can
        // leave beside them: `.DS_Store`, an AppleDouble `._x` per file, an
        // `.icloud` placeholder per file. The cap is what bounds the listing's
        // memory and its one fstat per entry, so it stays a small constant
        // instead of walking an arbitrarily large directory.
        let scanned = try access.relativeEntries(maxDepth: 0, maxEntries: 64)
        // Ignoring what the OS wrote beside the pack, exactly as the runtime
        // store does, so the installer and the loader agree on what a valid
        // pack directory is. A depth-0 scan yields basenames, which is what
        // isSidecarEntry matches on. Equality still rejects a missing, renamed
        // or genuinely unexpected file.
        let entries = Set(scanned.filter { !GTurboVisionFormatV1.isSidecarEntry($0) })
        guard entries == expectedEntries else {
            // Both directions, because a *missing* file used to be reported as
            // the surviving files being "unexpected", which sends the reader
            // looking for surplus entries that are not there.
            throw RepackError.configurationInvalid(
                detail: "vision companion directory does not match the v1 "
                    + "contract: " + GTurboVisionFormatV1.entryDifference(
                        expected: expectedEntries, actual: entries))
        }

        let manifestData = try access.readMetadata(
            GTurboVisionFormatV1.manifestFile,
            maxBytes: GTurboVisionFormatV1.metadataMaxBytes)
        let receiptData = try access.readMetadata(
            GTurboVisionFormatV1.receiptFile,
            maxBytes: GTurboVisionFormatV1.metadataMaxBytes)
        let manifest = try GTurboVisionManifestCodec.decode(manifestData)
        let receipt = try GTurboVisionReceiptCodec.decode(receiptData)

        let textAccess = try GTurboDirectoryAccess(rootPath: textModelDirectory.standardizedFileURL.path)
        let textManifestData = try textAccess.readMetadata(
            GTurboVisionFormatV1.manifestFile,
            maxBytes: GTurboVisionFormatV1.metadataMaxBytes)
        let textManifest = try GTurboManifestCodec.decode(textManifestData)
        guard let textSource = textManifest.sourceSnapshotHash,
              manifest.compatibleTextSourceSnapshotHash == textSource else {
            throw RepackError.installStateIncompatible(
                detail: "vision companion belongs to a different text checkpoint")
        }
        let textManifestSHA = hash(textManifestData)
        guard manifest.compatibleTextManifestSha256.lowercased() == textManifestSHA,
              receipt.compatibleTextManifestSha256.lowercased() == textManifestSHA else {
            throw RepackError.installStateIncompatible(
                detail: "vision companion text manifest binding changed")
        }

        let manifestSHA = hash(manifestData)
        let receiptDirectory = canonicalDirectoryPath(URL(
            fileURLWithPath: receipt.companionDirectoryPath,
            isDirectory: true))
        let expectedDirectory = canonicalDirectoryPath(installedDirectory)
        guard receipt.manifestSha256.lowercased() == manifestSHA else {
            throw RepackError.configurationInvalid(
                detail: "vision companion receipt manifest binding mismatch")
        }
        guard receiptDirectory == expectedDirectory else {
            throw RepackError.configurationInvalid(
                detail: "vision companion receipt directory binding mismatch")
        }
        guard receipt.sourceRepoID == manifest.modelID,
              receipt.sourceRevision == manifest.sourceRevision else {
            throw RepackError.configurationInvalid(
                detail: "vision companion receipt source binding mismatch")
        }

        guard let weightsEntry = manifest.files[GTurboVisionFormatV1.weightsFile] else {
            throw RepackError.configurationInvalid(
                detail: "vision manifest missing \(GTurboVisionFormatV1.weightsFile)")
        }

        var totalBytes: UInt64 = 0
        for (key, entry) in manifest.files
            where verifyWeights || key != GTurboVisionFormatV1.weightsFile {
            let sum = totalBytes.addingReportingOverflow(entry.size)
            guard !sum.overflow else {
                throw RepackError.configurationInvalid(
                    detail: "vision manifest byte total overflowed")
            }
            totalBytes = sum.partialValue
        }

        var hashedBytes: UInt64 = 0
        for relativePath in manifest.files.keys.sorted() {
            guard let expected = manifest.files[relativePath] else { continue }
            let fd = try access.openFile(relativePath)
            let actualSize: UInt64
            let actualSHA: String?
            do {
                actualSize = try access.fileSize(fileDescriptor: fd, relativePath: relativePath)
                guard actualSize == expected.size else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) size \(actualSize) != \(expected.size)")
                }
                guard receipt.files[relativePath] == expected else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) receipt mismatch")
                }
                if verifyWeights || relativePath != GTurboVisionFormatV1.weightsFile {
                    actualSHA = try access.hash(
                        fileDescriptor: fd,
                        relativePath: relativePath,
                        noCache: true)
                } else {
                    actualSHA = nil
                }
                close(fd)
            } catch {
                close(fd)
                throw error
            }
            if let actualSHA {
                let sum = hashedBytes.addingReportingOverflow(actualSize)
                guard !sum.overflow else {
                    throw RepackError.configurationInvalid(
                        detail: "vision verified byte total overflowed")
                }
                hashedBytes = sum.partialValue
                try onHashProgress?(hashedBytes, totalBytes)
                guard actualSHA.lowercased() == expected.sha256.lowercased() else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) hash mismatch")
                }
            }
        }
        guard let manifestReceipt = receipt.files[GTurboVisionFormatV1.manifestFile],
              manifestReceipt.size == UInt64(manifestData.count),
              manifestReceipt.sha256.lowercased() == manifestSHA else {
            throw RepackError.configurationInvalid(
                detail: "vision manifest receipt entry mismatch")
        }
        return VisionPackVerification(
            manifestSha256: manifestSHA,
            weightsBytes: weightsEntry.size,
            sourceRevision: manifest.sourceRevision,
            sourceIndexSha256: manifest.sourceIndexSha256,
            compatibleTextManifestSha256: textManifestSHA)
    }

    private static func hash(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }

    private static func canonicalDirectoryPath(_ url: URL) -> String {
        let standardized = url.standardizedFileURL
        return standardized.deletingLastPathComponent()
            .resolvingSymlinksInPath()
            .appendingPathComponent(standardized.lastPathComponent, isDirectory: true)
            .standardizedFileURL.path
    }
}
