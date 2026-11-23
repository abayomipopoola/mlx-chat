import Foundation
import Darwin
import TurboFieldfareFormat

public struct VerifyInstallOptions: Sendable {
    public let inputGTurbo: String

    public init(inputGTurbo: String) {
        self.inputGTurbo = inputGTurbo
    }
}

public struct VerifyInstallResult: Sendable {
    public let receiptPath: String
    public let fileCount: Int
    public let bytesVerified: UInt64
    public let unexpectedEntries: [String]
}

public enum VerifiedInstallTool {
    // 64 MiB: sized for Qwen 3.6's ~22 MB layout.json (40 layers x 256 experts).
    public static let metadataMaxBytes: UInt64 = 64 * 1024 * 1024

    public static func run(options: VerifyInstallOptions) throws -> VerifyInstallResult {
        let access = try GTurboDirectoryAccess(rootPath: options.inputGTurbo)
        let manifestFD = try access.openFile("manifest.json")
        let manifestData: Data
        do {
            manifestData = try access.readMetadata(
                fileDescriptor: manifestFD,
                relativePath: "manifest.json",
                maxBytes: metadataMaxBytes)
            close(manifestFD)
        } catch {
            close(manifestFD)
            throw error
        }
        let manifestSize = UInt64(manifestData.count)
        let manifestSha = hash(manifestData)
        let manifest = try decodeManifest(manifestData)
        try validatePackedExpertLayout(access: access, manifest: manifest)

        var files: [RepackAudit.OutputFile] = []
        files.reserveCapacity(manifest.files.count)
        var bytesVerified = manifestSize
        for relativePath in manifest.files.keys.sorted() {
            guard let entry = manifest.files[relativePath] else { continue }
            let fd = try access.openFile(relativePath)
            let actualSize: UInt64
            let actualSha: String
            do {
                actualSize = try access.fileSize(fileDescriptor: fd, relativePath: relativePath)
                guard actualSize == entry.size else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) size \(actualSize) != manifest \(entry.size)")
                }
                actualSha = try access.hash(
                    fileDescriptor: fd,
                    relativePath: relativePath,
                    noCache: true)
                close(fd)
            } catch {
                close(fd)
                throw error
            }
            guard actualSha.lowercased() == entry.sha256.lowercased() else {
                throw RepackError.configurationInvalid(detail: "\(relativePath) SHA mismatch")
            }
            let sum = bytesVerified.addingReportingOverflow(actualSize)
            guard !sum.overflow else {
                throw RepackError.configurationInvalid(detail: "verified byte total overflowed")
            }
            bytesVerified = sum.partialValue
            files.append(RepackAudit.OutputFile(relativePath: relativePath,
                                                size: actualSize,
                                                sha256: actualSha))
        }
        let unexpectedEntries = try findUnexpectedEntries(access: access, manifest: manifest)

        let receiptData = try VerifiedInstallReceiptWriter.encode(
            outputDir: access.rootPath,
            manifestSha256: manifestSha,
            manifestSize: manifestSize,
            sourceRepoID: nil,
            sourceRevision: manifest.sourceSnapshotHash,
            toolVersion: "TurboFieldfareRepack verify-install",
            files: files)
        try access.atomicWrite(receiptData, to: VerifiedInstallReceiptWriter.fileName)
        let receiptPath = URL(fileURLWithPath: access.rootPath)
            .appendingPathComponent(VerifiedInstallReceiptWriter.fileName).path
        return VerifyInstallResult(receiptPath: receiptPath,
                                   fileCount: files.count + 1,
                                   bytesVerified: bytesVerified,
                                   unexpectedEntries: unexpectedEntries)
    }

    static func validatePackedExpertLayout(inputGTurbo: String) throws {
        let access = try GTurboDirectoryAccess(rootPath: inputGTurbo)
        let manifestData = try access.readMetadata("manifest.json", maxBytes: metadataMaxBytes)
        let manifest = try decodeManifest(manifestData)
        try validatePackedExpertLayout(access: access, manifest: manifest)
    }

    private struct ManifestFileEntry: Decodable {
        let size: UInt64
        let sha256: String
    }

    private struct Manifest: Decodable {
        let files: [String: ManifestFileEntry]
        let expertsPerLayer: Int
        let numLayers: Int
        let expertStride: UInt64
        let sourceSnapshotHash: String?
    }

    private struct PackedExpertsLayout: Decodable {
        let expertStride: UInt64
        let numLayers: Int
        let expertsPerLayer: Int
        let layers: [Layer]
    }

    private struct Layer: Decodable {
        let layer: Int
        let file: String
        let experts: [Expert]
    }

    private struct Expert: Decodable {
        let expert: Int?
        let offset: UInt64
        let size: UInt64
    }

    private static func decodeManifest(_ data: Data) throws -> Manifest {
        do {
            return try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw RepackError.configurationInvalid(detail: "manifest.json invalid: \(error)")
        }
    }

    private static func loadLayout(access: GTurboDirectoryAccess) throws -> PackedExpertsLayout {
        do {
            let data = try access.readMetadata(
                "packed_experts/layout.json",
                maxBytes: metadataMaxBytes)
            return try JSONDecoder().decode(PackedExpertsLayout.self, from: data)
        } catch let error as RepackError {
            throw error
        } catch {
            throw RepackError.configurationInvalid(detail: "packed_experts/layout.json invalid: \(error)")
        }
    }

    private static func hash(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }

    private static func validatePackedExpertLayout(
        access: GTurboDirectoryAccess,
        manifest: Manifest
    ) throws {
        let layoutRelativePath = "packed_experts/layout.json"
        guard manifest.files[layoutRelativePath] != nil else {
            throw RepackError.configurationInvalid(detail: "manifest missing \(layoutRelativePath)")
        }
        let layout = try loadLayout(access: access)
        let pageSize = UInt64(getpagesize())
        guard layout.numLayers > 0,
              layout.expertsPerLayer > 0,
              layout.expertStride > 0 else {
            throw RepackError.configurationInvalid(
                detail: "packed expert layout dimensions must be positive")
        }
        guard layout.expertStride == manifest.expertStride,
              layout.numLayers == manifest.numLayers,
              layout.expertsPerLayer == manifest.expertsPerLayer else {
            throw RepackError.configurationInvalid(detail: "packed expert layout dimensions mismatch manifest")
        }
        guard layout.expertStride % pageSize == 0 else {
            throw RepackError.configurationInvalid(
                detail: "expertStride \(layout.expertStride) is not page-aligned")
        }
        guard layout.layers.count == layout.numLayers else {
            throw RepackError.configurationInvalid(detail: "packed expert layout layer count mismatch")
        }
        let expectedLayerSizeProduct = UInt64(layout.expertsPerLayer)
            .multipliedReportingOverflow(by: layout.expertStride)
        guard !expectedLayerSizeProduct.overflow else {
            throw RepackError.configurationInvalid(detail: "packed expert layer size overflowed")
        }
        let expectedLayerSize = expectedLayerSizeProduct.partialValue
        var seenLayers = Set<Int>()
        for layer in layout.layers {
            guard layer.layer >= 0 && layer.layer < layout.numLayers else {
                throw RepackError.configurationInvalid(detail: "packed expert layer index out of range")
            }
            guard seenLayers.insert(layer.layer).inserted else {
                throw RepackError.configurationInvalid(
                    detail: "duplicate packed expert layer \(layer.layer)")
            }
            do {
                try GTurboPathValidator.validateBasename(layer.file, field: "packed_experts.file")
            } catch {
                throw RepackError.configurationInvalid(
                    detail: "unsafe packed expert layer filename \(layer.file): \(error)")
            }
            guard layer.experts.count == layout.expertsPerLayer else {
                throw RepackError.configurationInvalid(
                    detail: "packed_experts/\(layer.file) expert count mismatch")
            }
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw RepackError.configurationInvalid(detail: "manifest missing \(relativePath)")
            }
            guard manifestEntry.size == expectedLayerSize else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) manifest size \(manifestEntry.size) != \(expectedLayerSize)")
            }
            let actualSize = try access.fileSize(relativePath)
            guard actualSize == expectedLayerSize else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(expectedLayerSize)")
            }
            var seenExperts = Set<Int>()
            for (index, expert) in layer.experts.enumerated() {
                let expertID = expert.expert ?? index
                guard expertID >= 0 && expertID < layout.expertsPerLayer else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert id out of range")
                }
                guard seenExperts.insert(expertID).inserted else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) duplicate expert \(expertID)")
                }
                guard expert.size == layout.expertStride else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) size mismatch")
                }
                guard expert.offset % pageSize == 0 else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) offset is not page-aligned")
                }
                guard expert.offset <= actualSize,
                      expert.size <= actualSize - expert.offset else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) range exceeds file size")
                }
            }
        }
    }

    private static func findUnexpectedEntries(
        access: GTurboDirectoryAccess,
        manifest: Manifest
    ) throws -> [String] {
        let declaredFiles = Set(manifest.files.keys)
            .union(["manifest.json", VerifiedInstallReceiptWriter.fileName])
        var allowed = declaredFiles
        for path in declaredFiles {
            var parts = path.split(separator: "/").map(String.init)
            while parts.count > 1 {
                _ = parts.removeLast()
                allowed.insert(parts.joined(separator: "/"))
            }
        }
        allowed.insert("tokenizer")

        let entries = try access.relativeEntries()
        var unexpected: [String] = []
        for rel in entries {
            if rel == ".DS_Store" { continue }
            if rel == "tokenizer" || rel.hasPrefix("tokenizer/") { continue }
            if !allowed.contains(rel) {
                unexpected.append(rel)
            }
        }
        return unexpected.sorted()
    }
}
