import Foundation
import TurboFieldfare

enum StreamedMoEArchitecture: String, Hashable, Sendable {
    case gemma4
    case qwen36

    var expectedArchitecture: ArchConfig {
        switch self {
        case .gemma4: .gemma4_26B_A4B
        case .qwen36: .qwen36_35B_A3B
        }
    }
}

struct StreamedMoEVariant: Identifiable, Hashable, Sendable {
    let modelID: String
    let sourceRepoID: String
    let sourceRevision: String
    let sourceIndexSHA256: String
    let displayName: String
    let quantLabel: String
    let helperModelArgument: String
    let managedDirectoryName: String
    let approximateDownloadBytes: UInt64
    let architecture: StreamedMoEArchitecture

    var id: String { modelID }

    var managedDirectory: URL {
        ModelStore.modelsRoot.appendingPathComponent(
            managedDirectoryName,
            isDirectory: true)
    }

    var managedInstallPaths: StreamedModelInstallPaths {
        StreamedModelInstallPaths(finalDirectory: managedDirectory)
    }

    /// Every app-owned artifact created by the resumable installer. Sidecars
    /// come first so the registered payload stays intact if moving one fails.
    var managedInstallArtifactURLs: [URL] {
        managedInstallPaths.removableArtifacts
    }
}

/// The final payload and resumable sidecars for one app-managed install.
/// Installer and removal code share this type so their path rules cannot drift.
struct StreamedModelInstallPaths: Equatable, Sendable {
    let finalDirectory: URL

    var checkpointFile: URL {
        URL(fileURLWithPath: finalDirectory.path + ".resume.json")
    }

    var partialDirectory: URL {
        URL(fileURLWithPath: finalDirectory.path + ".partial", isDirectory: true)
    }

    var removableArtifacts: [URL] {
        [checkpointFile, partialDirectory, finalDirectory]
    }
}

/// Memory and SSD-I/O balance for the TurboFieldfare backend. These settings
/// apply only to streamed MoE models; standard MLX models never read them.
struct StreamedMoERuntimeSettings: Equatable, Sendable {
    static let allowedContextTokens = [4_096, 8_192, 16_384, 32_768]
    static let allowedExpertCacheSlots = RuntimeConfiguration.allowedExpertCacheSlots

    let contextTokens: Int
    let expertCacheSlots: Int

    init(contextTokens: Int, expertCacheSlots: Int) {
        self.contextTokens = Self.allowedContextTokens.contains(contextTokens)
            ? contextTokens : Keys.Defaults.streamedModelContextTokens
        self.expertCacheSlots = Self.allowedExpertCacheSlots.contains(expertCacheSlots)
            ? expertCacheSlots : Keys.Defaults.streamedModelExpertCacheSlots
    }

    static func current(defaults: UserDefaults = .standard) -> Self {
        let contextTokens = defaults.object(forKey: Keys.streamedModelContextTokens) as? Int
            ?? Keys.Defaults.streamedModelContextTokens
        let expertCacheSlots = defaults.object(
            forKey: Keys.streamedModelExpertCacheSlots) as? Int
            ?? Keys.Defaults.streamedModelExpertCacheSlots
        return Self(
            contextTokens: contextTokens,
            expertCacheSlots: expertCacheSlots)
    }
}

/// Model-specific pins for each checkpoint supported by the SSD-streamed
/// backend. Each ordinary MLX model keeps a separate app-local streamed
/// identity, install directory, receipt, and resumable checkpoint.
enum StreamedMoEInstall {
    static let standard = StreamedMoEVariant(
        modelID: "streamed-moe/gemma-4-26b-a4b-it-4bit",
        sourceRepoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        sourceRevision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256: "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        displayName: "Gemma 4 26B A4B",
        quantLabel: "4-bit · SSD",
        helperModelArgument: "gemma4",
        managedDirectoryName: "Gemma4-26B-A4B.gturbo",
        approximateDownloadBytes: 14_620_479_420,
        architecture: .gemma4)

    static let qat = StreamedMoEVariant(
        modelID: "streamed-moe/gemma-4-26b-a4b-it-qat-4bit",
        sourceRepoID: "mlx-community/gemma-4-26B-A4B-it-qat-4bit",
        sourceRevision: "0e3cbab38ce568cf6e23543010d08d03b731910c",
        sourceIndexSHA256: "5455e83705bbdd4e3702c7d4f9d49d4900e84533036628f74500538075dd5c80",
        displayName: "Gemma 4 26B A4B QAT",
        quantLabel: "QAT 4-bit · SSD",
        helperModelArgument: "gemma4-qat",
        managedDirectoryName: "Gemma4-26B-A4B-QAT.gturbo",
        approximateDownloadBytes: 14_952_958_284,
        architecture: .gemma4)

    static let qwen36 = StreamedMoEVariant(
        modelID: "streamed-moe/qwen3.6-35b-a3b-4bit",
        sourceRepoID: "mlx-community/Qwen3.6-35B-A3B-4bit",
        sourceRevision: "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        sourceIndexSHA256: "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea",
        displayName: "Qwen3.6 35B A3B",
        quantLabel: "4-bit · SSD",
        helperModelArgument: "qwen36",
        managedDirectoryName: "Qwen3.6-35B-A3B.gturbo",
        approximateDownloadBytes: 19_529_025_048,
        architecture: .qwen36)

    static let ornith15 = StreamedMoEVariant(
        modelID: "streamed-moe/ornith-1.5-35b-a3b-4bit",
        sourceRepoID: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
        sourceRevision: "19504d912fa8fc7622bf6b1de3db5d5d890b1f02",
        sourceIndexSHA256: "c118f13c0dcb729e4ca2e3d653ab193067551eb1a6410badb5192eb426104f36",
        displayName: "Ornith 1.5 35B A3B",
        quantLabel: "4-bit · SSD",
        helperModelArgument: "ornith15",
        managedDirectoryName: "Ornith-1.5-35B-A3B.gturbo",
        approximateDownloadBytes: 19_528_808_680,
        architecture: .qwen36)

    static let variants = [standard, qat, qwen36, ornith15]

    /// Bookmark key used by the first streamed-MoE build, before the streamed
    /// install received its own app-local identity.
    static let legacyModelID = standard.sourceRepoID

    static var runtimeContextTokens: Int {
        StreamedMoERuntimeSettings.current().contextTokens
    }

    static func supportsSSDStreaming(sourceModelID: String) -> Bool {
        variant(forSourceModelID: sourceModelID) != nil
    }

    static func variant(forSourceModelID id: String) -> StreamedMoEVariant? {
        variants.first { $0.sourceRepoID.caseInsensitiveCompare(id) == .orderedSame }
    }

    static func variant(forStreamedModelID id: String) -> StreamedMoEVariant? {
        variants.first { $0.modelID == id }
    }

    static let requiredTokenizerPaths = [
        "tokenizer/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ]

    private static func missingRelativePaths(
        _ relativePaths: [String],
        at directory: URL,
        fileManager: FileManager = .default
    ) -> [String] {
        relativePaths.filter {
            !fileManager.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    static func validationError(
        at directory: URL,
        for variant: StreamedMoEVariant,
        fileManager: FileManager = .default
    ) -> String? {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return "Choose a completed .gturbo model directory."
        }
        do {
            let manifest = try ManifestReader.load(
                directoryURL: directory,
                expecting: variant.architecture.expectedArchitecture)
            let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
            guard let sourceRepoID = receipt.sourceRepoID,
                  sourceRepoID.caseInsensitiveCompare(variant.sourceRepoID) == .orderedSame,
                  receipt.sourceRevision == variant.sourceRevision else {
                return "This .gturbo directory was not built from MLX Chat’s pinned \(variant.displayName) checkpoint."
            }
            let manifestSHA = try Sha256Verifier.hashFile(
                at: directory.appendingPathComponent("manifest.json"))
            try VerifiedInstallReceiptReader.validateManifestBinding(
                receipt,
                directoryURL: directory,
                manifestSha256: manifestSHA)
            let requiredPaths = manifest.files.keys.sorted() + requiredTokenizerPaths
            let missing = missingRelativePaths(
                requiredPaths,
                at: directory,
                fileManager: fileManager)
            guard missing.isEmpty else {
                let preview = missing.prefix(3).joined(separator: ", ")
                let remainder = missing.count > 3 ? " and \(missing.count - 3) more" : ""
                return "This .gturbo install is incomplete. Missing \(preview)\(remainder)."
            }
        } catch {
            return "This .gturbo install is invalid: \(String(describing: error))"
        }
        return nil
    }
}
