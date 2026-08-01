import Foundation

/// A pinned upstream checkpoint the installer knows how to repack. Each value
/// fixes the repo, revision and index fingerprint so installs are exactly
/// reproducible.
public struct SupportedModelSource: Sendable, Equatable {
    /// CLI selector value (`--model <name>`).
    public let name: String
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    /// Value recorded as `manifest.modelID` when the source fingerprint matches.
    public let modelID: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64

    public func installOptions(outputDirectory: URL,
                               overwrite: Bool,
                               token: String?,
                               resume: Bool = false,
                               dryRun: Bool = false)
        -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume,
            dryRunSpaceCheck: dryRun)
    }

    public static let gemma4 = SupportedModelSource(
        name: "gemma4",
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256:
            "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        modelID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        reserveBytes: 1_073_741_824)

    public static let gemma4QAT = SupportedModelSource(
        name: "gemma4-qat",
        displayName: "Gemma 4 26B-A4B IT QAT 4-bit",
        repoID: "mlx-community/gemma-4-26B-A4B-it-qat-4bit",
        revision: "0e3cbab38ce568cf6e23543010d08d03b731910c",
        sourceIndexSHA256:
            "5455e83705bbdd4e3702c7d4f9d49d4900e84533036628f74500538075dd5c80",
        modelID: "mlx-community/gemma-4-26B-A4B-it-qat-4bit",
        approximateDownloadBytes: 14_952_958_284,
        installedBytes: 14_559_575_182,
        reserveBytes: 1_073_741_824)

    /// Download estimate covers the `language_model.*` tensors plus tokenizer
    /// and metadata sidecars; the vision tower is never fetched. Installed
    /// bytes add the resident index and per-expert 16 KB page rounding
    /// (the 1,769,472-byte expert blob is already page-aligned) plus
    /// layout/manifest sidecars.
    public static let qwen36 = SupportedModelSource(
        name: "qwen36",
        displayName: "Qwen3.6 35B-A3B 4-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-4bit",
        revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        sourceIndexSHA256:
            "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea",
        modelID: "qwen3.6-35b-a3b-4bit",
        approximateDownloadBytes: 19_529_025_048,
        installedBytes: 19_546_491_213,
        reserveBytes: 1_073_741_824)

    /// Default source when no `--model` selector is given.
    public static let `default` = gemma4

    public static let all: [SupportedModelSource] = [gemma4, gemma4QAT, qwen36]

    public static func named(_ name: String) -> SupportedModelSource? {
        // Keep the selectors used by MLX Chat's first streamed-model release.
        if name == "standard" { return gemma4 }
        if name == "qat" { return gemma4QAT }
        return all.first { $0.name == name }
    }
}
