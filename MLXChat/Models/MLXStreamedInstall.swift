import Foundation

struct MLXStreamedVariant: Identifiable, Hashable, Sendable {
    let modelID: String
    let sourceModelID: String
    let displayName: String
    let quantLabel: String
    let approximateDownloadBytes: UInt64
    let expertCount: Int
    let topK: Int
    let coreStart: Int
    let coreEnd: Int
    let expertBytesPerLayer: [UInt64]

    var id: String { modelID }

    /// Exact slot-cache memory footprint once all assigned slots across every layer are filled.
    func expertCacheBytes(
        forSlots slots: Int,
        loopWeighted: Bool = Inkling2StreamingConfig.defaultLoopWeightedSlots
    ) -> UInt64 {
        var total: UInt64 = 0
        for (layer, perExpert) in expertBytesPerLayer.enumerated() {
            let isCore = (coreStart ..< coreEnd).contains(layer)
            let raw = (loopWeighted && !isCore) ? (slots / 2) : slots
            let assigned = max(topK, min(raw, expertCount))
            total += UInt64(assigned) * perExpert
        }
        return total
    }
}

/// Curated MLX checkpoints that also support direct on-demand SSD expert streaming
/// from the same downloaded safetensors files (no separate `.gturbo` repack).
enum MLXStreamedInstall {
    private static let inkling2MiniNVFP4ExpertBytes: UInt64 = 4_478_976
    private static let inkling2MiniMXFP8ExpertBytes: UInt64 = 8_211_456

    static let inkling2Mini4Bit = MLXStreamedVariant(
        modelID: "mlx-streamed/inkling-2-mini-4bit",
        sourceModelID: "mlx-community/Inkling-2-Mini-4bit",
        displayName: "Inkling 2 Mini 4-bit",
        quantLabel: "4-bit · SSD",
        approximateDownloadBytes: 17_310_000_000,
        expertCount: 128,
        topK: 8,
        coreStart: 6,
        coreEnd: 18,
        expertBytesPerLayer: Array(repeating: inkling2MiniNVFP4ExpertBytes, count: 20)
            + Array(repeating: inkling2MiniMXFP8ExpertBytes, count: 4))

    static let variants = [inkling2Mini4Bit]

    static func supportsSSDStreaming(sourceModelID: String) -> Bool {
        variant(forSourceModelID: sourceModelID) != nil
    }

    static func variant(forSourceModelID id: String) -> MLXStreamedVariant? {
        variants.first { $0.sourceModelID.caseInsensitiveCompare(id) == .orderedSame }
    }

    static func variant(forStreamedModelID id: String) -> MLXStreamedVariant? {
        variants.first { $0.modelID == id }
    }
}
