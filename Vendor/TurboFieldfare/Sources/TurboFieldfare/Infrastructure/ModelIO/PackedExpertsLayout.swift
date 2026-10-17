import Foundation
import TurboFieldfareFormat

struct SubTensorEntry: Sendable, Equatable {
    let offset: UInt64    // relative to the expert blob's start
    let size: UInt64
    let dtype: String
    let shape: [UInt32]
    let bits: Int?

    init(offset: UInt64,
         size: UInt64 = 0,
         dtype: String = "",
         shape: [UInt32] = [],
         bits: Int? = nil) {
        self.offset = offset
        self.size = size
        self.dtype = dtype
        self.shape = shape
        self.bits = bits
    }
}

struct ExpertEntry: Sendable {
    /// Logical routed-expert id used by the model/router.
    let expert: Int
    /// Physical rank inside the layer file. Old identity layouts use
    /// `physicalRank == expert`.
    let physicalRank: Int
    /// Absolute byte offset of this expert blob's start inside its layer file.
    let offset: UInt64
    /// Total bytes consumed by this expert blob (== `expertStride`).
    let size: UInt64
    /// Sub-tensors keyed by role (gate / up / down / shared) and component
    /// (raw, `_scales`, `_biases`).
    let subTensors: [String: SubTensorEntry]

    init(expert: Int,
         physicalRank: Int? = nil,
         offset: UInt64,
         size: UInt64,
         subTensors: [String: SubTensorEntry]) {
        self.expert = expert
        self.physicalRank = physicalRank ?? expert
        self.offset = offset
        self.size = size
        self.subTensors = subTensors
    }
}

struct LayerLayout: Sendable {
    let layer: Int
    let file: String          // basename, e.g. "layer_00.bin"
    let experts: [ExpertEntry]
}

struct PackedExpertsLayout: Sendable {
    let expertStride: UInt64
    let numLayers: Int
    let expertsPerLayer: Int
    let layers: [LayerLayout]

    /// Resolve `(layer, expert)` -> `ExpertEntry`. O(1).
    func expert(layer: Int, expert: Int) -> ExpertEntry {
        return layers[layer].experts[expert]
    }
}

enum PackedExpertsLayoutReader {
    // 64 MiB: Qwen 3.6's 40 layers x 256 experts x 9 sub-tensors produce a
    // ~22 MB layout.json; Gemma's is ~5 MB. Upstream #87 used 16 MiB, which
    // cannot hold the Qwen layout.
    static let defaultMaxBytes: UInt64 = 64 * 1024 * 1024

    static func load(directoryURL: URL,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> PackedExpertsLayout {
        try load(directoryURL: directoryURL, manifest: nil, maxBytes: maxBytes)
    }

    package static func load(directoryURL: URL,
                             manifest: Manifest,
                             maxBytes: UInt64 = defaultMaxBytes) throws -> PackedExpertsLayout {
        try load(directoryURL: directoryURL, manifest: Optional(manifest), maxBytes: maxBytes)
    }

    private static func load(directoryURL: URL,
                             manifest: Manifest?,
                             maxBytes: UInt64) throws -> PackedExpertsLayout {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data = try directory.readMetadata(
            "packed_experts/layout.json", maxBytes: maxBytes)
        return try decode(data: data, manifest: manifest)
    }

    package static func decode(data: Data,
                               manifest: Manifest?) throws -> PackedExpertsLayout {
        do {
            return try decodeV1(data: data, manifest: manifest)
        } catch {
            return try decodeLegacy(data: data)
        }
    }

    private static func decodeV1(data: Data, manifest: Manifest?) throws -> PackedExpertsLayout {
        let wire = try GTurboPackedExpertsLayoutCodec.decode(data)
        if let manifest {
            try GTurboV1StructuralValidator.crossValidate(
                manifestNumLayers: manifest.numLayers,
                manifestExpertsPerLayer: manifest.expertsPerLayer,
                manifestExpertStride: manifest.expertStride,
                manifestFileSizes: manifest.files.mapValues(\.size),
                layout: wire)
        }
        var slots = Array(repeating: Optional<LayerLayout>.none, count: wire.numLayers)
        for layer in wire.layers {
            var experts = [ExpertEntry?](repeating: nil, count: wire.expertsPerLayer)
            for (position, expert) in layer.experts.enumerated() {
                let logical = expert.expert ?? position
                experts[logical] = ExpertEntry(
                    expert: logical,
                    physicalRank: expert.physicalRank,
                    offset: expert.offset,
                    size: expert.size,
                    subTensors: expert.tensors.mapValues {
                        SubTensorEntry(offset: $0.offset, size: $0.size,
                                       dtype: $0.dtype, shape: $0.shape,
                                       bits: $0.bits)
                    })
            }
            guard experts.allSatisfy({ $0 != nil }) else {
                throw ModelError.indexCorrupt(detail: "layout.json: missing expert entries")
            }
            slots[layer.layer] = LayerLayout(
                layer: layer.layer,
                file: layer.file,
                experts: experts.map { $0! })
        }
        guard slots.allSatisfy({ $0 != nil }) else {
            throw ModelError.indexCorrupt(detail: "layout.json: missing layer entries")
        }
        return PackedExpertsLayout(expertStride: wire.expertStride,
                                   numLayers: wire.numLayers,
                                   expertsPerLayer: wire.expertsPerLayer,
                                   layers: slots.map { $0! })
    }

    private static func decodeLegacy(data: Data) throws -> PackedExpertsLayout {
        let root: [String: Any]
        do {
            root = try (JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        } catch {
            throw ModelError.indexCorrupt(detail: "layout.json: \(error)")
        }
        guard
            let expertStride = (root["expertStride"] as? NSNumber)?.uint64Value,
            let numLayers = root["numLayers"] as? Int,
            let expertsPerLayer = root["expertsPerLayer"] as? Int,
            let layersArr = root["layers"] as? [[String: Any]]
        else {
            throw ModelError.indexCorrupt(detail: "layout.json: missing top-level keys")
        }

        var layers: [LayerLayout] = []
        layers.reserveCapacity(layersArr.count)
        for layerObj in layersArr {
            guard
                let layerIdx = layerObj["layer"] as? Int,
                let file = layerObj["file"] as? String,
                let expertsArr = layerObj["experts"] as? [[String: Any]]
            else {
                throw ModelError.indexCorrupt(detail: "layout.json: malformed layer entry")
            }
            var experts = [ExpertEntry?](repeating: nil, count: expertsPerLayer)
            for expertObj in expertsArr {
                guard
                    let offset = (expertObj["offset"] as? NSNumber)?.uint64Value,
                    let size = (expertObj["size"] as? NSNumber)?.uint64Value,
                    let tensorsObj = expertObj["tensors"] as? [String: [String: Any]]
                else {
                    throw ModelError.indexCorrupt(detail: "layout.json: malformed expert entry")
                }
                var subTensors: [String: SubTensorEntry] = [:]
                for (role, t) in tensorsObj {
                    guard
                        let toff = (t["offset"] as? NSNumber)?.uint64Value,
                        t["size"] is NSNumber,
                        t["dtype"] is String,
                        t["shape"] is [Int]
                    else {
                        throw ModelError.indexCorrupt(detail: "layout.json: malformed tensor \(role)")
                    }
                    if let bits = t["bits"], !(bits is Int) {
                        throw ModelError.indexCorrupt(detail: "layout.json: malformed tensor bits \(role)")
                    }
                    let tsize = (t["size"] as? NSNumber)?.uint64Value ?? 0
                    let dtype = t["dtype"] as? String ?? ""
                    let shape = (t["shape"] as? [Int])?.map { UInt32(clamping: $0) } ?? []
                    let bits = t["bits"] as? Int
                    subTensors[role] = SubTensorEntry(offset: toff, size: tsize,
                                                      dtype: dtype, shape: shape, bits: bits)
                }
                let expertID = expertObj["expert"] as? Int ?? experts.compactMap { $0 }.count
                guard expertID >= 0 && expertID < expertsPerLayer else {
                    throw ModelError.indexCorrupt(detail: "layout.json: expert id out of range")
                }
                let physicalRank = expertObj["physicalRank"] as? Int
                if let physicalRank,
                   (physicalRank < 0 || physicalRank >= expertsPerLayer) {
                    throw ModelError.indexCorrupt(detail: "layout.json: physicalRank out of range")
                }
                experts[expertID] = ExpertEntry(expert: expertID,
                                                physicalRank: physicalRank,
                                                offset: offset,
                                                size: size,
                                                subTensors: subTensors)
            }
            guard experts.allSatisfy({ $0 != nil }) else {
                throw ModelError.indexCorrupt(detail: "layout.json: missing expert entries")
            }
            layers.append(LayerLayout(layer: layerIdx,
                                      file: file,
                                      experts: experts.map { $0! }))
        }

        return PackedExpertsLayout(expertStride: expertStride,
                                   numLayers: numLayers,
                                   expertsPerLayer: expertsPerLayer,
                                   layers: layers)
    }
}
