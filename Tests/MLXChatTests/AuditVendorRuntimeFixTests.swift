import Foundation
import Testing
@testable import TurboFieldfare

@Suite("Audit vendor runtime fixes (AUDIT-33 to AUDIT-38)")
struct AuditVendorRuntimeFixTests {
    // AUDIT-33 (E-B1): FlintForwardRunner continuation rejects double rewinds past highWaterPosition.
    @Test("AUDIT-33 FlintForwardRunner tracks highWaterPosition across repeated rewinds")
    func audit33FlintHighWaterRewindCheck() {
        // Max rewind distance 4096 (ring 4609, window 513):
        // Reach 6000 -> rewind to 3000 (allowed: 6000 - 3000 = 3000 <= 4096).
        #expect(FlintForwardRunner.canContinue(
            position: 6000,
            highWaterPosition: 6000,
            expectedPosition: 3000,
            maxRewindDistance: 4096))

        // After generating to 3050, rewinding to 500 would pass (3050 - 500 <= 4096)
        // if highWaterPosition were ignored, but 6000 - 500 = 5500 > 4096 must be rejected.
        #expect(!FlintForwardRunner.canContinue(
            position: 3050,
            highWaterPosition: 6000,
            expectedPosition: 500,
            maxRewindDistance: 4096))

        // After reset (highWaterPosition == 0) or within high-water window, continuation succeeds.
        #expect(FlintForwardRunner.canContinue(
            position: 3050,
            highWaterPosition: 6000,
            expectedPosition: 1904,
            maxRewindDistance: 4096))
        #expect(!FlintForwardRunner.canContinue(
            position: 3050,
            highWaterPosition: 6000,
            expectedPosition: -1,
            maxRewindDistance: 4096))
        #expect(!FlintForwardRunner.canContinue(
            position: 3050,
            highWaterPosition: 6000,
            expectedPosition: 3051,
            maxRewindDistance: 4096))
    }

    // AUDIT-34 (E-B2): ResidentIndexReader.parseLegacyEntries rejects truncated/overflowing entry tables without reading past indexBuf.
    @Test("AUDIT-34 ResidentIndexReader.parseLegacyEntries rejects truncated entry tables safely")
    func audit34ResidentIndexLegacyBoundsCheck() {
        let header = ResidentIndexHeader(indexSize: 64, residentSize: 0, entryCount: 100)
        let truncated = [UInt8](repeating: 0, count: 64)
        #expect(throws: ModelError.self) {
            _ = try ResidentIndexReader.parseLegacyEntries(
                indexBuf: truncated,
                header: header,
                regionLen: truncated.count)
        }

        let hugeHeader = ResidentIndexHeader(indexSize: 64, residentSize: 0, entryCount: UInt64.max)
        #expect(throws: ModelError.self) {
            _ = try ResidentIndexReader.parseLegacyEntries(
                indexBuf: truncated,
                header: hugeHeader,
                regionLen: truncated.count)
        }

        // regionLen > indexBuf.count must also throw instead of reading out of bounds.
        let oneEntryHeader = ResidentIndexHeader(indexSize: 200, residentSize: 0, entryCount: 1)
        #expect(throws: ModelError.self) {
            _ = try ResidentIndexReader.parseLegacyEntries(
                indexBuf: truncated,
                header: oneEntryHeader,
                regionLen: 200)
        }
    }

    // AUDIT-35 (E-B2): ManifestReader throws ModelError.archMismatch instead of trapping on out-of-range fullAttentionLayerMask.
    @Test("AUDIT-35 ManifestReader rejects out-of-range fullAttentionLayerMask without trapping")
    func audit35ManifestReaderOutOfRangeLayerMaskThrows() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("audit35-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let expected = ArchConfig.gemma4_26B_A4B
        var badMask = expected.fullAttentionLayerMask.map { Int($0) }
        badMask[0] = -1
        badMask[1] = 300

        let manifestObj: [String: Any] = [
            "magic": "GTURBO1",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": [:],
            "modelID": "test-bad-mask",
            "arch": [
                "hiddenSize": expected.hiddenSize,
                "ffnIntermediate": expected.intermediateSize,
                "moeIntermediateSize": expected.moeIntermediateSize,
                "numHeads": expected.numHeads,
                "numKVHeads": expected.numKVHeads,
                "numFullKVHeads": expected.numFullKVHeads,
                "headDim": expected.headDim,
                "fullHeadDim": expected.fullHeadDim,
                "vocabSize": expected.vocabSize,
                "slidingWindow": expected.slidingWindow,
                "finalLogitSoftcap": expected.finalLogitSoftcap,
                "ropeTheta": expected.ropeTheta,
                "fullRopeTheta": expected.fullRopeTheta,
                "partialRotaryFactor": expected.partialRotaryFactor,
                "numLayers": expected.numLayers,
                "numExperts": expected.numExperts,
                "topKExperts": expected.topKExperts,
                "tieWordEmbeddings": expected.tieWordEmbeddings,
                "attentionKEqV": expected.attentionKEqV,
                "fullAttentionLayerMask": badMask,
                "hiddenActivation": expected.hiddenActivation,
            ],
            "files": [
                "model_weights.bin": ["size": 16_384, "sha256": String(repeating: "a", count: 64)],
                "packed_experts/layout.json": ["size": 128, "sha256": String(repeating: "b", count: 64)],
            ],
            "expertsPerLayer": expected.numExperts,
            "numLayers": expected.numLayers,
            "expertStride": 16_384,
        ]
        let data = try JSONSerialization.data(withJSONObject: manifestObj)
        try data.write(to: dir.appendingPathComponent("manifest.json"))

        #expect(throws: ModelError.self) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: expected)
        }
    }

    // AUDIT-36 (E-B6): GenerationConfig.validate() rejects non-finite and non-positive repetitionPenalty.
    @Test("AUDIT-36 GenerationConfig.validate rejects invalid repetitionPenalty values")
    func audit36GenerationConfigValidatesRepetitionPenalty() throws {
        for bad: Float in [0, -1, -.infinity, .infinity, .nan] {
            let cfg = GenerationConfig(maxNewTokens: 8, temperature: 0, repetitionPenalty: bad)
            #expect(throws: GeneratorError.self) {
                try cfg.validate()
            }
        }
        let ok = GenerationConfig(maxNewTokens: 8, temperature: 0, repetitionPenalty: 1.0)
        try ok.validate()
    }

    // AUDIT-37 (E-B7): PartialByteBuffer drains tens of thousands of invalid UTF-8 bytes iteratively without stack overflow.
    @Test("AUDIT-37 PartialByteBuffer handles long invalid UTF-8 runs iteratively")
    func audit37PartialByteBufferIterativeDrain() {
        var buffer = PartialByteBuffer()
        var payload = [UInt8](repeating: 0xFF, count: 50_000)
        payload.append(contentsOf: Array("ok".utf8))
        let out = buffer.pushBytes(payload) + buffer.flush()
        #expect(out == "ok")
    }

    // AUDIT-38 (E-B8): ImageMetadataReader.decodedByteEstimate rounds 12-bit channels up to 2 bytes and detects overflow safely.
    @Test("AUDIT-38 ImageMetadataReader decodedByteEstimate handles 12-bit channels and overflow")
    func audit38ImageMetadataDecodedByteEstimate() {
        // 12-bit depth rounds up to 2 bytes/channel -> 8 bytes/pixel across 4 RGBA channels.
        #expect(ImageMetadataReader.decodedByteEstimate(pixels: 100, depth: 12) == 800)
        #expect(ImageMetadataReader.decodedByteEstimate(pixels: 100, depth: 8) == 400)
        #expect(ImageMetadataReader.decodedByteEstimate(pixels: 100, depth: 16) == 800)
        // Negative or huge overflowing depth returns nil instead of trapping.
        #expect(ImageMetadataReader.decodedByteEstimate(pixels: 100, depth: -8) == nil)
        #expect(ImageMetadataReader.decodedByteEstimate(pixels: 100, depth: Int.max) == nil)
        #expect(ImageMetadataReader.decodedByteEstimate(pixels: Int.max, depth: 8) == nil)
    }
}
