import Foundation
import XCTest
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

final class AuditRepackFixTests: XCTestCase {
    // AUDIT-39 (F-B17): DiskSpaceChecker probes dot-containing directory names instead of stripping them to parent mount.
    func testAudit39DiskSpaceCheckerKeepsDotContainingDirectoryName() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("audit39-\(UUID().uuidString)", isDirectory: true)
        let dottedDir = base.appendingPathComponent("External.SSD", isDirectory: true)
        try FileManager.default.createDirectory(at: dottedDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }

        let assessed = try DiskSpaceChecker.assess(path: dottedDir.path, bytes: 1, reserveBytes: 0)
        XCTAssertEqual(
            URL(fileURLWithPath: assessed.path).standardizedFileURL.path,
            dottedDir.standardizedFileURL.path)

        let childDotDir = base.appendingPathComponent("dev.mlxchat.models", isDirectory: true)
        let required = try DiskSpaceChecker.requireAvailable(path: childDotDir.path, bytes: 1, reserveBytes: 0)
        XCTAssertEqual(
            URL(fileURLWithPath: required.path).standardizedFileURL.path,
            childDotDir.standardizedFileURL.path)
        var isDir: ObjCBool = false
        XCTAssertTrue(FileManager.default.fileExists(atPath: childDotDir.path, isDirectory: &isDir) && isDir.boolValue)
    }

    // AUDIT-40 (F-B18): Safetensors.parseHeaderBytes rejects fileSize < 8, end < begin, negative/bool numbers, and overflow without trapping.
    func testAudit40SafetensorsParseHeaderRejectsUnderflowOverflowAndBadNumbers() throws {
        // fileSize < 8
        XCTAssertThrowsError(
            try Safetensors.parseHeaderBytes(path: "tiny.safetensors", fileSize: 4, headerBytes: Data("{}".utf8))
        )

        // end < begin
        let badOrder = Data(#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[16,8]}}"#.utf8)
        XCTAssertThrowsError(
            try Safetensors.parseHeaderBytes(path: "order.safetensors", fileSize: 1024, headerBytes: badOrder)
        )

        // negative shape entry
        let negShape = Data(#"{"w":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#.utf8)
        XCTAssertThrowsError(
            try Safetensors.parseHeaderBytes(path: "neg.safetensors", fileSize: 1024, headerBytes: negShape)
        )

        // boolean shape entry
        let boolShape = Data(#"{"w":{"dtype":"F32","shape":[true],"data_offsets":[0,4]}}"#.utf8)
        XCTAssertThrowsError(
            try Safetensors.parseHeaderBytes(path: "bool.safetensors", fileSize: 1024, headerBytes: boolShape)
        )

        // shape multiplication overflow
        let overflowShape = Data(
            #"{"w":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,4]}}"#.utf8
        )
        XCTAssertThrowsError(
            try Safetensors.parseHeaderBytes(path: "overflow.safetensors", fileSize: 1024, headerBytes: overflowShape)
        )

        // offset addition overflow
        let hugeOffset = Data(
            #"{"w":{"dtype":"F32","shape":[1],"data_offsets":[18446744073709551610,18446744073709551614]}}"#.utf8
        )
        XCTAssertThrowsError(
            try Safetensors.parseHeaderBytes(path: "huge.safetensors", fileSize: UInt64.max, headerBytes: hugeOffset)
        )
    }

    // AUDIT-41 (F-B24): RangeCopyPlanner clamps remoteGapBytesDownloaded to 0 instead of trapping when source ranges overlap.
    func testAudit41RangeCopyPlannerHandlesOverlappingSourceRangesWithoutUnderflow() throws {
        let shard = "shared.safetensors"
        let t1 = SourceTensor(
            name: "vision_tower.a.weight", shardPath: shard,
            dtype: .bf16, shape: [32], absoluteOffset: 0, sizeBytes: 64)
        let t2 = SourceTensor(
            name: "vision_tower.b.weight", shardPath: shard,
            dtype: .bf16, shape: [32], absoluteOffset: 0, sizeBytes: 64)
        let visionPlan = VisionPackPlan(
            entries: [
                .init(source: t1, executionPosition: 0, fileOffset: 0, quantSpec: nil, groupSize: 64),
                .init(source: t2, executionPosition: 1, fileOffset: 64, quantSpec: nil, groupSize: 64),
            ],
            weightsFileSize: 128,
            sourcePayloadBytes: 128)
        let rangePlan = try RangeCopyPlanner.plan(
            vision: visionPlan,
            weightsPath: "/tmp/audit41/vision_weights.bin",
            rangeChunkBytes: 64 * 1024)
        XCTAssertEqual(rangePlan.remoteBytesToDownload, 64)
        XCTAssertEqual(rangePlan.remoteGapBytesDownloaded, 0)
    }

    // AUDIT-42 (F-B29): ArchInfo.load rejects LFM2 MoE configs with zero or non-dividing num_attention_heads without trapping.
    func testAudit42ArchInfoLFM2MoERejectsZeroHeads() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("audit42-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let configURL = dir.appendingPathComponent("config.json")
        let badJSON = """
        {
          "model_type": "lfm2_moe",
          "hidden_size": 2048,
          "intermediate_size": 4096,
          "moe_intermediate_size": 1024,
          "num_attention_heads": 0,
          "num_key_value_heads": 8,
          "vocab_size": 65536,
          "num_hidden_layers": 4,
          "num_experts": 8,
          "num_experts_per_tok": 2,
          "num_dense_layers": 2,
          "conv_L_cache": 3,
          "layer_types": ["conv", "conv", "full_attention", "conv"],
          "rope_theta": 1000000.0
        }
        """
        try Data(badJSON.utf8).write(to: configURL)

        XCTAssertThrowsError(try ArchInfo.load(configPath: configURL.path))
    }

    // AUDIT-43 (F-B22): GTurboLayoutValidator rejects empty layers arrays and duplicate layer indices.
    func testAudit43GTurboLayoutValidatorRejectsEmptyAndDuplicateLayers() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("audit43-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let arch = ArchInfo(
            hiddenSize: 64,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            numHeads: 2,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 32,
            fullHeadDim: 32,
            vocabSize: 256,
            slidingWindow: 0,
            finalLogitSoftcap: 0,
            ropeTheta: 10_000,
            fullRopeTheta: 10_000,
            partialRotaryFactor: 1,
            numLayers: 2,
            numExperts: 2,
            topKExperts: 1,
            tieWordEmbeddings: true,
            attentionKEqV: false,
            fullAttentionLayerMask: [1, 1],
            hiddenActivation: "silu",
            family: .gemma4,
            attnOutputGate: false,
            attentionScale: 1.0,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: false,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0)
        let resident = ResidentFilePlan(
            path: dir.appendingPathComponent("model_weights.bin").path,
            entries: [],
            stringTable: [],
            stringTableOffsets: [],
            indexSize: Layout.pageBytes,
            residentSize: 0)
        let layer0 = LayerFilePlan(
            layerIndex: 0,
            path: dir.appendingPathComponent("experts_layer_00.bin").path,
            expertsPerLayer: 1,
            expertStride: Layout.pageBytes,
            subTensors: [])
        let layer1 = LayerFilePlan(
            layerIndex: 1,
            path: dir.appendingPathComponent("experts_layer_01.bin").path,
            expertsPerLayer: 1,
            expertStride: Layout.pageBytes,
            subTensors: [])
        let plan = RepackPlan(
            arch: arch,
            baseMode: "affine",
            baseGroupSize: 64,
            bitsOverrideCount: 0,
            resident: resident,
            layers: [layer0, layer1],
            matchedModelID: nil,
            excludedMultimodalTensorNames: [])

        // Empty layers array must fail.
        let emptyPath = dir.appendingPathComponent("layout-empty.json").path
        try Data(#"{"layers":[]}"#.utf8).write(to: URL(fileURLWithPath: emptyPath))
        XCTAssertThrowsError(try GTurboLayoutValidator.validate(path: emptyPath, plan: plan))

        // Duplicate layer 0 (missing layer 1) must fail.
        let dupPath = dir.appendingPathComponent("layout-dup.json").path
        let dupJSON = """
        {
          "layers": [
            {"layer": 0, "experts": [{"expert": 0, "offset": 0, "size": \(Layout.pageBytes)}]},
            {"layer": 0, "experts": [{"expert": 0, "offset": 0, "size": \(Layout.pageBytes)}]}
          ]
        }
        """
        try Data(dupJSON.utf8).write(to: URL(fileURLWithPath: dupPath))
        XCTAssertThrowsError(try GTurboLayoutValidator.validate(path: dupPath, plan: plan))

        // Valid layers 0 and 1 must pass.
        let validPath = dir.appendingPathComponent("layout-valid.json").path
        let validJSON = """
        {
          "layers": [
            {"layer": 0, "experts": [{"expert": 0, "offset": 0, "size": \(Layout.pageBytes)}]},
            {"layer": 1, "experts": [{"expert": 0, "offset": 0, "size": \(Layout.pageBytes)}]}
          ]
        }
        """
        try Data(validJSON.utf8).write(to: URL(fileURLWithPath: validPath))
        XCTAssertNoThrow(try GTurboLayoutValidator.validate(path: validPath, plan: plan))
    }

    // MARK: - Wave 2 Verification Hardening (Items 20 & 21)

    private func sha256Hex(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }

    private func makeValidInstallDirectory(
        base: URL,
        name: String = "valid.gturbo",
        layerCount: Int = 1,
        duplicateLayerIndex: Bool = false,
        manifestOverridePath: String? = nil,
        layoutLayerFile: String = "experts_layer_00.bin"
    ) throws -> URL {
        let dir = base.appendingPathComponent(name, isDirectory: true)
        let packedDir = dir.appendingPathComponent("packed_experts", isDirectory: true)
        try FileManager.default.createDirectory(at: packedDir, withIntermediateDirectories: true)

        let pageSize = UInt64(getpagesize())
        let layerBytes = Data(repeating: 0xAB, count: Int(pageSize))
        let layerURL = packedDir.appendingPathComponent("experts_layer_00.bin")
        try layerBytes.write(to: layerURL)

        var layersArray: [[String: Any]] = []
        for idx in 0..<layerCount {
            let layerNumber = duplicateLayerIndex ? 0 : idx
            let fileName = idx == 0 ? layoutLayerFile : "experts_layer_0\(idx).bin"
            if idx > 0 {
                try layerBytes.write(to: packedDir.appendingPathComponent(fileName))
            }
            layersArray.append([
                "layer": layerNumber,
                "file": fileName,
                "experts": [["expert": 0, "offset": 0, "size": pageSize]],
            ])
        }

        let layoutObj: [String: Any] = [
            "expertStride": pageSize,
            "numLayers": layerCount,
            "expertsPerLayer": 1,
            "layers": layersArray,
        ]
        let layoutData = try JSONSerialization.data(withJSONObject: layoutObj, options: [.sortedKeys])
        let layoutURL = packedDir.appendingPathComponent("layout.json")
        try layoutData.write(to: layoutURL)

        var filesDict: [String: [String: Any]] = [
            "packed_experts/layout.json": [
                "size": UInt64(layoutData.count),
                "sha256": sha256Hex(layoutData),
            ],
            "packed_experts/\(layoutLayerFile)": [
                "size": UInt64(layerBytes.count),
                "sha256": sha256Hex(layerBytes),
            ],
        ]
        if layerCount > 1 {
            for idx in 1..<layerCount {
                filesDict["packed_experts/experts_layer_0\(idx).bin"] = [
                    "size": UInt64(layerBytes.count),
                    "sha256": sha256Hex(layerBytes),
                ]
            }
        }
        if let manifestOverridePath {
            filesDict[manifestOverridePath] = [
                "size": UInt64(layerBytes.count),
                "sha256": sha256Hex(layerBytes),
            ]
        }

        let manifestObj: [String: Any] = [
            "files": filesDict,
            "expertsPerLayer": 1,
            "numLayers": layerCount,
            "expertStride": pageSize,
            "sourceSnapshotHash": String(repeating: "a", count: 40),
        ]
        let manifestData = try JSONSerialization.data(withJSONObject: manifestObj, options: [.sortedKeys])
        try manifestData.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    func testWave2VerifiedInstallToolValidInstallWritesReceiptAndRejectsTraversalSymlinkDuplicateAndOverflow() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("wave2-verify-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }

        // 1. Valid install verifies and writes receipt atomically.
        let validDir = try makeValidInstallDirectory(base: base, name: "ok.gturbo")
        let result = try VerifiedInstallTool.run(options: VerifyInstallOptions(inputGTurbo: validDir.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.receiptPath))
        XCTAssertEqual(result.unexpectedEntries, [])
        XCTAssertEqual(result.fileCount, 3)

        // 2. Path traversal in manifest.files is rejected.
        let outsideFile = base.appendingPathComponent("secret.bin")
        try Data(repeating: 0xAB, count: Int(getpagesize())).write(to: outsideFile)
        let traversalDir = try makeValidInstallDirectory(
            base: base, name: "traversal.gturbo", manifestOverridePath: "../secret.bin")
        XCTAssertThrowsError(try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: traversalDir.path)))

        // 3. Path traversal in layout layer.file is rejected.
        let badLayerFileDir = try makeValidInstallDirectory(
            base: base, name: "bad-layer-file.gturbo", layoutLayerFile: "../secret.bin")
        XCTAssertThrowsError(try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: badLayerFileDir.path)))

        // 4. Symlinked payload file inside install root is rejected (O_NOFOLLOW).
        let symlinkDir = try makeValidInstallDirectory(base: base, name: "symlink.gturbo")
        let layerFile = symlinkDir.appendingPathComponent("packed_experts/experts_layer_00.bin")
        try FileManager.default.removeItem(at: layerFile)
        try FileManager.default.createSymbolicLink(at: layerFile, withDestinationURL: outsideFile)
        XCTAssertThrowsError(try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: symlinkDir.path)))

        // 5. Duplicate layer index in packed_experts/layout.json is rejected.
        let dupLayerDir = try makeValidInstallDirectory(
            base: base, name: "dup-layer.gturbo", layerCount: 2, duplicateLayerIndex: true)
        XCTAssertThrowsError(try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: dupLayerDir.path)))

        // 6. Overflow in expertsPerLayer * expertStride is rejected instead of trapping.
        let overflowDir = base.appendingPathComponent("overflow.gturbo", isDirectory: true)
        let overflowPacked = overflowDir.appendingPathComponent("packed_experts", isDirectory: true)
        try FileManager.default.createDirectory(at: overflowPacked, withIntermediateDirectories: true)
        let pageSize = UInt64(getpagesize())
        let hugeStride = (UInt64.max / pageSize) * pageSize
        let overflowLayout: [String: Any] = [
            "expertStride": hugeStride,
            "numLayers": 1,
            "expertsPerLayer": 2,
            "layers": [[
                "layer": 0,
                "file": "experts_layer_00.bin",
                "experts": [],
            ]],
        ]
        let overflowLayoutData = try JSONSerialization.data(withJSONObject: overflowLayout)
        try overflowLayoutData.write(to: overflowPacked.appendingPathComponent("layout.json"))
        let overflowManifest: [String: Any] = [
            "files": [
                "packed_experts/layout.json": [
                    "size": UInt64(overflowLayoutData.count),
                    "sha256": sha256Hex(overflowLayoutData),
                ],
            ],
            "expertsPerLayer": 2,
            "numLayers": 1,
            "expertStride": hugeStride,
        ]
        try JSONSerialization.data(withJSONObject: overflowManifest)
            .write(to: overflowDir.appendingPathComponent("manifest.json"))
        XCTAssertThrowsError(try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: overflowDir.path)))
    }

    func testWave2VisionPackVerifierPropagatesProgressErrorAndRejectsSymlinkedWeights() throws {
        struct ProgressCancelled: Error {}

        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("wave2-vision-\(UUID().uuidString)", isDirectory: true)
        let textDir = base.appendingPathComponent("text.gturbo", isDirectory: true)
        let visionDir = base.appendingPathComponent("text.vision.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: textDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: visionDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }

        let sourceSnapshot = String(repeating: "b", count: 40)
        let arch = GTurboManifestArchV1(
            hiddenSize: 64,
            ffnIntermediate: 128,
            moeIntermediateSize: 64,
            numHeads: 2,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 32,
            fullHeadDim: 32,
            vocabSize: 256,
            slidingWindow: 0,
            finalLogitSoftcap: 0,
            ropeTheta: 10_000,
            fullRopeTheta: 10_000,
            partialRotaryFactor: 1,
            numLayers: 1,
            numExperts: 2,
            topKExperts: 1,
            tieWordEmbeddings: true,
            attentionKEqV: false,
            hiddenActivation: "silu",
            fullAttentionLayerMask: [1])
        let textManifest = GTurboManifestV1(
            flags: [:],
            modelID: "mlx-community/gemma-4-26b-a4b-it-4bit",
            sourceSnapshotHash: sourceSnapshot,
            arch: arch,
            quant: nil,
            files: [:],
            expertsPerLayer: 2,
            numLayers: 1,
            expertStride: GTurboFormatV1.alignmentBytes,
            bitWidthOverridesHonored: nil)
        let textManifestData = try GTurboManifestCodec.encode(textManifest)
        try textManifestData.write(to: textDir.appendingPathComponent(GTurboVisionFormatV1.manifestFile))
        let textManifestSHA = sha256Hex(textManifestData)

        let weightsData = Data(repeating: 0x3C, count: Int(GTurboVisionFormatV1.alignmentBytes))
        let processorData = Data(#"{"image_size":224}"#.utf8)
        let weightsURL = visionDir.appendingPathComponent(GTurboVisionFormatV1.weightsFile)
        let processorURL = visionDir.appendingPathComponent(GTurboVisionFormatV1.processorFile)
        try weightsData.write(to: weightsURL)
        try processorData.write(to: processorURL)

        let weightsSHA = sha256Hex(weightsData)
        let processorSHA = sha256Hex(processorData)
        let visionManifest = GTurboVisionManifestV1(
            modelID: "mlx-community/gemma-4-26b-a4b-it-4bit",
            sourceRevision: sourceSnapshot,
            sourceIndexSha256: String(repeating: "c", count: 64),
            processorConfigSha256: processorSHA,
            compatibleTextSourceSnapshotHash: sourceSnapshot,
            compatibleTextManifestSha256: textManifestSHA,
            files: [
                GTurboVisionFormatV1.weightsFile: GTurboManifestFileV1(
                    size: UInt64(weightsData.count), sha256: weightsSHA),
                GTurboVisionFormatV1.processorFile: GTurboManifestFileV1(
                    size: UInt64(processorData.count), sha256: processorSHA),
            ],
            tensors: [
                GTurboVisionTensorRegionV1(
                    name: "vision_tower.weight",
                    executionPosition: 0,
                    offset: 0,
                    size: 16,
                    shape: [8],
                    dtype: .bf16),
            ])
        let visionManifestData = try GTurboVisionManifestCodec.encode(visionManifest)
        try visionManifestData.write(to: visionDir.appendingPathComponent(GTurboVisionFormatV1.manifestFile))
        let visionManifestSHA = sha256Hex(visionManifestData)

        let receipt = GTurboVisionReceiptV1(
            manifestSha256: visionManifestSHA,
            companionDirectoryPath: visionDir.standardizedFileURL.path,
            compatibleTextManifestSha256: textManifestSHA,
            sourceRepoID: visionManifest.modelID,
            sourceRevision: visionManifest.sourceRevision,
            verificationTimestamp: "2026-11-22T00:00:00Z",
            toolVersion: "test",
            files: [
                GTurboVisionFormatV1.manifestFile: GTurboManifestFileV1(
                    size: UInt64(visionManifestData.count), sha256: visionManifestSHA),
                GTurboVisionFormatV1.weightsFile: GTurboManifestFileV1(
                    size: UInt64(weightsData.count), sha256: weightsSHA),
                GTurboVisionFormatV1.processorFile: GTurboManifestFileV1(
                    size: UInt64(processorData.count), sha256: processorSHA),
            ])
        let receiptData = try JSONEncoder().encode(receipt)
        try receiptData.write(to: visionDir.appendingPathComponent(GTurboVisionFormatV1.receiptFile))

        // 1. Valid vision pack passes verification.
        let verified = try VisionPackVerifier.verify(
            directory: visionDir,
            textModelDirectory: textDir)
        XCTAssertEqual(verified.manifestSha256, visionManifestSHA)
        XCTAssertEqual(verified.weightsBytes, UInt64(weightsData.count))

        // 2. Error thrown from onHashProgress must propagate instead of being swallowed.
        XCTAssertThrowsError(try VisionPackVerifier.verify(
            directory: visionDir,
            textModelDirectory: textDir,
            onHashProgress: { _, _ in throw ProgressCancelled() }
        )) { error in
            XCTAssertTrue(error is ProgressCancelled)
        }

        // 3. Symlinked weights file must be rejected through GTurboDirectoryAccess.
        let externalWeights = base.appendingPathComponent("external_weights.bin")
        try weightsData.write(to: externalWeights)
        try FileManager.default.removeItem(at: weightsURL)
        try FileManager.default.createSymbolicLink(at: weightsURL, withDestinationURL: externalWeights)
        XCTAssertThrowsError(try VisionPackVerifier.verify(
            directory: visionDir,
            textModelDirectory: textDir))
    }
}
