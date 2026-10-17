import Foundation
import XCTest
@testable import TurboFieldfareRepackCore

final class VisionCompanionInstallTests: XCTestCase {
    func testHelpDocumentsVisionWriteMode() {
        let usage = RepackCommandArguments.usage
        XCTAssertTrue(usage.contains("--write-vision"))
        XCTAssertTrue(usage.contains("--text-model <text.gturbo>"))
        XCTAssertTrue(usage.contains("--output <name.vision.gturbo>"))
        XCTAssertTrue(usage.contains("--model <gemma4|gemma4-qat>"))
        XCTAssertTrue(usage.contains("[--overwrite] [--resume]"))
        XCTAssertTrue(usage.contains("Vision-write mode"))
        XCTAssertTrue(usage.contains("--model <gemma4|gemma4-qat|qwen36|ornith15>"))
    }

    func testWriteVisionParseRequiresGemmaAndTextModel() throws {
        let parsed = try RepackCommandArguments.parse([
            "TurboFieldfareRepack",
            "--write-vision",
            "--model", "gemma4-qat",
            "--text-model", "/tmp/Gemma4-26B-A4B-QAT.gturbo",
            "--output", "/tmp/Gemma4-26B-A4B-QAT.vision.gturbo",
            "--overwrite",
            "--resume",
        ])
        XCTAssertEqual(parsed.mode, .writeVision)
        XCTAssertEqual(parsed.model, .gemma4QAT)
        XCTAssertEqual(parsed.textModel, "/tmp/Gemma4-26B-A4B-QAT.gturbo")
        XCTAssertEqual(parsed.output, "/tmp/Gemma4-26B-A4B-QAT.vision.gturbo")
        XCTAssertTrue(parsed.overwrite)
        XCTAssertTrue(parsed.resume)
    }

    func testWriteVisionRejectsQwenAndOrnith() {
        for name in ["qwen36", "ornith15"] {
            XCTAssertThrowsError(try RepackCommandArguments.parse([
                "TurboFieldfareRepack",
                "--write-vision",
                "--model", name,
                "--text-model", "/tmp/text.gturbo",
                "--output", "/tmp/name.vision.gturbo",
            ])) { error in
                XCTAssertEqual(
                    error as? RepackCommandParseError,
                    .invalidMode("vision-write accepts only --model gemma4 or gemma4-qat"))
            }
        }
    }

    func testDefaultTextInstallStillAcceptsAllFourModels() throws {
        for name in ["gemma4", "gemma4-qat", "qwen36", "ornith15"] {
            let parsed = try RepackCommandArguments.parse([
                "TurboFieldfareRepack",
                "--model", name,
                "--output", "/tmp/\(name).gturbo",
            ])
            XCTAssertEqual(parsed.mode, .install)
            XCTAssertEqual(parsed.model.name, name)
            XCTAssertNil(parsed.textModel)
        }
    }

    func testTextInstallDoesNotAcceptTextModelFlag() {
        XCTAssertThrowsError(try RepackCommandArguments.parse([
            "TurboFieldfareRepack",
            "--output", "/tmp/model.gturbo",
            "--text-model", "/tmp/text.gturbo",
        ])) { error in
            XCTAssertEqual(
                error as? RepackCommandParseError,
                .invalidMode("--text-model requires --write-vision"))
        }
    }

    func testVisionCompanionOptionsRejectsQwen() {
        XCTAssertThrowsError(try SupportedModelSource.qwen36.visionCompanionOptions(
            textModelDirectory: URL(fileURLWithPath: "/tmp/text.gturbo"),
            outputDirectory: URL(fileURLWithPath: "/tmp/name.vision.gturbo"),
            overwrite: false,
            token: nil))
        XCTAssertTrue(SupportedModelSource.gemma4.supportsVisionCompanion)
        XCTAssertTrue(SupportedModelSource.gemma4QAT.supportsVisionCompanion)
        XCTAssertFalse(SupportedModelSource.qwen36.supportsVisionCompanion)
        XCTAssertFalse(SupportedModelSource.ornith15.supportsVisionCompanion)
    }

    func testPlanVisionCompanionKeepsPinnedTensorCountAndBytes() throws {
        let plan = try RepackPlanner.planVisionCompanion(
            meta: visionMetadata(),
            shardHeaders: [Safetensors.Header(tensors: pinnedVisionTensors())])
        XCTAssertEqual(plan.entries.count, 358)
        XCTAssertEqual(plan.sourcePayloadBytes, 1_140_925_536)
    }

    func testPlanVisionCompanionRejectsWrongTensorCountOrBytes() {
        var short = pinnedVisionTensors()
        short.removeLast()
        XCTAssertThrowsError(try RepackPlanner.planVisionCompanion(
            meta: visionMetadata(),
            shardHeaders: [Safetensors.Header(tensors: short)]))

        var wrongBytes = pinnedVisionTensors()
        let last = wrongBytes.removeLast()
        wrongBytes.append(SourceTensor(
            name: last.name, shardPath: last.shardPath, dtype: .bf16,
            shape: [2], absoluteOffset: last.absoluteOffset, sizeBytes: 4))
        XCTAssertThrowsError(try RepackPlanner.planVisionCompanion(
            meta: visionMetadata(),
            shardHeaders: [Safetensors.Header(tensors: wrongBytes)]))
    }

    func testRangeCopyPlannerCopiesOnlyVisionTensors() throws {
        let shard = "model-00001-of-00002.safetensors"
        let first = SourceTensor(
            name: "vision_tower.patch_embedder.weight", shardPath: shard,
            dtype: .bf16, shape: [32], absoluteOffset: 64, sizeBytes: 64)
        let second = SourceTensor(
            name: "embed_vision.proj.weight", shardPath: shard,
            dtype: .bf16, shape: [16], absoluteOffset: 256, sizeBytes: 32)
        let plan = VisionPackPlan(
            entries: [
                .init(source: first, executionPosition: 0, fileOffset: 0,
                      quantSpec: nil, groupSize: 64),
                .init(source: second, executionPosition: 1, fileOffset: 16_384,
                      quantSpec: nil, groupSize: 64),
            ],
            weightsFileSize: 16_416,
            sourcePayloadBytes: 96)
        let rangePlan = try RangeCopyPlanner.plan(
            vision: plan,
            weightsPath: "/tmp/partial/vision_weights.bin",
            rangeChunkBytes: RemoteChunkPolicy.defaultBytes)
        XCTAssertEqual(rangePlan.scalarCopies.count, 2)
        XCTAssertEqual(rangePlan.scalarCopies.map(\.size), [64, 32])
        XCTAssertEqual(rangePlan.expectedOutputs, [
            RemoteExpectedOutput(relativePath: "vision_weights.bin", size: 16_416)
        ])
        XCTAssertEqual(rangePlan.remoteBytesToDownload, rangePlan.coalescedCopies.reduce(0) { $0 + $1.size })
    }

    func testFakeProviderCopiesVisionRangesWithoutARemoteFetch() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-range-\(UUID().uuidString)", isDirectory: true)
        let partial = root.appendingPathComponent("partial", isDirectory: true)
        try FileManager.default.createDirectory(at: partial, withIntermediateDirectories: true)
        let weights = partial.appendingPathComponent("vision_weights.bin")
        let payload = Data((0..<96).map { UInt8($0) })
        let shard = "vision-shard.safetensors"
        let source = SourceTensor(
            name: "vision_tower.patch_embedder.weight", shardPath: shard,
            dtype: .bf16, shape: [48], absoluteOffset: 0, sizeBytes: 96)
        let plan = VisionPackPlan(
            entries: [
                .init(source: source, executionPosition: 0, fileOffset: 0,
                      quantSpec: nil, groupSize: 64)
            ],
            weightsFileSize: 96,
            sourcePayloadBytes: 96)
        let rangePlan = try RangeCopyPlanner.plan(
            vision: plan,
            weightsPath: weights.path,
            rangeChunkBytes: RemoteChunkPolicy.defaultBytes)
        FileManager.default.createFile(atPath: weights.path, contents: Data(count: 96))
        let provider = FixtureSourceByteProvider(payloads: [shard: payload])
        let audit = RepackAudit()
        try await provider.copyBatch(
            rangePlan.coalescedCopies,
            completedRangeIDs: [],
            partialDirectory: partial.path,
            temporaryPath: partial.appendingPathComponent("range.tmp").path,
            audit: audit,
            progress: { _ in },
            commit: { _ in })
        XCTAssertEqual(try Data(contentsOf: weights), payload)
    }

    private func visionMetadata() -> IndexLoader.SourceMetadata {
        IndexLoader.SourceMetadata(
            indexPath: "/tmp/index.json",
            configPath: "/tmp/config.json",
            indexSha256Hex: SupportedModelSource.gemma4.sourceIndexSHA256,
            weightMap: [:],
            baseBits: 4,
            baseGroupSize: 64,
            baseMode: "affine",
            bitsOverrides: [:],
            shardFilenames: ["model.safetensors"])
    }

    private func pinnedVisionTensors() -> [SourceTensor] {
        let total: UInt64 = 1_140_925_536
        let count = 358
        let headCount = count - 1
        let headSize: UInt64 = 3_186_384
        let tailSize = total - headSize * UInt64(headCount)
        return (0..<count).map { index in
            let size = index == headCount ? tailSize : headSize
            let name = index == headCount
                ? "embed_vision.proj.weight"
                : "vision_tower.encoder.layers.\(index).weight"
            return SourceTensor(
                name: name,
                shardPath: "model.safetensors",
                dtype: .bf16,
                shape: [size / 2],
                absoluteOffset: UInt64(index) * headSize,
                sizeBytes: size)
        }
    }
}

private final class FixtureSourceByteProvider: SourceByteProvider {
    private let payloads: [String: Data]

    init(payloads: [String: Data]) {
        self.payloads = payloads
    }

    func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws {
        for copy in copies where !completedRangeIDs.contains(copy.id) {
            guard let payload = payloads[copy.shardID] else {
                throw RepackError.configurationInvalid(detail: "missing fixture \(copy.shardID)")
            }
            let start = Int(copy.sourceOffset)
            let end = start + Int(copy.size)
            let slice = payload.subdata(in: start..<end)
            for destination in copy.destinations {
                let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: destination.destinationPath))
                try handle.seek(toOffset: destination.destinationOffset)
                try handle.write(contentsOf: slice)
                try handle.close()
            }
            try commit(RemoteCompletedRange(
                id: copy.id,
                destinationDigest: String(repeating: "ab", count: 32),
                sourceBytes: copy.size,
                destinationBytes: copy.destinations.reduce(0) { $0 + $1.size }))
            progress(copy.size)
        }
    }
}
