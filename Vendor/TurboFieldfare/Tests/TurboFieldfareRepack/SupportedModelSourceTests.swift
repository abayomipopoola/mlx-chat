import Foundation
import XCTest
@testable import TurboFieldfareRepackCore

final class SupportedModelSourceTests: XCTestCase {
    func testAllPinnedSourcesRemainDistinctAndSelectable() {
        let sources = SupportedModelSource.all

        XCTAssertEqual(sources.map(\.name), ["gemma4", "gemma4-qat", "qwen36", "ornith15", "lfm25"])
        XCTAssertEqual(Set(sources.map(\.repoID)).count, sources.count)
        XCTAssertEqual(Set(sources.map(\.revision)).count, sources.count)
        XCTAssertEqual(Set(sources.map(\.sourceIndexSHA256)).count, sources.count)
        XCTAssertEqual(SupportedModelSource.named("standard"), .gemma4)
        XCTAssertEqual(SupportedModelSource.named("qat"), .gemma4QAT)
        XCTAssertEqual(SupportedModelSource.named("qwen36"), .qwen36)
        XCTAssertEqual(SupportedModelSource.named("ornith15"), .ornith15)
    }

    func testQwenFingerprintUsesCanonicalManifestModelID() {
        XCTAssertEqual(SourceFingerprint.modelID(
            forIndexSha256: SupportedModelSource.qwen36.sourceIndexSHA256),
            "qwen3.6-35b-a3b-4bit")
    }

    func testQwenSourceMatchesTheValidatedUpstreamCheckpoint() {
        let qwen = SupportedModelSource.qwen36

        XCTAssertEqual(qwen.repoID, "mlx-community/Qwen3.6-35B-A3B-4bit")
        XCTAssertEqual(qwen.revision, "38740b847e4cb78f352aba30aa41c76e08e6eb46")
        XCTAssertEqual(qwen.sourceIndexSHA256,
            "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea")
        XCTAssertEqual(qwen.approximateDownloadBytes, 19_529_025_048)
        XCTAssertEqual(qwen.installedBytes, 19_546_491_213)
    }

    func testOrnith15SourceMatchesTheValidatedUpstreamCheckpoint() {
        let ornith = SupportedModelSource.ornith15

        XCTAssertEqual(ornith.repoID, "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit")
        XCTAssertEqual(ornith.revision, "19504d912fa8fc7622bf6b1de3db5d5d890b1f02")
        XCTAssertEqual(ornith.sourceIndexSHA256,
            "c118f13c0dcb729e4ca2e3d653ab193067551eb1a6410badb5192eb426104f36")
        XCTAssertEqual(ornith.modelID, "ornith-1.5-35b-a3b-4bit")
        XCTAssertEqual(ornith.approximateDownloadBytes, 19_528_808_680)
        XCTAssertEqual(ornith.installedBytes, 19_546_274_845)
        XCTAssertEqual(
            SourceFingerprint.modelID(forIndexSha256: ornith.sourceIndexSHA256),
            "ornith-1.5-35b-a3b-4bit")
    }
}
