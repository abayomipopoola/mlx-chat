import Foundation
import XCTest
@testable import TurboFieldfareRepackCore

final class SupportedModelSourceTests: XCTestCase {
    func testAllPinnedSourcesRemainDistinctAndSelectable() {
        let sources = SupportedModelSource.all

        XCTAssertEqual(sources.map(\.name), ["gemma4", "gemma4-qat", "qwen36"])
        XCTAssertEqual(Set(sources.map(\.repoID)).count, sources.count)
        XCTAssertEqual(Set(sources.map(\.revision)).count, sources.count)
        XCTAssertEqual(Set(sources.map(\.sourceIndexSHA256)).count, sources.count)
        XCTAssertEqual(SupportedModelSource.named("standard"), .gemma4)
        XCTAssertEqual(SupportedModelSource.named("qat"), .gemma4QAT)
        XCTAssertEqual(SupportedModelSource.named("qwen36"), .qwen36)
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
}
