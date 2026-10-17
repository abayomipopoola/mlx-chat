import CryptoKit
import Foundation
import Testing
@testable import MLXChat

@Suite struct SnapshotDownloaderTests {
    @Test func rootRepositoryPathsStayUnchanged() {
        #expect(SnapshotDownloader.localPath(
            for: "config.json", subfolder: nil) == "config.json")
        #expect(SnapshotDownloader.localPath(
            for: "model-00001-of-00002.safetensors", subfolder: nil)
            == "model-00001-of-00002.safetensors")
    }

    @Test func selectedSubfolderIsFlattened() {
        #expect(SnapshotDownloader.localPath(
            for: "4bit/config.json", subfolder: "4bit") == "config.json")
        #expect(SnapshotDownloader.localPath(
            for: "4bit/model.safetensors", subfolder: "4bit") == "model.safetensors")
    }

    @Test func weightsBelowTheModelRootAreSkipped() {
        #expect(SnapshotDownloader.localPath(
            for: "mtp/model.safetensors", subfolder: nil) == nil)
        #expect(SnapshotDownloader.localPath(
            for: "4bit/mtp/model.safetensors", subfolder: "4bit") == nil)
        #expect(SnapshotDownloader.localPath(
            for: "model.safetensors", subfolder: nil) == "model.safetensors")
    }

    @Test func otherPrecisionsAndUnneededFilesAreSkipped() {
        #expect(SnapshotDownloader.localPath(
            for: "bf16/config.json", subfolder: "4bit") == nil)
        #expect(SnapshotDownloader.localPath(
            for: "4bit/README.md", subfolder: "4bit") == nil)
    }

    @Test func nextLinkReadsRelNext() {
        #expect(SnapshotDownloader.nextLink(from: #"<https://example.com/page/2>; rel="next""#)
            == "https://example.com/page/2")
    }

    @Test func nextLinkAbsentWhenNoRelNext() {
        #expect(SnapshotDownloader.nextLink(from: "") == nil)
        #expect(SnapshotDownloader.nextLink(from: #"<https://example.com/page/1>; rel="prev""#) == nil)
    }

    @Test func nextLinkPicksNextFromMultiRelHeader() {
        let header = #"<https://example.com/prev>; rel="prev", <https://example.com/next>; rel="next", <https://example.com/last>; rel="last""#
        #expect(SnapshotDownloader.nextLink(from: header) == "https://example.com/next")
    }

    @Test func stagedFileSHAAcceptsMatchingHash() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-sha-accept-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        let data = Data("hello-weights".utf8)
        try data.write(to: url)
        let hex = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #expect(SnapshotDownloader.fileSHA256(url, matches: hex))
        #expect(SnapshotDownloader.fileSHA256(url, matches: "sha256:" + hex))
        #expect(SnapshotDownloader.canReuseStagedFile(at: url, size: Int64(data.count), sha256: hex))
        #expect(SnapshotDownloader.canReuseStagedFile(at: url, size: Int64(data.count), sha256: nil))
    }

    @Test func stagedFileSHARejectsMismatch() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-sha-reject-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        let data = Data("hello-weights".utf8)
        try data.write(to: url)
        let wrong = String(repeating: "0", count: 64)
        #expect(!SnapshotDownloader.fileSHA256(url, matches: wrong))
        #expect(!SnapshotDownloader.canReuseStagedFile(
            at: url, size: Int64(data.count), sha256: wrong))
        #expect(!SnapshotDownloader.canReuseStagedFile(
            at: url, size: Int64(data.count) + 1, sha256: nil))
    }
}
