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

    @Test func otherPrecisionsAndUnneededFilesAreSkipped() {
        #expect(SnapshotDownloader.localPath(
            for: "bf16/config.json", subfolder: "4bit") == nil)
        #expect(SnapshotDownloader.localPath(
            for: "4bit/README.md", subfolder: "4bit") == nil)
    }
}
