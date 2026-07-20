import Testing
import UniformTypeIdentifiers
@testable import MLXChat

/// Contract: import repo-id normalization and attachment content-type allowlist.
@Suite struct ImportNormalizationTests {

    @Test func bareIdPassesThrough() {
        #expect(ImportModelSheet.normalizeRepoID("mlx-community/Foo-4bit")
            == "mlx-community/Foo-4bit")
    }

    @Test func fullHttpsURLStripsToOwnerRepo() {
        #expect(
            ImportModelSheet.normalizeRepoID(
                "https://huggingface.co/mlx-community/Foo-4bit")
            == "mlx-community/Foo-4bit")
    }

    @Test func hostOnlyWithExtraPathKeepsOwnerRepo() {
        #expect(ImportModelSheet.normalizeRepoID("huggingface.co/a/b/c") == "a/b")
    }

    /// Bare HF root normalizes to empty — documents why runImport no-ops even
    /// when the Import button is enabled (cheap emptiness gate, not normalize).
    @Test func bareHuggingFaceURLNormalizesToEmpty() {
        #expect(ImportModelSheet.normalizeRepoID("https://huggingface.co/") == "")
    }

    @Test @MainActor func fileContentTypesIncludePDFAndPlainText() {
        let types = AttachmentPicker.fileContentTypes
        #expect(types.contains(.pdf))
        #expect(types.contains(.plainText))
    }
}
