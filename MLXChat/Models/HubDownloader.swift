import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon

/// Adapts `HuggingFace.HubClient` to `MLXLMCommon.Downloader`.
/// Copied from mlx-swift-lm 3.31.4 `Documentation.docc/using.md` ("Implementing Protocols").
struct HubDownloader: MLXLMCommon.Downloader {
    private let upstream: HubClient

    init(_ upstream: HubClient) {
        self.upstream = upstream
    }

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        guard let repoID = HuggingFace.Repo.ID(rawValue: id) else {
            throw HuggingFaceDownloaderError.invalidRepositoryID(id)
        }
        let revision = revision ?? "main"

        return try await upstream.downloadSnapshot(
            of: repoID,
            revision: revision,
            matching: patterns,
            progressHandler: { @MainActor progress in
                progressHandler(progress)
            }
        )
    }
}
