import Foundation

/// Memoizes `MessageSegmenter.segment` per message id so finished rows skip
/// re-parsing when a sibling flush rebuilds the transcript.
@MainActor
final class SegmentCache {
    static let shared = SegmentCache()

    private struct Entry {
        var content: String
        var blocks: [RenderBlock]
    }

    private var cache: TwoGenerationCache<UUID, Entry>

    init(capacity: Int = 256) {
        self.cache = TwoGenerationCache(capacity: capacity)
    }

    func segments(for id: UUID, content: String) -> [RenderBlock] {
        if let cached = cache.value(forKey: id), cached.content == content {
            return cached.blocks
        }
        let blocks = MessageSegmenter.segment(content)
        cache.insert(Entry(content: content, blocks: blocks), forKey: id)
        return blocks
    }
}
