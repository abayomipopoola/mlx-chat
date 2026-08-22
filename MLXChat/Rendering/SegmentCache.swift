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

    private var entries: [UUID: Entry] = [:]
    private var order: [UUID] = []
    private let capacity: Int

    init(capacity: Int = 256) {
        self.capacity = capacity
    }

    func segments(for id: UUID, content: String) -> [RenderBlock] {
        if let cached = entries[id], cached.content == content {
            touch(id)
            return cached.blocks
        }
        let blocks = MessageSegmenter.segment(content)
        entries[id] = Entry(content: content, blocks: blocks)
        touch(id)
        evictIfNeeded()
        return blocks
    }

    private func touch(_ id: UUID) {
        if let index = order.firstIndex(of: id) {
            order.remove(at: index)
        }
        order.append(id)
    }

    private func evictIfNeeded() {
        while order.count > capacity {
            let evicted = order.removeFirst()
            entries.removeValue(forKey: evicted)
        }
    }
}
