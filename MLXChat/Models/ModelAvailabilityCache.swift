import Foundation

/// Caches `localMLXModelDirectory` hits and misses so SwiftUI bodies do not
/// walk `~/Models` on every render. Streamed registrations stay outside this
/// cache. `nil` is a stored miss, not a missing key.
final class ModelAvailabilityCache {
    private let probe: (String) -> URL?
    private var cachedDirectories: [String: URL?] = [:]

    init(probe: @escaping (String) -> URL? = ModelStore.localMLXModelDirectory(for:)) {
        self.probe = probe
    }

    func directory(for id: String) -> URL? {
        if let index = cachedDirectories.index(forKey: id) {
            return cachedDirectories[index].value
        }
        let url = probe(id)
        cachedDirectories.updateValue(url, forKey: id)
        return url
    }

    func invalidate(_ id: String) {
        cachedDirectories.removeValue(forKey: id)
    }

    func invalidateAll() {
        cachedDirectories.removeAll()
    }
}
