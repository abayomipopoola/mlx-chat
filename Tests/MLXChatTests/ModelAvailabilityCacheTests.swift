import Foundation
import Testing
@testable import MLXChat

@Suite struct ModelAvailabilityCacheTests {
    private final class Probe {
        var callCount = 0
        var result: URL?

        func call(_ id: String) -> URL? {
            callCount += 1
            return result
        }
    }

    @Test func cacheHitReusesURLWithoutReprobe() {
        let hit = URL(fileURLWithPath: "/model-availability-cache/hit", isDirectory: true)
        let probe = Probe()
        probe.result = hit
        let cache = ModelAvailabilityCache(probe: probe.call)

        #expect(cache.directory(for: "org/hit") == hit)
        #expect(probe.callCount == 1)
        #expect(cache.directory(for: "org/hit") == hit)
        #expect(probe.callCount == 1)
    }

    @Test func negativeCacheServesUnavailableIDWithoutReprobe() {
        let probe = Probe()
        probe.result = nil
        let cache = ModelAvailabilityCache(probe: probe.call)

        #expect(cache.directory(for: "org/miss") == nil)
        #expect(probe.callCount == 1)
        #expect(cache.directory(for: "org/miss") == nil)
        #expect(probe.callCount == 1)
    }

    @Test func invalidatingIDRefetchesAndFlipsResult() {
        let hit = URL(fileURLWithPath: "/model-availability-cache/hit", isDirectory: true)
        let probe = Probe()
        probe.result = nil
        let cache = ModelAvailabilityCache(probe: probe.call)

        #expect(cache.directory(for: "org/model") == nil)
        #expect(probe.callCount == 1)

        probe.result = hit
        #expect(cache.directory(for: "org/model") == nil)
        #expect(probe.callCount == 1)

        cache.invalidate("org/model")
        #expect(cache.directory(for: "org/model") == hit)
        #expect(probe.callCount == 2)
        #expect(cache.directory(for: "org/model") == hit)
        #expect(probe.callCount == 2)
    }

    @Test func invalidateAllClearsEveryID() {
        let first = URL(fileURLWithPath: "/model-availability-cache/first", isDirectory: true)
        let second = URL(fileURLWithPath: "/model-availability-cache/second", isDirectory: true)
        let probe = Probe()
        probe.result = first
        let cache = ModelAvailabilityCache(probe: probe.call)

        #expect(cache.directory(for: "org/one") == first)
        #expect(cache.directory(for: "org/two") == first)
        #expect(probe.callCount == 2)

        probe.result = second
        cache.invalidateAll()
        #expect(cache.directory(for: "org/one") == second)
        #expect(cache.directory(for: "org/two") == second)
        #expect(probe.callCount == 4)
    }
}
