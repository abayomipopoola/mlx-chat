import AppKit
import Foundation
import Testing
@testable import MLXChat

/// Focused regression tests for each Part 2 performance optimization via pure helpers.
@Suite("Performance fixes")
struct PerformanceFixTests {

    @Test("ModelStore deduplicates weight index shards and looks models up in O(1)")
    func shardDedupeAndO1Lookup() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-perf-shards-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))

        var weightMap: [String: String] = [:]
        for i in 0 ..< 500 {
            weightMap["layer.\(i).a"] = "model-00001-of-00002.safetensors"
            weightMap["layer.\(i).b"] = "model-00002-of-00002.safetensors"
        }
        let indexData = try JSONSerialization.data(withJSONObject: ["weight_map": weightMap])
        try indexData.write(to: dir.appendingPathComponent("model.safetensors.index.json"))

        var checkedPaths: [String] = []
        let ok = ModelStore.directoryHasWeights(dir) { url in
            checkedPaths.append(url.lastPathComponent)
            return true
        }
        #expect(ok)
        #expect(checkedPaths.count == 2)
        #expect(Set(checkedPaths) == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ])

        let custom = ModelCatalog.importedModel(repoID: "owner/Custom-Fast-Lookup")
        let customByID = [custom.id: custom]
        #expect(ModelStore.lookupModel(id: custom.id, customModelsByID: customByID)?.id == custom.id)
        #expect(
            ModelStore.lookupModel(
                id: "thinkingmachines/Inkling-2-Mini", customModelsByID: customByID)?.id
                == "thinkingmachines/Inkling-2-Mini")
        #expect(ModelStore.lookupModel(id: "owner/Missing", customModelsByID: customByID) == nil)
    }

    @Test("SnapshotDownloader skips root mtp.safetensors draft weights")
    func mtpSafetensorsSkip() {
        #expect(SnapshotDownloader.localPath(for: "mtp.safetensors", subfolder: nil) == nil)
        #expect(SnapshotDownloader.localPath(for: "sub/mtp.safetensors", subfolder: "sub") == nil)
        #expect(SnapshotDownloader.localPath(for: "mtp/model.safetensors", subfolder: nil) == nil)
        #expect(
            SnapshotDownloader.localPath(for: "model-00001-of-00001.safetensors", subfolder: nil)
                == "model-00001-of-00001.safetensors")
        #expect(SnapshotDownloader.localPath(for: "config.json", subfolder: nil) == "config.json")
    }

    @Test("MathPreprocessor fast-paths plain prose without scanning regexes")
    func plainTextFastPath() {
        let prose = "Hello world! This paragraph has numbers 123, punctuation, and no math at all."
        #expect(!MathPreprocessor.hasPotentialMath(prose))
        #expect(MathPreprocessor.preprocess(prose) == prose)

        #expect(MathPreprocessor.hasPotentialMath("Inline $x + 1$ here"))
        #expect(MathPreprocessor.hasPotentialMath(#"Display \[x^2\]"#))
        #expect(MathPreprocessor.hasPotentialMath("```math\nx^2\n```"))
        #expect(MathPreprocessor.hasPotentialMath("Sum ∑_{i=1}^{n} i"))
    }

    @Test("TwoGenerationCache promotes hits from previous and drops untouched older entries")
    func twoGenerationEviction() {
        var cache = TwoGenerationCache<String, Int>(capacity: 2)
        cache.insert(1, forKey: "a")
        cache.insert(2, forKey: "b")
        #expect(cache.currentCount == 2 && cache.previousCount == 0)

        // Third insert rotates current -> previous.
        cache.insert(3, forKey: "c")
        #expect(cache.currentCount == 1 && cache.previousCount == 2)

        // Touching "a" promotes it from previous back into current.
        #expect(cache.value(forKey: "a") == 1)
        #expect(cache.currentCount == 2 && cache.previousCount == 1)

        // Next insert rotates current ("c", "a") -> previous, dropping untouched "b".
        cache.insert(4, forKey: "d")
        #expect(cache.value(forKey: "b") == nil)
        #expect(cache.value(forKey: "a") == 1)
        #expect(cache.value(forKey: "c") == 3)
        #expect(cache.value(forKey: "d") == 4)
    }

    @Test("MessageSegmenter counts lines from UTF-8 newline bytes without splitting")
    func newlineByteCount() {
        #expect(MessageSegmenter.lineCount(of: "") == 0)
        #expect(MessageSegmenter.lineCount(of: "single line") == 1)
        #expect(MessageSegmenter.lineCount(of: "one\ntwo\nthree") == 3)
        #expect(MessageSegmenter.lineCount(of: "trailing\n") == 2)

        let artifact = Artifact(
            id: 0, kind: .code, language: "swift", title: "Swift code",
            content: "a\nb\nc\nd", isComplete: true)
        #expect(artifact.lineCount == 4)
    }

    @Test("DecodedMessageImageCache decodes once per message id and serves hits from cache")
    func imageCacheHit() {
        let cache = DecodedMessageImageCache(countLimit: 16)
        let id = UUID()
        let payload = Data([1, 2, 3])
        var decodeCalls = 0
        let first = cache.image(for: id, data: payload) { _ in
            decodeCalls += 1
            return NSImage(size: NSSize(width: 10, height: 10))
        }
        let second = cache.image(for: id, data: payload) { _ in
            decodeCalls += 1
            return NSImage(size: NSSize(width: 20, height: 20))
        }
        #expect(first != nil)
        #expect(second === first)
        #expect(decodeCalls == 1)
    }

    @Test("EngineReadinessNotice ticks only for Apple Intelligence or --fake-ai-preparing")
    func readinessTickGate() {
        #expect(
            !EngineReadinessNotice.shouldTickReadinessNotice(
                selectedModelID: "thinkingmachines/Inkling-2-Mini",
                arguments: ["MLXChat"]))
        #expect(
            EngineReadinessNotice.shouldTickReadinessNotice(
                selectedModelID: appleIntelligenceEngineID,
                arguments: ["MLXChat"]))
        #expect(
            EngineReadinessNotice.shouldTickReadinessNotice(
                selectedModelID: "thinkingmachines/Inkling-2-Mini",
                arguments: ["MLXChat", "--fake-ai-preparing"]))
    }

    @Test("SidebarView returns conversations directly on empty search without running matcher")
    func emptySearchPassthrough() {
        let items = ["Alpha", "Beta", "Gamma"]
        var predicateCalls = 0
        let empty = SidebarView.filterConversations(
            items, query: "   ", hideHistory: false
        ) { item, query in
            predicateCalls += 1
            return item.localizedCaseInsensitiveContains(query)
        }
        #expect(empty == items)
        #expect(predicateCalls == 0)

        let filtered = SidebarView.filterConversations(
            items, query: "bet", hideHistory: false
        ) { item, query in
            predicateCalls += 1
            return item.localizedCaseInsensitiveContains(query)
        }
        #expect(filtered == ["Beta"])
        #expect(predicateCalls == 3)
    }

    @Test("Active MLX buffer cache limit scales above 20 MB and caps at 512 MB")
    func activeBufferCacheLimitBounds() {
        let twentyMB = 20 * 1024 * 1024
        let fiveTwelveMB = 512 * 1024 * 1024
        for gigabytes: UInt64 in [4, 8, 16, 24, 36, 64, 128] {
            let limit = MLXEngine.activeBufferCacheLimit(
                physicalMemoryBytes: gigabytes * 1024 * 1024 * 1024)
            #expect(limit > twentyMB, "Expected > 20 MB for \(gigabytes) GB RAM, got \(limit)")
            #expect(limit <= fiveTwelveMB, "Expected <= 512 MB for \(gigabytes) GB RAM, got \(limit)")
        }
        #expect(
            MLXEngine.activeBufferCacheLimit(physicalMemoryBytes: 128 * 1024 * 1024 * 1024)
                == fiveTwelveMB)
    }

    @Test("MathRenderer uses numeric RGBA cache key and 512 entry limit; streaming tail helper isolates final block")
    func mathCacheKeyAndStreamingTail() {
        #expect(MathRenderer.cacheCountLimit == 512)
        let key = MathRenderer.cacheKey(
            base64URL: "eA", display: true, fontSize: 16,
            color: NSColor(srgbRed: 0.5, green: 0.25, blue: 1.0, alpha: 1.0))
        #expect(key == "d|160|128,64,255,255|eA")
        #expect(!key.contains("ColorSpace"))

        #expect(!MessageView.isStreamingTailBlock(isStreamingMessage: true, blockID: 0, lastBlockID: 2))
        #expect(MessageView.isStreamingTailBlock(isStreamingMessage: true, blockID: 2, lastBlockID: 2))
        #expect(!MessageView.isStreamingTailBlock(isStreamingMessage: false, blockID: 2, lastBlockID: 2))
    }
}
