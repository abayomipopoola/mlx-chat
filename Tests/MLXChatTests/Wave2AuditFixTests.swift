import AppKit
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers
@testable import MLXChat

private final class DummyLocalEngine: ChatEngine, @unchecked Sendable {
    let id: String
    var isLoaded: Bool

    init(id: String, isLoaded: Bool = true) {
        self.id = id
        self.isLoaded = isLoaded
    }

    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        isLoaded = true
    }

    func unload() {
        isLoaded = false
    }

    func stream(
        history: [EngineMessage],
        systemPrompt: String,
        temperature: Double,
        conversationID: UUID
    ) -> AsyncThrowingStream<EngineEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(.completed(tokensPerSecond: nil))
            continuation.finish()
        }
    }
}

private actor ConcurrencyProbe {
    private var active = 0
    private(set) var maxConcurrent = 0
    private(set) var events: [String] = []

    func enter(_ label: String) {
        active += 1
        if active > maxConcurrent { maxConcurrent = active }
        events.append("enter:\(label)")
    }

    func exit(_ label: String) {
        events.append("exit:\(label)")
        active -= 1
    }
}

@Suite(.serialized) struct Wave2AuditFixTests {
    // MARK: - Group 1: Selected-model fallback, EngineRuntime, MLXEngine, StreamedMoE

    @Test("Item 1: A1 fallback prefers Apple Intelligence when available, else first usable local, else Apple Intelligence")
    func selectedModelFallbackPolicy() {
        #expect(
            ChatController.resolveFallbackModelID(
                excluding: "owner/model-a",
                appleIntelligenceAvailable: true,
                usableLocalModelIDs: ["owner/model-a", "owner/model-b"])
            == appleIntelligenceEngineID)

        #expect(
            ChatController.resolveFallbackModelID(
                excluding: "owner/model-a",
                appleIntelligenceAvailable: false,
                usableLocalModelIDs: ["owner/model-a", "owner/model-b"])
            == "owner/model-b")

        #expect(
            ChatController.resolveFallbackModelID(
                excluding: "owner/model-a",
                appleIntelligenceAvailable: false,
                usableLocalModelIDs: ["owner/model-a"])
            == appleIntelligenceEngineID)

        #expect(
            ChatController.resolveFallbackModelID(
                excluding: nil,
                appleIntelligenceAvailable: false,
                usableLocalModelIDs: [])
            == appleIntelligenceEngineID)
    }

    @Test("Item 1: EngineRuntime.residentModelID excludes downloading/loading and rejects mid-stream local switch")
    @MainActor
    func engineRuntimeResidentModelIDAndMidStreamSwitchGuard() async throws {
        let store = ModelStore()
        let modelA = ModelCatalog.importedModel(repoID: "wave2/test-a")
        let modelB = ModelCatalog.importedModel(repoID: "wave2/test-b")
        store.registerCustomModelForTesting(modelA)
        store.registerCustomModelForTesting(modelB)

        let runtime = EngineRuntime(modelStore: store)
        let engineA = DummyLocalEngine(id: modelA.id, isLoaded: true)

        runtime.installSlot(engineA)
        #expect(runtime.loadedModelID == modelA.id)
        #expect(runtime.residentModelID == nil)

        try runtime.loadDidSucceed(engineA, id: modelA.id)
        #expect(runtime.residentModelID == modelA.id)

        runtime.beginGenerating(id: modelA.id)
        #expect(runtime.residentModelID == modelA.id)

        var rejectedSwitch = false
        do {
            _ = try await runtime.engine(for: modelB.id)
        } catch EngineError.unavailable {
            rejectedSwitch = true
        }
        #expect(rejectedSwitch)
        #expect(runtime.residentModelID == modelA.id)

        runtime.endGenerating()
        runtime.unloadCurrent()
        #expect(runtime.residentModelID == nil)
    }

    @Test("Item 1: EngineRuntime local-to-local switch unloads prior model and finishes in ready state")
    @MainActor
    func engineRuntimeLocalToLocalSwitchEndsReady() async throws {
        let store = ModelStore()
        let modelA = ModelCatalog.importedModel(repoID: "wave2/switch-a")
        let modelB = ModelCatalog.importedModel(repoID: "wave2/switch-b")
        store.registerCustomModelForTesting(modelA)
        store.registerCustomModelForTesting(modelB)
        store.isAvailableOverride = { [modelA.id, modelB.id].contains($0) }

        let runtime = EngineRuntime(modelStore: store)
        var created: [String: DummyLocalEngine] = [:]
        runtime.localEngineFactory = { catalogModel in
            let engine = DummyLocalEngine(id: catalogModel.id, isLoaded: false)
            created[catalogModel.id] = engine
            return engine
        }

        let resolvedA = try await runtime.engine(for: modelA.id)
        #expect(resolvedA.id == modelA.id)
        #expect(runtime.state == .ready(id: modelA.id))
        #expect(runtime.residentModelID == modelA.id)
        #expect(created[modelA.id]?.isLoaded == true)

        let resolvedB = try await runtime.engine(for: modelB.id)
        #expect(resolvedB.id == modelB.id)
        #expect(runtime.state == .ready(id: modelB.id))
        #expect(runtime.residentModelID == modelB.id)
        #expect(created[modelA.id]?.isLoaded == false)
        #expect(created[modelB.id]?.isLoaded == true)
    }

    @Test("Item 2: MLXEngine rebuilds session on same-count edited history and invalidates load generation on unload")
    func mlxEngineSessionSignatureAndLoadGeneration() {
        let convID = UUID()
        let initialPrior = [
            EngineMessage(role: .user, text: "hello"),
            EngineMessage(role: .assistant, text: "first answer"),
        ]
        let knownSignature = initialPrior.map {
            MLXEngine.PriorMessageKey(role: $0.role, text: $0.text)
        }

        // Identical prior reuses the session.
        #expect(!MLXEngine.shouldRebuildSession(
            hasSession: true,
            sessionConversationID: convID,
            sessionSystemPrompt: "sys",
            sessionKnownPriorSignature: knownSignature,
            conversationID: convID,
            systemPrompt: "sys",
            prior: initialPrior))

        // Same count (2) but regenerated/edited assistant text forces rebuild.
        let regeneratedPrior = [
            EngineMessage(role: .user, text: "hello"),
            EngineMessage(role: .assistant, text: "regenerated answer"),
        ]
        #expect(MLXEngine.shouldRebuildSession(
            hasSession: true,
            sessionConversationID: convID,
            sessionSystemPrompt: "sys",
            sessionKnownPriorSignature: knownSignature,
            conversationID: convID,
            systemPrompt: "sys",
            prior: regeneratedPrior))

        // Unloading bumps load generation so a racing load cannot resurrect a container.
        let engine = MLXEngine(model: ModelCatalog.importedModel(repoID: "wave2/mlx-gen"))
        let genBefore = engine.currentLoadGeneration
        #expect(engine.isLoadGenerationCurrent(genBefore))
        engine.unload()
        #expect(!engine.isLoadGenerationCurrent(genBefore))
    }

    @Test("Item 3: StreamedMoESession serializes reentrant generate calls and disables multimodal thinking routing")
    func streamedMoESessionSerializationAndMultimodalThinking() async throws {
        // B-B3: Multimodal turns never open thinking on the event router.
        #expect(!StreamedMoEEngine.routerThinkingEnabled(
            architecture: .gemma4, thinkingEnabled: true, wantsImages: true))
        #expect(!StreamedMoEEngine.routerThinkingEnabled(
            architecture: .flint, thinkingEnabled: true, wantsImages: true))
        #expect(StreamedMoEEngine.routerThinkingEnabled(
            architecture: .gemma4, thinkingEnabled: true, wantsImages: false))
        #expect(!StreamedMoEEngine.routerThinkingEnabled(
            architecture: .gemma4, thinkingEnabled: false, wantsImages: false))
        #expect(StreamedMoEEngine.routerThinkingEnabled(
            architecture: .flint, thinkingEnabled: false, wantsImages: false))

        // B-B2: Overlapping runSerialized turns cancel and await turn 1 before turn 2 enters.
        let session = StreamedMoESession()
        let probe = ConcurrencyProbe()
        let firstEntered = AsyncStream<Void>.makeStream()

        let task1 = Task {
            try await session.runSerialized { _ in
                await probe.enter("first")
                firstEntered.continuation.yield(())
                firstEntered.continuation.finish()
                do {
                    while !Task.isCancelled {
                        try await Task.sleep(for: .milliseconds(10))
                    }
                } catch {}
                await probe.exit("first")
                throw CancellationError()
            }
        }

        for await _ in firstEntered.stream { break }

        let secondResult = try await session.runSerialized { _ in
            await probe.enter("second")
            try? await Task.sleep(for: .milliseconds(15))
            await probe.exit("second")
            return 42
        }
        _ = await task1.result

        #expect(secondResult == 42)
        let maxConcurrent = await probe.maxConcurrent
        let events = await probe.events
        #expect(maxConcurrent == 1)
        #expect(events == ["enter:first", "exit:first", "enter:second", "exit:second"])
    }

    // MARK: - Group 2: Attachments, Downloads, ModelStore, Installer, Support, Catalog

    @Test("Item 4: FileTextExtractor accepts expanded text/source types and extensionless UTF-8 while rejecting binary")
    func fileTextExtractorExpandedAllowlist() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("wave2-extract-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        for ext in ["tsx", "jsx", "kt", "sql", "svg", "plist", "zsh"] {
            let file = dir.appendingPathComponent("sample.\(ext)")
            try Data("hello-\(ext)".utf8).write(to: file)
            #expect(try FileTextExtractor.extract(url: file) == "hello-\(ext)")
        }

        // Extensionless valid UTF-8 text file is accepted.
        let makefile = dir.appendingPathComponent("Makefile")
        try Data("all:\n\techo ok\n".utf8).write(to: makefile)
        #expect(try FileTextExtractor.extract(url: makefile) == "all:\n\techo ok\n")

        // Extensionless binary file containing NUL bytes is rejected.
        let binary = dir.appendingPathComponent("rawbin")
        try Data([0x7F, 0x45, 0x4C, 0x46, 0x00, 0x01]).write(to: binary)
        #expect(throws: FileTextExtractor.ExtractError.self) {
            _ = try FileTextExtractor.extract(url: binary)
        }

        // .zip archive continues to throw .unsupportedType.
        let zip = dir.appendingPathComponent("archive.zip")
        try Data([0x50, 0x4B, 0x03, 0x04]).write(to: zip)
        do {
            _ = try FileTextExtractor.extract(url: zip)
            Issue.record("Expected .zip to throw .unsupportedType")
        } catch FileTextExtractor.ExtractError.unsupportedType {
            // Expected
        }
    }

    @Test("Item 5: SnapshotDownloader.verifyDownloadedFile enforces byte count and SHA-256")
    func snapshotDownloaderVerifiesDownloadedFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("wave2-snap-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let file = dir.appendingPathComponent("weights.safetensors")
        let payload = Data("mlx-weight-bytes".utf8)
        try payload.write(to: file)
        let sha = try SnapshotDownloader.sha256Hex(ofFileAt: file)

        try SnapshotDownloader.verifyDownloadedFile(
            at: file, expectedSize: Int64(payload.count), expectedSHA256: sha, remotePath: "model.safetensors")

        #expect(throws: SnapshotDownloader.DownloadError.self) {
            try SnapshotDownloader.verifyDownloadedFile(
                at: file, expectedSize: Int64(payload.count + 1), expectedSHA256: sha, remotePath: "model.safetensors")
        }

        #expect(throws: SnapshotDownloader.DownloadError.self) {
            try SnapshotDownloader.verifyDownloadedFile(
                at: file,
                expectedSize: Int64(payload.count),
                expectedSHA256: String(repeating: "0", count: 64),
                remotePath: "model.safetensors")
        }
    }

    @Test("Item 6 & 7: ModelStore per-model downloadErrors, gemmaVisionPackMissing, and installer skip")
    @MainActor
    func modelStoreDownloadErrorsAndInstallerVisionSkip() {
        let store = ModelStore()
        // Restricted model surfaces per-model error cleanly without starting network download.
        let badID = "invalid-single-segment-id"
        store.download(badID)
        #expect(store.downloadError(for: badID) != nil)
        #expect(store.downloadError(for: "unrelated/model") == nil)

        // Non-Gemma4 streamed model never reports Gemma vision pack missing.
        let ornithStatus = store.gemmaVisionPackMissing(for: StreamedMoEInstall.ornith15.modelID)
        #expect(!ornithStatus.missing)

        // StreamedModelInstaller skips vision companion when already present or non-Gemma4.
        #expect(StreamedModelInstaller.shouldInstallVisionCompanion(
            architecture: .gemma4, visionCompanionPresent: false))
        #expect(!StreamedModelInstaller.shouldInstallVisionCompanion(
            architecture: .gemma4, visionCompanionPresent: true))
        #expect(!StreamedModelInstaller.shouldInstallVisionCompanion(
            architecture: .flint, visionCompanionPresent: false))
    }

    @Test("Item 8: ModelSupport rejects bools, floats, nested PyTorch quant, bad per-layer quant, and empty vision_config")
    func modelSupportHardening() {
        // Boolean bits rejected.
        let boolBits = Data(#"{"model_type":"qwen3","quantization":{"bits":true,"group_size":64,"mode":"affine"}}"#.utf8)
        #expect(ModelSupport.unsupportedReason(configJSON: boolBits) != nil)

        // Fractional float bits rejected.
        let floatBits = Data(#"{"model_type":"qwen3","quantization":{"bits":4.5,"group_size":64,"mode":"affine"}}"#.utf8)
        #expect(ModelSupport.unsupportedReason(configJSON: floatBits) != nil)

        // Nested text_config.quantization_config PyTorch method rejected.
        let nestedPyTorch = Data(#"{"model_type":"gemma4","text_config":{"quantization_config":{"quant_method":"fp8"}}}"#.utf8)
        #expect(ModelSupport.unsupportedReason(configJSON: nestedPyTorch) != nil)

        // Per-layer quantization override with unsupported bit width rejected.
        let badPerLayer = Data(#"{"model_type":"qwen3","quantization":{"bits":4,"group_size":64,"mode":"affine","lm_head":{"bits":1,"group_size":64}}}"#.utf8)
        #expect(ModelSupport.unsupportedReason(configJSON: badPerLayer) != nil)

        // Per-layer quantization override with `false` (unquantized layer) and valid override passes.
        let goodPerLayer = Data(#"{"model_type":"qwen3","quantization":{"bits":4,"group_size":64,"mode":"affine","lm_head":false,"embed":{"bits":8,"group_size":64}}}"#.utf8)
        #expect(ModelSupport.unsupportedReason(configJSON: goodPerLayer) == nil)

        // General nvfp4/mxfp4/mxfp8 imports remain allowed (A3).
        let mxfp4 = Data(#"{"model_type":"qwen3","quantization":{"bits":4,"group_size":32,"mode":"mxfp4"}}"#.utf8)
        #expect(ModelSupport.unsupportedReason(configJSON: mxfp4) == nil)
        let nvfp4 = Data(#"{"model_type":"qwen3","quantization":{"bits":4,"group_size":16,"mode":"nvfp4"}}"#.utf8)
        #expect(ModelSupport.unsupportedReason(configJSON: nvfp4) == nil)

        // Empty vision_config dictionary does not mark a shared model type as vision-capable.
        let emptyVision = Data(#"{"model_type":"gemma4","vision_config":{}}"#.utf8)
        #expect(!ModelSupport.visionCapable(configJSON: emptyVision))
        let realVision = Data(#"{"model_type":"gemma4","vision_config":{"hidden_size":1152}}"#.utf8)
        #expect(ModelSupport.visionCapable(configJSON: realVision))
    }

    @Test("Item 9: ModelCatalog imported Gemma 4 gets <turn|> EOS and imported models honor max_position_embeddings")
    func modelCatalogImportedGemma4AndMaxPositionEmbeddings() {
        let gemma4Config = Data(#"{"model_type":"gemma4","max_position_embeddings":131072}"#.utf8)
        let importedGemma = ModelCatalog.importedModel(
            repoID: "mlx-community/custom-gemma-4-it-4bit",
            configJSON: gemma4Config)
        #expect(importedGemma.extraEOSTokens.contains("<turn|>"))
        #expect(!importedGemma.extraEOSTokens.contains("<|im_end|>"))
        #expect(importedGemma.thinking == .gemmaChannel)

        let llamaConfig = Data(#"{"model_type":"llama","max_position_embeddings":131072}"#.utf8)
        let importedLlama = ModelCatalog.importedModel(
            repoID: "mlx-community/custom-llama-8b-4bit",
            configJSON: llamaConfig)
        #expect(importedLlama.ctxTokens == 131_072)

        let hugeConfig = Data(#"{"model_type":"llama","max_position_embeddings":99999999}"#.utf8)
        let cappedLlama = ModelCatalog.importedModel(
            repoID: "mlx-community/custom-llama-huge-4bit",
            configJSON: hugeConfig)
        #expect(cappedLlama.ctxTokens == 1_048_576)
    }

    // MARK: - Group 3: Rendering, Math, View Helpers, Window Nudge, EXIF

    @Test("Item 10: CodeHighlighter strips Highlightr font attributes so code font stays uniform")
    func codeHighlighterStripsFontAttributes() throws {
        let attributed = try #require(
            HighlightrHighlighter.light.highlightedAttributedString("let x = 42", language: "swift"))
        for run in attributed.runs {
            let hasAppKitFont = run.appKit.font != nil
            let hasSwiftUIFont = run.font != nil
            #expect(!hasAppKitFont)
            #expect(!hasSwiftUIFont)
        }
    }

    @Test("Item 12: MathRenderer.cacheKey distinguishes display style and light/dark ink colors")
    func mathRendererCacheKeyDistinguishesStyleAndInk() {
        let encoded = MathPreprocessor.base64URL("x^2")
        let lightKey = MathRenderer.cacheKey(
            base64URL: encoded, display: false, fontSize: 15, color: NSColor(white: 0.12, alpha: 1))
        let darkKey = MathRenderer.cacheKey(
            base64URL: encoded, display: false, fontSize: 15, color: NSColor(white: 0.92, alpha: 1))
        let displayKey = MathRenderer.cacheKey(
            base64URL: encoded, display: true, fontSize: 15, color: NSColor(white: 0.12, alpha: 1))
        #expect(lightKey != darkKey)
        #expect(lightKey != displayKey)
    }

    @Test("Item 13: MathPreprocessor protects multi-backtick code, CRLF code blocks, and currency phrases")
    func mathPreprocessorMultiBacktickCRLFAndCurrency() {
        // Multi-backtick inline code containing single backticks and math stays literal.
        let multiBacktick = "Use `` `$x_1$` `` literally."
        #expect(MathPreprocessor.preprocess(multiBacktick) == multiBacktick)

        // CRLF fenced code block stays protected.
        let crlfFence = "```swift\r\nlet formula = \"$x_1$\"\r\n```\r\n"
        #expect(MathPreprocessor.preprocess(crlfFence) == crlfFence)

        // Currency false positives stay literal.
        let perDay = "It costs $5/day and $10/month."
        #expect(MathPreprocessor.preprocess(perDay) == perDay)

        let wordPrice = "Between $5/day and $ten total."
        #expect(MathPreprocessor.preprocess(wordPrice) == wordPrice)

        // Currency amount followed by real inline math still transforms the math.
        let mixed = "Costs $5 and $x^2 + 1$."
        let mixedOut = MathPreprocessor.preprocess(mixed)
        #expect(mixedOut.hasPrefix("Costs $5 and "))
        #expect(mixedOut.contains("swiftmath://i/"))
    }

    @Test("Item 14: ChatScreen.isStreamingConversation and ComposerView.measuringDraftText")
    func chatScreenAndComposerHelpers() {
        let idA = UUID()
        let idB = UUID()
        #expect(ChatScreen.isStreamingConversation(streamingMessageID: idB, messageIDs: [idA, idB]))
        #expect(!ChatScreen.isStreamingConversation(streamingMessageID: UUID(), messageIDs: [idA, idB]))
        #expect(!ChatScreen.isStreamingConversation(streamingMessageID: nil, messageIDs: [idA, idB]))

        #expect(ComposerView.measuringDraftText(for: "") == " ")
        #expect(ComposerView.measuringDraftText(for: "line1\n") == "line1\n\u{200B}")
        #expect(ComposerView.measuringDraftText(for: "hello") == "hello")
    }

    @Test("Item 16: SidebarView rename enablement and relative timestamp formatting")
    func sidebarRenameAndRelativeTimestampHelpers() {
        #expect(!SidebarView.canConfirmRename(""))
        #expect(!SidebarView.canConfirmRename("   \n\t  "))
        #expect(SidebarView.canConfirmRename("New Title"))

        let now = Date(timeIntervalSince1970: 1_700_000_000)
        #expect(SidebarView.relativeLabel(for: now.addingTimeInterval(-30), relativeTo: now) == "just now")
        let twoMinAgo = SidebarView.relativeLabel(for: now.addingTimeInterval(-120), relativeTo: now)
        #expect(twoMinAgo != "just now")
    }

    @Test("Item 17: ModelsPage search deduplication and ModelRow SSD button visibility")
    func modelsPageDeduplicationAndAddStreamedVisibility() {
        let sourceID = StreamedMoEInstall.ornith15.sourceRepoID
        let streamedID = StreamedMoEInstall.ornith15.modelID
        let restrictedSource = CatalogModel(
            id: sourceID,
            displayName: "Ornith Source",
            sizeGB: 20,
            ctxTokens: 32_768,
            quantLabel: "4-bit",
            category: .general,
            blurb: "Source",
            extraEOSTokens: [],
            thinking: nil,
            backend: .mlx,
            installRestriction: "SSD only")
        let streamedModel = CatalogModel(
            id: streamedID,
            displayName: "Ornith SSD",
            sizeGB: 20,
            ctxTokens: 32_768,
            quantLabel: "4-bit",
            category: .general,
            blurb: "Streamed",
            extraEOSTokens: [],
            thinking: nil,
            backend: .streamedMoE)

        let deduped = ModelsPage.deduplicatedSearchResults(
            [restrictedSource, streamedModel],
            visibleStreamedIDs: [streamedID])
        #expect(deduped.map(\.id) == [streamedID])

        let keptWhenStreamedHidden = ModelsPage.deduplicatedSearchResults(
            [restrictedSource],
            visibleStreamedIDs: [])
        #expect(keptWhenStreamedHidden.map(\.id) == [sourceID])

        #expect(ModelRow.showsAddStreamedVersionButton(
            sourceModelID: sourceID, isStreamedAvailable: false, isStreamedDownloading: false))
        #expect(!ModelRow.showsAddStreamedVersionButton(
            sourceModelID: sourceID, isStreamedAvailable: true, isStreamedDownloading: false))
        #expect(!ModelRow.showsAddStreamedVersionButton(
            sourceModelID: sourceID, isStreamedAvailable: false, isStreamedDownloading: true))
    }

    @Test("Item 18: WindowOnScreenNudge.clampedFrame clamps oversized windows inside visible bounds")
    func windowOnScreenNudgeClampedFrame() {
        let visible = NSRect(x: 0, y: 25, width: 1280, height: 775)
        let oversizedOffscreen = NSRect(x: -2000, y: -1000, width: 1600, height: 1000)
        let clamped = WindowOnScreenNudge.clampedFrame(oversizedOffscreen, within: visible)

        #expect(clamped.width == visible.width)
        #expect(clamped.height == visible.height)
        #expect(clamped.minX >= visible.minX)
        #expect(clamped.minY >= visible.minY)
        #expect(clamped.maxX <= visible.maxX)
        #expect(clamped.maxY <= visible.maxY)
    }

    @Test("Item 19: AttachmentImage.jpegData(from: URL) applies EXIF orientation 6 (90° CW) to portrait dimensions")
    func attachmentImagePreservesEXIFOrientation() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("wave2-exif-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        // Create an 80x40 landscape bitmap tagged with EXIF orientation 6 (90° CW -> 40x80 portrait).
        let rawWidth = 80
        let rawHeight = 40
        let rep = try #require(NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: rawWidth,
            pixelsHigh: rawHeight,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0))
        let cgImage = try #require(rep.cgImage)

        let url = dir.appendingPathComponent("portrait-exif6.jpg")
        let destination = try #require(CGImageDestinationCreateWithURL(
            url as CFURL, UTType.jpeg.identifier as CFString, 1, nil))
        let props: [CFString: Any] = [
            kCGImagePropertyOrientation: CGImagePropertyOrientation.right.rawValue,
        ]
        CGImageDestinationAddImage(destination, cgImage, props as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))

        let outputData = try #require(AttachmentImage.jpegData(from: url, maxDimension: 1024))
        let outSource = try #require(CGImageSourceCreateWithData(outputData as CFData, nil))
        let outImage = try #require(CGImageSourceCreateImageAtIndex(outSource, 0, nil))
        #expect(outImage.width == rawHeight)
        #expect(outImage.height == rawWidth)
    }
}
