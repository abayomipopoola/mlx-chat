import Foundation
import Testing
@testable import MLXChat

@Suite struct StreamedMoETests {
    @Test func runtimeSettingsOfferIndependentValidatedChoices() throws {
        #expect(StreamedMoERuntimeSettings.allowedContextTokens == [
            4_096, 8_192, 16_384, 32_768,
        ])
        #expect(StreamedMoERuntimeSettings.allowedExpertCacheSlots == [8, 16, 24, 32])

        let balanced = StreamedMoERuntimeSettings(
            contextTokens: 16_384,
            expertCacheSlots: 24)
        #expect(balanced.contextTokens == 16_384)
        #expect(balanced.expertCacheSlots == 24)

        let invalid = StreamedMoERuntimeSettings(
            contextTokens: 1_024,
            expertCacheSlots: 12)
        #expect(invalid.contextTokens == Keys.Defaults.streamedModelContextTokens)
        #expect(invalid.expertCacheSlots == Keys.Defaults.streamedModelExpertCacheSlots)
    }

    @Test func runtimeSettingsReadBothPersistedValuesIndependently() throws {
        let defaults = try #require(UserDefaults(
            suiteName: "org.mlxchat.tests.streamed-runtime-settings"))
        defaults.set(32_768, forKey: Keys.streamedModelContextTokens)
        defaults.set(8, forKey: Keys.streamedModelExpertCacheSlots)

        let settings = StreamedMoERuntimeSettings.current(defaults: defaults)

        #expect(settings.contextTokens == 32_768)
        #expect(settings.expertCacheSlots == 8)
    }

    @Test func installRequiresLocalTokenizerSidecars() {
        #expect(StreamedMoEInstall.requiredTokenizerPaths == [
            "tokenizer/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        ])
    }

    @Test func ssdOptionIsAvailableForAllPinnedMoESources() {
        #expect(StreamedMoEInstall.variants.count == 4)
        for variant in StreamedMoEInstall.variants {
            #expect(StreamedMoEInstall.supportsSSDStreaming(
                sourceModelID: variant.sourceRepoID))
            #expect(StreamedMoEInstall.variant(
                forStreamedModelID: variant.modelID) == variant)
            #expect(variant.modelID != variant.sourceRepoID)
            #expect(ModelDownloadMode.availableModes(
                for: variant.sourceRepoID) == [.mlx, .ssdStreamed])
        }
        #expect(!StreamedMoEInstall.supportsSSDStreaming(
            sourceModelID: "mlx-community/gemma-4-12B-it-4bit"))
        #expect(StreamedMoEInstall.standard.managedDirectory.lastPathComponent
            == "Gemma4-26B-A4B.gturbo")
        #expect(StreamedMoEInstall.qat.managedDirectory.lastPathComponent
            == "Gemma4-26B-A4B-QAT.gturbo")
        #expect(StreamedMoEInstall.qwen36.managedDirectory.lastPathComponent
            == "Qwen3.6-35B-A3B.gturbo")
        #expect(StreamedMoEInstall.qwen36.architecture == .qwen36)
        #expect(StreamedMoEInstall.qwen36.helperModelArgument == "qwen36")
        #expect(StreamedMoEInstall.ornith15.managedDirectory.lastPathComponent
            == "Ornith-1.5-35B-A3B.gturbo")
        #expect(StreamedMoEInstall.ornith15.architecture == .qwen36)
        #expect(StreamedMoEInstall.ornith15.helperModelArgument == "ornith15")
        #expect(ModelDownloadMode.defaultMode == .mlx)
        #expect(ModelDownloadMode.availableModes(
            for: "mlx-community/gemma-4-12B-it-4bit") == [.mlx])
        // Ornith-1.5-9B is a dense Qwen 3.5 checkpoint. SSD streaming is pinned
        // to Gemma 4 26B-A4B, Qwen3.6 35B-A3B, and Ornith 1.5 35B-A3B.
        #expect(!StreamedMoEInstall.supportsSSDStreaming(
            sourceModelID: "ornith-ai/Ornith-1.5-9B-MLX-4bit"))
        #expect(ModelDownloadMode.availableModes(
            for: "ornith-ai/Ornith-1.5-9B-MLX-4bit") == [.mlx])
        #expect(StreamedMoEInstall.supportsSSDStreaming(
            sourceModelID: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"))
        #expect(ModelDownloadMode.availableModes(
            for: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit") == [.mlx, .ssdStreamed])
    }

    @Test func streamedRemovalKeepsExternalFoldersButOwnsManagedArtifacts() {
        let managed = URL(
            fileURLWithPath: "/private/tmp/mlxchat-tests/Managed.gturbo",
            isDirectory: true)
        let external = URL(
            fileURLWithPath: "/private/tmp/mlxchat-tests/External.gturbo",
            isDirectory: true)

        #expect(ModelStore.streamedModelRemovalKind(
            registeredDirectory: managed,
            managedDirectory: managed,
            managedArtifactsExist: true) == .managed)
        #expect(ModelStore.streamedModelRemovalKind(
            registeredDirectory: external,
            managedDirectory: managed,
            managedArtifactsExist: true) == .external)
        #expect(ModelStore.streamedModelRemovalKind(
            registeredDirectory: nil,
            managedDirectory: managed,
            managedArtifactsExist: true) == .managed)
        #expect(ModelStore.streamedModelRemovalKind(
            registeredDirectory: nil,
            managedDirectory: managed,
            managedArtifactsExist: false) == nil)
    }

    @Test func managedRemovalMovesOnlyExistingArtifactsInSafeOrder() throws {
        let variant = StreamedMoEInstall.standard
        let artifacts = variant.managedInstallArtifactURLs
        let skipped = artifacts[1]
        var moved: [URL] = []

        try ModelStore.moveExistingManagedArtifactsToTrash(
            artifacts,
            fileExists: { $0 != skipped.path },
            moveToTrash: { moved.append($0) })

        #expect(artifacts.last == variant.managedDirectory)
        #expect(moved == [artifacts[0], artifacts[2]])
        #expect(moved.last == variant.managedDirectory)
    }

    @Test func managedInstallStateDistinguishesCompleteAndResumableFiles() {
        #expect(ModelStore.managedStreamedInstallState(
            finalDirectoryExists: false,
            partialDirectoryExists: false,
            checkpointExists: false) == .none)
        #expect(ModelStore.managedStreamedInstallState(
            finalDirectoryExists: false,
            partialDirectoryExists: true,
            checkpointExists: true) == .resumable)
        #expect(ModelStore.managedStreamedInstallState(
            finalDirectoryExists: true,
            partialDirectoryExists: true,
            checkpointExists: true) == .complete)

        let paths = StreamedMoEInstall.standard.managedInstallPaths
        #expect(paths.removableArtifacts == [
            paths.checkpointFile,
            paths.partialDirectory,
            paths.finalDirectory,
        ])
    }

    @Test func bundledInstallerIsEmbeddedAndRunnable() throws {
        let helper = try #require(StreamedModelInstaller.bundledHelperURL())
        #expect(FileManager.default.isExecutableFile(atPath: helper.path))

        let process = Process()
        let output = Pipe()
        process.executableURL = helper
        process.arguments = ["--help"]
        process.standardOutput = output
        try process.run()
        process.waitUntilExit()

        let text = String(
            decoding: output.fileHandleForReading.readDataToEndOfFile(),
            as: UTF8.self)
        #expect(process.terminationStatus == 0)
        #expect(text.contains("--model <gemma4|gemma4-qat|qwen36|ornith15>"))
        #expect(text.contains("--dry-run"))
    }

    @Test func installerProgressReadsResumableCheckpoint() throws {
        let data = try JSONSerialization.data(withJSONObject: [
            "totalSourceBytes": 1_000,
            "completedRanges": [
                ["sourceBytes": 200],
                ["sourceBytes": 300],
            ],
        ])
        let progress = try #require(StreamedModelInstaller.checkpointProgress(data: data))
        #expect(progress.phase == .downloading)
        #expect(progress.fraction == 0.5)
        #expect(progress.downloadedBytes == 500)
        #expect(progress.totalBytes == 1_000)
    }

    @Test func promptRendersSystemAndMultiTurnHistory() {
        let prompt = StreamedMoEPrompt.renderGemmaThinkingPrompt(
            history: [
                EngineMessage(role: .user, text: " First question "),
                EngineMessage(role: .assistant, text: " First answer "),
                EngineMessage(role: .user, text: " Follow-up "),
            ],
            systemPrompt: " Be concise. ")

        #expect(prompt == """
            <bos><|turn>system
            <|think|>
            Be concise.<turn|>
            <|turn>user
            First question<turn|>
            <|turn>model
            First answer<turn|>
            <|turn>user
            Follow-up<turn|>
            <|turn>model
            <|channel>thought
            """ + "\n")
    }

    @Test func thinkingPromptStartsInsideThoughtChannel() {
        let prompt = StreamedMoEPrompt.renderGemmaThinkingPrompt(
            history: [EngineMessage(role: .user, text: "Question")],
            systemPrompt: "Be accurate.")

        #expect(prompt.contains("<|turn>system\n<|think|>\nBe accurate.<turn|>"))
        #expect(prompt.hasSuffix("<|turn>model\n<|channel>thought\n"))
        #expect(!prompt.hasSuffix("<channel|>"))
    }

    @Test func qwenThinkingPromptUsesChatMLAndStartsInsideThinkSpan() {
        let prompt = StreamedMoEPrompt.renderChatMLThinkingPrompt(
            history: [
                EngineMessage(role: .user, text: " First question "),
                EngineMessage(role: .assistant, text: " First answer "),
                EngineMessage(role: .user, text: " Follow-up "),
            ],
            systemPrompt: " Be concise. ")

        #expect(prompt == """
            <|im_start|>system
            Be concise.<|im_end|>
            <|im_start|>user
            First question<|im_end|>
            <|im_start|>assistant
            First answer<|im_end|>
            <|im_start|>user
            Follow-up<|im_end|>
            <|im_start|>assistant
            <think>
            """ + "\n")
        #expect(!prompt.hasSuffix("</think>"))
    }

    @Test func strippedChannelEndTokenStillSeparatesThoughtFromAnswer() {
        let channelEndID: Int32 = 101
        let fencedHTML = """
            ```html
            <style>.board { display: grid; }</style>
            <div class="board"></div>
            <script>document.querySelector('.board')</script>
            ```
            """
        var router = StreamedMoEEventRouter(
            thinkingConfig: .gemmaChannel,
            thinkingEnabled: true,
            thoughtCloseID: channelEndID)
        var thinking = ""
        var content = ""
        var reclassified = false

        func collect(_ event: EngineEvent) {
            switch event {
            case .thinkingDelta(let delta): thinking += delta
            case .delta(let delta): content += delta
            case .thinkingReclassified: reclassified = true
            case .completed: break
            }
        }

        router.consume(tokenID: 1, delta: "Check distance and conditions.", emit: collect)
        // GFDetokenizer uses skipSpecialTokens=true, so the token's text delta
        // is empty even though its ID marks the end of the thought channel.
        router.consume(tokenID: channelEndID, delta: "", emit: collect)
        router.consume(tokenID: 2, delta: fencedHTML, emit: collect)
        router.finish(emit: collect)

        #expect(thinking == "Check distance and conditions.")
        #expect(content == fencedHTML)
        #expect(!reclassified)
    }

    @Test func channelEndTokenRemainsHiddenWhenThinkingIsOff() {
        let channelEndID: Int32 = 101
        var router = StreamedMoEEventRouter(
            thinkingConfig: .gemmaChannel,
            thinkingEnabled: false,
            thoughtCloseID: channelEndID)
        var content = ""

        func collect(_ event: EngineEvent) {
            if case .delta(let delta) = event { content += delta }
        }

        router.consume(tokenID: channelEndID, delta: "", emit: collect)
        router.consume(tokenID: 2, delta: "Direct answer.", emit: collect)
        router.finish(emit: collect)

        #expect(content == "Direct answer.")
    }

    @Test func strippedQwenThinkEndStillSeparatesThoughtFromAnswer() {
        let thinkEndID: Int32 = 151_668
        var router = StreamedMoEEventRouter(
            thinkingConfig: .thinkTags(startsInside: true, toggleable: true),
            thinkingEnabled: true,
            thoughtCloseID: thinkEndID)
        var thinking = ""
        var content = ""
        var reclassified = false

        func collect(_ event: EngineEvent) {
            switch event {
            case .thinkingDelta(let delta): thinking += delta
            case .delta(let delta): content += delta
            case .thinkingReclassified: reclassified = true
            case .completed: break
            }
        }

        router.consume(tokenID: 1, delta: "Compare the options.", emit: collect)
        router.consume(tokenID: thinkEndID, delta: "", emit: collect)
        router.consume(tokenID: 2, delta: "The answer is 42.", emit: collect)
        router.finish(emit: collect)

        #expect(thinking == "Compare the options.")
        #expect(content == "The answer is 42.")
        #expect(!reclassified)
    }
}
