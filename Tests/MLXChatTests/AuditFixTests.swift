import AppKit
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import PDFKit
import SwiftData
import Testing
import WebKit
@testable import MLXChat

// MARK: - Local test doubles

private struct AuditTestError: Error, LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

private final class ControlledChatEngine: ChatEngine, @unchecked Sendable {
    let id: String
    var isLoaded: Bool

    private let lock = NSLock()
    private var loadHandler: ((@escaping @Sendable (Double) -> Void) async throws -> Void)?
    private var streamContinuation: AsyncThrowingStream<EngineEvent, Error>.Continuation?
    private var onStreamStarted: (() -> Void)?

    init(
        id: String,
        isLoaded: Bool = true,
        loadHandler: ((@escaping @Sendable (Double) -> Void) async throws -> Void)? = nil,
        onStreamStarted: (() -> Void)? = nil
    ) {
        self.id = id
        self.isLoaded = isLoaded
        self.loadHandler = loadHandler
        self.onStreamStarted = onStreamStarted
    }

    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        if let loadHandler {
            try await loadHandler(progress)
        }
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
            lock.withLock {
                self.streamContinuation = continuation
            }
            onStreamStarted?()
        }
    }

    func yield(_ event: EngineEvent) {
        let cont = lock.withLock { streamContinuation }
        cont?.yield(event)
    }

    func finish(throwing error: Error? = nil) {
        let cont = lock.withLock { () -> AsyncThrowingStream<EngineEvent, Error>.Continuation? in
            let current = streamContinuation
            streamContinuation = nil
            return current
        }
        if let error {
            cont?.finish(throwing: error)
        } else {
            cont?.finish()
        }
    }
}

private final class MinimalAuditTokenizer: Tokenizer, @unchecked Sendable {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private actor AsyncGate {
    private var opened = false
    private var waiters: [UUID: CheckedContinuation<Void, Error>] = [:]

    func wait(timeout: Duration = .seconds(5)) async throws {
        if opened { return }
        let id = UUID()
        let timeoutTask = Task {
            try await Task.sleep(for: timeout)
            self.fail(id: id)
        }
        defer { timeoutTask.cancel() }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            if opened {
                continuation.resume()
            } else {
                waiters[id] = continuation
            }
        }
    }

    private func fail(id: UUID) {
        if let waiter = waiters.removeValue(forKey: id) {
            waiter.resume(throwing: AuditTestError(message: "Timed out waiting on AsyncGate"))
        }
    }

    func open() {
        opened = true
        let pending = waiters.values
        waiters.removeAll()
        for waiter in pending {
            waiter.resume()
        }
    }
}

private final class LockedBox<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value

    init(_ value: Value) { self.value = value }

    func withValue<T>(_ body: (inout Value) -> T) -> T {
        lock.withLock { body(&value) }
    }

    func read() -> Value {
        lock.withLock { value }
    }
}

@MainActor
private func makeAuditConversationStore() throws -> (container: SwiftData.ModelContainer, store: ConversationStore) {
    let schema = Schema([Conversation.self, ChatMessage.self])
    let configuration = SwiftData.ModelConfiguration(
        "audit-fix-tests-\(UUID().uuidString)",
        schema: schema,
        isStoredInMemoryOnly: true)
    let container = try SwiftData.ModelContainer(for: schema, configurations: [configuration])
    return (container, ConversationStore(context: container.mainContext))
}

private func withRestoredUserDefaults(
    keys: [String],
    body: () async throws -> Void
) async rethrows {
    let defaults = UserDefaults.standard
    let saved: [String: Any?] = Dictionary(
        uniqueKeysWithValues: keys.map { ($0, defaults.object(forKey: $0)) })
    defer {
        for (key, value) in saved {
            if let value {
                defaults.set(value, forKey: key)
            } else {
                defaults.removeObject(forKey: key)
            }
        }
    }
    try await body()
}

private func makeTempDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("mlxchat-audit-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func makePDF(text: String) throws -> Data {
    let pageRect = CGRect(x: 0, y: 0, width: 200, height: 100)
    let data = NSMutableData()
    guard let consumer = CGDataConsumer(data: data as CFMutableData) else {
        throw AuditTestError(message: "Failed to create CGDataConsumer")
    }
    var mediaBox = pageRect
    guard let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
        throw AuditTestError(message: "Failed to create PDF CGContext")
    }
    context.beginPDFPage(nil)
    let nsContext = NSGraphicsContext(cgContext: context, flipped: false)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = nsContext
    (text as NSString).draw(
        in: NSRect(x: 10, y: 40, width: 180, height: 40),
        withAttributes: [.font: NSFont.systemFont(ofSize: 12)])
    NSGraphicsContext.restoreGraphicsState()
    context.endPDFPage()
    context.closePDF()
    return data as Data
}

// MARK: - App & Controller Audit Fixes (AUDIT-01..AUDIT-04, AUDIT-23..AUDIT-25)

@Suite(.serialized)
struct AuditControllerAndRuntimeTests {

    @MainActor
    @Test("AUDIT-01: conversationWillBeDeleted invalidates active turn and prevents stale finishTurn mutations")
    func audit01ConversationDeletionInvalidatesActiveTurn() async throws {
        try await withRestoredUserDefaults(keys: [Keys.selectedModelID, Keys.customModels]) {
            let fixture = try makeAuditConversationStore()
            let store = fixture.store
            let chat = ChatController()
            chat.conversationStore = store

            let modelID = "audit-org/audit-01-model"
            let model = ModelCatalog.importedModel(repoID: modelID)
            chat.modelStore.registerCustomModelForTesting(model)
            chat.selectModel(modelID)

            let streamStarted = AsyncGate()
            let fakeEngine = ControlledChatEngine(
                id: modelID,
                isLoaded: true,
                onStreamStarted: {
                    Task { await streamStarted.open() }
                })
            chat.runtime.localEngineFactory = { _ in fakeEngine }
            // Bypass download check by installing a pre-loaded engine after selecting a known available path
            // or registering a fake download handler that doesn't touch disk.
            let dir = try makeTempDirectory()
            defer { try? FileManager.default.removeItem(at: dir) }

            let plain = try #require(ModelStore.plainDirectory(for: modelID))
            // Keep test completely isolated: install the loaded engine directly and skip download via fake handler.
            chat.modelStore.mlxDownloadHandler = { _, _, _, _ in }
            chat.runtime.installSlot(fakeEngine)
            try chat.runtime.loadDidSucceed(fakeEngine, id: modelID)

            // Wait: engine(for:) checks modelStore.isAvailable(id). Instead of touching ~/Models,
            // verify using Apple Intelligence unavailable? No — create a temporary plain directory
            // ONLY if it doesn't exist, and clean up in defer, so modelStore.isAvailable(modelID) is true.
            let fileManager = FileManager.default
            let orgDir = plain.deletingLastPathComponent()
            let orgExisted = fileManager.fileExists(atPath: orgDir.path)
            try fileManager.createDirectory(at: plain, withIntermediateDirectories: true)
            try Data(#"{"model_type":"qwen3"}"#.utf8).write(to: plain.appendingPathComponent("config.json"))
            try Data([1, 2, 3, 4]).write(to: plain.appendingPathComponent("model.safetensors"))
            defer {
                try? fileManager.removeItem(at: plain)
                if !orgExisted {
                    try? fileManager.removeItem(at: orgDir)
                }
            }
            chat.modelStore.noteExternalModelDirectoryMayHaveChanged()

            let conversation = store.newConversation()
            var saveCountAfterDelete = 0
            var deleted = false
            store.saveExecutor = {
                if deleted { saveCountAfterDelete += 1 }
                try fixture.container.mainContext.save()
            }

            chat.send("Hello before delete", in: conversation)
            try await streamStarted.wait()
            #expect(chat.isStreaming)
            #expect(chat.streamingMessageID != nil)

            chat.conversationWillBeDeleted(conversation)
            deleted = true
            store.delete(conversation)
            let savesRightAfterDelete = saveCountAfterDelete

            // Late stream events and errors after deletion must be ignored completely.
            fakeEngine.yield(.delta("stale delta"))
            fakeEngine.finish(throwing: AuditTestError(message: "late stream error"))
            try? await Task.sleep(for: .milliseconds(40))

            #expect(!chat.isStreaming)
            #expect(chat.streamingMessageID == nil)
            #expect(chat.errorBanner == nil)
            #expect(!chat.canRetry)
            #expect(saveCountAfterDelete == savesRightAfterDelete)
            withExtendedLifetime(fixture) {}
        }
    }

    @MainActor
    @Test("AUDIT-02: ChatController tracks a single warmLoadTask and cancels prior warm loads")
    func audit02WarmLoadTaskCancelsPreviousLoad() async throws {
        try await withRestoredUserDefaults(keys: [Keys.selectedModelID]) {
            let chat = ChatController()
            let firstCancelled = LockedBox(false)
            let firstGate = AsyncGate()

            chat.scheduleWarmLoad {
                try? await firstGate.wait()
                if Task.isCancelled {
                    firstCancelled.withValue { $0 = true }
                }
            }
            let firstTask = try #require(chat.currentWarmLoadTask)

            chat.scheduleWarmLoad {
                try? await Task.sleep(for: .seconds(5))
            }
            let secondTask = try #require(chat.currentWarmLoadTask)
            #expect(firstTask.isCancelled)

            await firstGate.open()
            await firstTask.value
            #expect(firstCancelled.read())

            // Switching to Apple Intelligence cancels the in-flight warm load task.
            chat.selectModel("mlx-community/Qwen3-0.6B-4bit")
            chat.scheduleWarmLoad {
                try? await Task.sleep(for: .seconds(5))
            }
            let thirdTask = try #require(chat.currentWarmLoadTask)
            chat.selectModel(appleIntelligenceEngineID)
            #expect(thirdTask.isCancelled)
            #expect(chat.currentWarmLoadTask == nil)
            secondTask.cancel()
        }
    }

    @MainActor
    @Test("AUDIT-03: errorBanner pairs with retryConversation and retry/regenerate respect preparing guard")
    func audit03PairedErrorBannerAndPreparingGuards() async throws {
        try await withRestoredUserDefaults(keys: [Keys.selectedModelID]) {
            let fixture = try makeAuditConversationStore()
            let store = fixture.store
            let chat = ChatController()
            chat.conversationStore = store

            let conversation = store.newConversation()
            store.appendMessage(role: "user", content: "Hi", to: conversation)
            let assistant = store.appendMessage(role: "assistant", content: "Hello", to: conversation)

            // Paired setter / overwrite / dismiss behavior.
            chat.setErrorBanner("Initial failure", retryConversation: conversation)
            #expect(chat.errorBanner == "Initial failure")
            #expect(chat.canRetry)

            chat.errorBanner = "Unrelated banner"
            #expect(chat.errorBanner == "Unrelated banner")
            #expect(!chat.canRetry)

            chat.setErrorBanner("Retryable failure", retryConversation: conversation)
            #expect(chat.canRetry)
            chat.dismissErrorBanner()
            #expect(chat.errorBanner == nil)
            #expect(!chat.canRetry)

            // While the selected model is preparing (.loading), retry() and regenerate() are no-ops.
            let modelID = "mlx-community/Qwen3-0.6B-4bit"
            chat.selectModel(modelID)
            let loadingEngine = ControlledChatEngine(id: modelID, isLoaded: false)
            chat.runtime.installSlot(loadingEngine)
            #expect(chat.isSelectedModelPreparing)

            chat.setErrorBanner("Retryable failure", retryConversation: conversation)
            chat.retry()
            #expect(!chat.isStreaming)
            #expect(chat.errorBanner == "Retryable failure")
            #expect(chat.canRetry)

            chat.regenerate(assistant, in: conversation)
            #expect(!chat.isStreaming)
            #expect(conversation.orderedMessages.count == 2)
            #expect(chat.canRetry)

            withExtendedLifetime(fixture) {}
        }
    }

    @MainActor
    @Test("AUDIT-04: EngineRuntime monotonic request token rejects stale downloads, progress, and loads")
    func audit04EngineRuntimeMonotonicRequestToken() async throws {
        let modelStore = ModelStore()
        let runtime = EngineRuntime(modelStore: modelStore)

        let modelA = "audit-org/runtime-model-a"
        let modelB = "audit-org/runtime-model-b"
        modelStore.registerCustomModelForTesting(ModelCatalog.importedModel(repoID: modelA))
        modelStore.registerCustomModelForTesting(ModelCatalog.importedModel(repoID: modelB))

        let downloadGate = AsyncGate()
        let downloadStarted = AsyncGate()
        modelStore.mlxDownloadHandler = { _, _, _, _ in
            await downloadStarted.open()
            try? await downloadGate.wait()
            throw AuditTestError(message: "model A download failed")
        }

        // Request 1 enters downloadAndWait for model A.
        let request1 = Task { try await runtime.engine(for: modelA) }
        try await downloadStarted.wait()
        #expect(runtime.state == .downloading(id: modelA))

        // Request 2 supersedes the slot with model B.
        let engineB = ControlledChatEngine(id: modelB, isLoaded: false)
        runtime.installSlot(engineB)
        #expect(runtime.state == .loading(id: modelB, fraction: 0))

        // Request 1's download now fails; its stale catch must NOT wipe model B's state.
        await downloadGate.open()
        do {
            _ = try await request1.value
            Issue.record("Expected request 1 to throw after model A download failure")
        } catch {
            // Expected failure
        }
        #expect(runtime.state == .loading(id: modelB, fraction: 0))

        // Also verify late progress callback from an older engine load is ignored once superseded.
        let loadStarted = AsyncGate()
        let loadGate = AsyncGate()
        let engineA = ControlledChatEngine(id: modelA, isLoaded: false) { progress in
            await loadStarted.open()
            try? await loadGate.wait()
            progress(0.75)
        }
        runtime.localEngineFactory = { model in
            model.id == modelA ? engineA : engineB
        }

        let plainA = try #require(ModelStore.plainDirectory(for: modelA))
        let fm = FileManager.default
        let orgDir = plainA.deletingLastPathComponent()
        let orgExisted = fm.fileExists(atPath: orgDir.path)
        try fm.createDirectory(at: plainA, withIntermediateDirectories: true)
        try Data(#"{"model_type":"qwen3"}"#.utf8).write(to: plainA.appendingPathComponent("config.json"))
        try Data([1, 2, 3, 4]).write(to: plainA.appendingPathComponent("model.safetensors"))
        defer {
            try? fm.removeItem(at: plainA)
            if !orgExisted {
                try? fm.removeItem(at: orgDir)
            }
        }
        modelStore.noteExternalModelDirectoryMayHaveChanged()

        let loadTaskA = Task { try await runtime.engine(for: modelA) }
        try await loadStarted.wait()

        // Supersede with model B before model A reports progress or finishes.
        runtime.installSlot(engineB)
        await loadGate.open()
        do {
            _ = try await loadTaskA.value
            Issue.record("Expected superseded engine(for: modelA) to throw CancellationError")
        } catch is CancellationError {
            // Expected
        }
        try? await Task.sleep(for: .milliseconds(30))
        #expect(runtime.state == .loading(id: modelB, fraction: 0))
    }

    @Test("AUDIT-23: EscKeyMonitor matches bare Escape only and ignores modified Escape chords")
    func audit23EscKeyMonitorBareEscapeOnly() {
        #expect(EscKeyMonitor.isBareEscapeKey(keyCode: 53, modifiers: []))
        #expect(!EscKeyMonitor.isBareEscapeKey(keyCode: 53, modifiers: [.option]))
        #expect(!EscKeyMonitor.isBareEscapeKey(keyCode: 53, modifiers: [.command]))
        #expect(!EscKeyMonitor.isBareEscapeKey(keyCode: 53, modifiers: [.control]))
        #expect(!EscKeyMonitor.isBareEscapeKey(keyCode: 53, modifiers: [.shift]))
        #expect(!EscKeyMonitor.isBareEscapeKey(keyCode: 36, modifiers: []))
    }

    @Test("AUDIT-24: UpdateController disables Sparkle under XCTest and release-video capture")
    func audit24UpdateControllerTestAndCaptureGuards() {
        #expect(!UpdateController.shouldEnableUpdater(
            arguments: ["MLXChat"],
            environment: ["XCTestConfigurationFilePath": "/tmp/test.xctestconfiguration"]))
        #expect(!UpdateController.shouldEnableUpdater(
            arguments: ["MLXChat", "--capture-release-video"],
            environment: [:]))
        #expect(UpdateController.shouldEnableUpdater(
            arguments: ["MLXChat"],
            environment: [:]))
    }

    @MainActor
    @Test("AUDIT-25: ConversationStore.delete cleans greetingDerivedTitleIDs")
    func audit25ConversationDeleteCleansGreetingDerivedIDs() throws {
        let fixture = try makeAuditConversationStore()
        let store = fixture.store
        let conversation = store.newConversation()
        store.appendMessage(role: "user", content: "hi", to: conversation)
        #expect(store.isGreetingDerivedTitle(conversation.id))

        let deletedID = conversation.id
        store.delete(conversation)
        #expect(!store.isGreetingDerivedTitle(deletedID))
        withExtendedLifetime(fixture) {}
    }
}

// MARK: - Models, Downloads & Import Audit Fixes (AUDIT-05..AUDIT-11, AUDIT-14, AUDIT-22)

@Suite(.serialized)
struct AuditModelAndDownloadTests {

    @MainActor
    @Test("AUDIT-05: ModelStore.downloadAndWait awaits streamedInstallTasks and reports streamedInstallErrors")
    func audit05DownloadAndWaitAwaitsStreamedInstall() async throws {
        let store = ModelStore()
        guard let variant = StreamedMoEInstall.variants.first(where: { !store.isAvailable($0.modelID) }) else {
            // Fallback: unregister in memory isn't safe if real models exist, so assert error propagation
            // on a non-existent id if all variants happen to be installed.
            Issue.record("Expected at least one uninstalled StreamedMoE variant for AUDIT-05")
            return
        }
        let finishedInstallTask = LockedBox(false)
        store.streamedInstallHandler = { _, _, _, _, _ in
            try? await Task.sleep(for: .milliseconds(30))
            finishedInstallTask.withValue { $0 = true }
            throw StreamedModelInstallerError.helperFailed("streamed install audit failure")
        }

        do {
            try await store.downloadAndWait(variant.modelID)
            Issue.record("Expected downloadAndWait to throw streamed install failure")
        } catch let EngineError.generationFailed(message) {
            #expect(finishedInstallTask.read())
            #expect(message == "streamed install audit failure")
        }
    }

    @MainActor
    @Test("AUDIT-06: Download tokens guard stale callbacks, restarts await dying tasks, storage stats supersede")
    func audit06DownloadTokensDyingTaskAwaitAndStorageRevision() async throws {
        let store = ModelStore()
        let modelID = "audit-org/audit-06-restart"

        let events = LockedBox<[String]>([])
        let callCount = LockedBox(0)
        let firstStarted = AsyncGate()
        let firstExited = AsyncGate()
        let secondProceed = AsyncGate()
        let secondStarted = AsyncGate()

        store.mlxDownloadHandler = { _, _, _, onProgress in
            let attempt = callCount.withValue { count -> Int in
                count += 1
                return count
            }
            if attempt == 1 {
                events.withValue { $0.append("start-1") }
                await firstStarted.open()
                try? await Task.sleep(for: .milliseconds(60))
                // Emit late progress AFTER attempt 1 was cancelled and attempt 2 started.
                onProgress(0.99)
                events.withValue { $0.append("end-1") }
                await firstExited.open()
                throw CancellationError()
            } else {
                events.withValue { $0.append("start-2") }
                onProgress(0.25)
                await secondStarted.open()
                try? await secondProceed.wait()
                events.withValue { $0.append("end-2") }
                throw CancellationError()
            }
        }

        store.download(modelID)
        try await firstStarted.wait()
        store.cancelDownload(modelID)
        store.download(modelID)

        try await secondStarted.wait()
        try? await Task.sleep(for: .milliseconds(20))

        // Attempt 2 must not start until attempt 1 has completely exited (no .partial race).
        let recorded = events.read()
        let end1 = try #require(recorded.firstIndex(of: "end-1"))
        let start2 = try #require(recorded.firstIndex(of: "start-2"))
        #expect(end1 < start2)

        // Late progress / cleanup from attempt 1 must NOT overwrite attempt 2's progress or state.
        #expect(store.isDownloading(modelID))
        #expect(store.downloadProgress[modelID] == 0.25)

        await secondProceed.open()
        store.cancelDownload(modelID)

        // C-B15: Out-of-order storage stat refreshes cannot overwrite newer stats with older ones.
        let slowGate = AsyncGate()
        let fastDone = AsyncGate()
        store.refreshStorageStats {
            try? await slowGate.wait()
            return (used: 111, available: 111, total: 111)
        }
        store.refreshStorageStats {
            await fastDone.open()
            return (used: 222, available: 333, total: 444)
        }
        try await fastDone.wait()
        try? await Task.sleep(for: .milliseconds(25))
        #expect(store.usedBytes == 222)
        await slowGate.open()
        try? await Task.sleep(for: .milliseconds(25))
        #expect(store.usedBytes == 222)
        #expect(store.availableBytes == 333)
        #expect(store.totalBytes == 444)
    }

    @MainActor
    @Test("AUDIT-07: delete cancels active MLX download and removeImported keeps entry when Trash fails")
    func audit07DeleteCancelsActiveDownloadAndRemoveImportedPreservesOnFailure() async throws {
        try await withRestoredUserDefaults(keys: [Keys.customModels]) {
            let store = ModelStore()
            let modelID = "audit-org/audit-07-imported"
            let model = ModelCatalog.importedModel(repoID: modelID)
            store.registerCustomModelForTesting(model)

            let downloadStarted = AsyncGate()
            let downloadCancelledGate = AsyncGate()
            let downloadCancelled = LockedBox(false)
            store.mlxDownloadHandler = { _, _, _, _ in
                await downloadStarted.open()
                do {
                    try await Task.sleep(for: .seconds(5))
                } catch is CancellationError {
                    downloadCancelled.withValue { $0 = true }
                    await downloadCancelledGate.open()
                    throw CancellationError()
                }
            }

            store.download(modelID)
            try await downloadStarted.wait()
            #expect(store.isDownloading(modelID))

            // Deleting must cancel the active MLX download first.
            let deleteErr = store.delete(modelID)
            #expect(deleteErr == nil)
            #expect(!store.isDownloading(modelID))
            try await downloadCancelledGate.wait()
            #expect(downloadCancelled.read())

            // removeImported keeps entry in customModels when deleteAction fails.
            let failure = store.removeImported(modelID) { _ in "Permission denied" }
            #expect(failure == "Permission denied")
            #expect(store.customModels.contains { $0.id == modelID })

            // removeImported removes entry when deleteAction succeeds (including missing folder).
            let success = store.removeImported(modelID) { _ in nil }
            #expect(success == nil)
            #expect(!store.customModels.contains { $0.id == modelID })
        }
    }

    @MainActor
    @Test("AUDIT-08: Import validation enforces ASCII, blocks traversal, checks duplicates case-insensitively, safe URLs")
    func audit08ImportValidationDuplicatesAndSafeURLs() async throws {
        try await withRestoredUserDefaults(keys: [Keys.customModels]) {
            // ImportModelSheet button gating for empty normalized repo.
            #expect(!ImportModelSheet.canSubmit(input: "   ", importing: false))
            #expect(!ImportModelSheet.canSubmit(input: "https://huggingface.co/", importing: false))
            #expect(!ImportModelSheet.canSubmit(input: "org/repo", importing: true))
            #expect(ImportModelSheet.canSubmit(input: "org/repo", importing: false))

            // ASCII-only and traversal rejection.
            #expect(ModelStore.validateImportRepoID("  org-1/Model_2.0 \n") == "org-1/Model_2.0")
            #expect(ModelStore.validateImportRepoID("org/mödél") == nil)
            #expect(ModelStore.validateImportRepoID("../evil") == nil)
            #expect(ModelStore.validateImportRepoID("org/..") == nil)
            #expect(ModelStore.validateImportRepoID("./org") == nil)
            #expect(ModelStore.validateImportRepoID("org/.") == nil)
            #expect(ModelStore.validateImportRepoID("org/repo/extra") == nil)

            // plainDirectory rejects traversal segments.
            #expect(ModelStore.plainDirectory(for: "../escape") == nil)
            #expect(ModelStore.plainDirectory(for: "org/..") == nil)
            #expect(ModelStore.plainDirectory(for: "./foo") == nil)
            #expect(ModelStore.plainDirectory(for: "org/.") == nil)
            #expect(ModelStore.plainDirectory(for: "org//repo") == nil)

            // ModelCatalog.hugginFaceURL never crashes even on spaces/unusual characters.
            let weirdModel = ModelCatalog.importedModel(repoID: "org/name with spaces")
            #expect(weirdModel.hugginFaceURL.absoluteString.contains("name%20with%20spaces"))

            // Case-insensitive duplicate check before network fetch.
            let store = ModelStore()
            let fetchCalls = LockedBox(0)
            store.importFetchHandler = { url in
                fetchCalls.withValue { $0 += 1 }
                let response = HTTPURLResponse(
                    url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!
                return (Data(#"{"model_type":"qwen3"}"#.utf8), response)
            }
            let dupBefore = await store.importModel(repoID: "MLX-COMMUNITY/QWEN3-0.6B-4BIT")
            #expect(dupBefore == "This model is already in the list.")
            #expect(fetchCalls.read() == 0)

            // Case-insensitive duplicate re-check AFTER await.
            let racedID = "AuditOrg/RacedModel-4bit"
            let injected = LockedBox(false)
            store.importFetchHandler = { url in
                if !injected.read() {
                    injected.withValue { $0 = true }
                    await MainActor.run {
                        store.registerCustomModelForTesting(
                            ModelCatalog.importedModel(repoID: racedID.lowercased()))
                    }
                }
                let response = HTTPURLResponse(
                    url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!
                return (Data(#"{"model_type":"qwen3"}"#.utf8), response)
            }
            let dupAfter = await store.importModel(repoID: racedID)
            #expect(dupAfter == "This model is already in the list.")
        }
    }

    @Test("AUDIT-09: SnapshotDownloader TaskBox and StreamedModelInstaller cancel cleanly without races")
    func audit09DownloaderAndInstallerCancellation() async throws {
        // SnapshotDownloader.TaskBox: cancel before start never calls resume or makePoller.
        let box = SnapshotDownloader.TaskBox()
        box.cancel()
        var resumeCalled = false
        var pollerCalled = false
        let started = box.start(
            task: nil,
            makePoller: {
                pollerCalled = true
                return nil
            },
            resume: {
                resumeCalled = true
            })
        #expect(!started)
        #expect(!resumeCalled)
        #expect(!pollerCalled)

        // SnapshotDownloader.mapDownloadFailure maps URLError.cancelled and box cancellation to CancellationError.
        let mappedURLCancel = SnapshotDownloader.mapDownloadFailure(
            URLError(.cancelled), filename: "model.safetensors", isCancelled: false)
        #expect(mappedURLCancel is CancellationError)

        let mappedFlagCancel = SnapshotDownloader.mapDownloadFailure(
            AuditTestError(message: "transport error"), filename: "model.safetensors", isCancelled: true)
        #expect(mappedFlagCancel is CancellationError)

        let mappedOther = SnapshotDownloader.mapDownloadFailure(
            AuditTestError(message: "disk full"), filename: "model.safetensors", isCancelled: false)
        #expect(mappedOther is SnapshotDownloader.DownloadError)

        // StreamedModelInstaller: cancel before execute never launches the subprocess.
        let preCancelled = StreamedModelInstaller(helperURL: URL(fileURLWithPath: "/bin/sleep"))
        preCancelled.cancel()
        await #expect(throws: CancellationError.self) {
            try await preCancelled.execute(
                helperURL: URL(fileURLWithPath: "/bin/sleep"),
                arguments: ["5"])
        }

        // StreamedModelInstaller: in-flight cancel throws CancellationError (not helperFailed).
        let activeInstaller = StreamedModelInstaller(helperURL: URL(fileURLWithPath: "/bin/sleep"))
        let runTask = Task {
            try await activeInstaller.execute(
                helperURL: URL(fileURLWithPath: "/bin/sleep"),
                arguments: ["5"])
        }
        try? await Task.sleep(for: .milliseconds(60))
        activeInstaller.cancel()
        await #expect(throws: CancellationError.self) {
            try await runTask.value
        }

        // StreamedModelInstaller: launch failure throws cleanly without hanging.
        let missingInstaller = StreamedModelInstaller(helperURL: URL(fileURLWithPath: "/nonexistent/helper-bin"))
        do {
            try await missingInstaller.execute(
                helperURL: URL(fileURLWithPath: "/nonexistent/helper-bin"),
                arguments: [])
            Issue.record("Expected launch failure for missing helper binary")
        } catch is CancellationError {
            Issue.record("Launch failure should not map to CancellationError")
        } catch {
            // Expected launch error
        }
    }

    @Test("AUDIT-10: StreamedMoEEngine.mapLoadError preserves EngineError and CancellationError unwrapped")
    func audit10StreamedMoEMapLoadErrorUnwrapped() {
        let unavailable = StreamedMoEEngine.mapLoadError(EngineError.unavailable("Missing manifest"))
        if case EngineError.unavailable(let message) = unavailable {
            #expect(message == "Missing manifest")
        } else {
            Issue.record("Expected EngineError.unavailable unwrapped, got \(unavailable)")
        }

        let cancelled = StreamedMoEEngine.mapLoadError(CancellationError())
        #expect(cancelled is CancellationError)

        let wrapped = StreamedMoEEngine.mapLoadError(AuditTestError(message: "boom"))
        if case EngineError.generationFailed(let message) = wrapped {
            #expect(message.contains("Couldn’t load the SSD-streamed model"))
        } else {
            Issue.record("Expected EngineError.generationFailed for arbitrary error")
        }
    }

    @Test("AUDIT-11: SnapshotDownloader blocks traversal, encodes URLs, replaces destination, propagates SHA read errors")
    func audit11SnapshotDownloaderSafetyAndAtomicity() throws {
        let temp = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        // Traversal rejection in localPath and safeStagedFileURL.
        #expect(SnapshotDownloader.localPath(for: "../evil.json", subfolder: nil) == nil)
        #expect(SnapshotDownloader.localPath(for: "sub/../../evil.json", subfolder: nil) == nil)
        #expect(SnapshotDownloader.localPath(for: "/abs/evil.json", subfolder: nil) == nil)
        #expect(SnapshotDownloader.safeStagedFileURL(staging: temp, localPath: "../escape.json") == nil)
        #expect(SnapshotDownloader.safeStagedFileURL(staging: temp, localPath: "config.json") != nil)

        // Percent-encoding of special characters in remotePath.
        let resolved = try #require(
            SnapshotDownloader.resolveFileURL(
                repoID: "org/repo",
                remotePath: "dir with spaces/model #1.json"))
        #expect(resolved.absoluteString.contains("dir%20with%20spaces/model%20%231.json"))

        // Atomic replacement of an existing destination directory.
        let destination = temp.appendingPathComponent("dest", isDirectory: true)
        let staging = temp.appendingPathComponent("dest.partial", isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        try Data("old".utf8).write(to: destination.appendingPathComponent("old.txt"))
        try Data("new".utf8).write(to: staging.appendingPathComponent("new.txt"))

        try SnapshotDownloader.publishStagedDirectory(staging: staging, to: destination)
        #expect(FileManager.default.fileExists(atPath: destination.appendingPathComponent("new.txt").path))
        #expect(!FileManager.default.fileExists(atPath: destination.appendingPathComponent("old.txt").path))
        #expect(!FileManager.default.fileExists(atPath: staging.path))

        // sha256Hex propagates I/O read errors instead of hashing a truncated prefix.
        var reads = 0
        #expect(throws: AuditTestError.self) {
            _ = try SnapshotDownloader.sha256Hex {
                reads += 1
                if reads == 1 { return Data("partial".utf8) }
                throw AuditTestError(message: "simulated read failure")
            }
        }
    }

    @MainActor
    @Test("AUDIT-14: External refresh clears unsupportedReasonCache and bumps streamedStorageRevision")
    func audit14CacheInvalidationAndStreamedStorageRevision() async throws {
        let store = ModelStore()
        _ = store.unsupportedReason(for: "audit-org/cache-probe")
        #expect(store.cachedUnsupportedReasonCount > 0)

        let revisionBefore = store.streamedStorageRevision
        store.noteExternalModelDirectoryMayHaveChanged()
        #expect(store.cachedUnsupportedReasonCount == 0)
        #expect(store.streamedStorageRevision == revisionBefore + 1)

        if let variant = StreamedMoEInstall.variants.first(where: { !store.isAvailable($0.modelID) }) {
            let revBeforeInstall = store.streamedStorageRevision
            store.streamedInstallHandler = { _, _, _, _, _ in
                throw StreamedModelInstallerError.helperFailed("expected test exit")
            }
            try? await store.downloadAndWait(variant.modelID)
            #expect(store.streamedStorageRevision > revBeforeInstall)
        }
    }

    @Test("AUDIT-22: regularFileHasPositiveSize resolves symlinks and rejects directories and empty targets")
    func audit22RegularFileHasPositiveSizeChecksSymlinksAndDirectories() throws {
        let temp = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }
        let fm = FileManager.default

        let dirNamedWeights = temp.appendingPathComponent("dir.safetensors", isDirectory: true)
        try fm.createDirectory(at: dirNamedWeights, withIntermediateDirectories: true)
        #expect(!ModelStore.regularFileHasPositiveSize(dirNamedWeights))

        let emptyFile = temp.appendingPathComponent("empty.safetensors")
        try Data().write(to: emptyFile)
        #expect(!ModelStore.regularFileHasPositiveSize(emptyFile))

        let validFile = temp.appendingPathComponent("blob.bin")
        try Data([1, 2, 3]).write(to: validFile)
        #expect(ModelStore.regularFileHasPositiveSize(validFile))

        let symlinkToValid = temp.appendingPathComponent("link-valid.safetensors")
        try fm.createSymbolicLink(at: symlinkToValid, withDestinationURL: validFile)
        #expect(ModelStore.regularFileHasPositiveSize(symlinkToValid))

        let symlinkToEmpty = temp.appendingPathComponent("link-empty.safetensors")
        try fm.createSymbolicLink(at: symlinkToEmpty, withDestinationURL: emptyFile)
        #expect(!ModelStore.regularFileHasPositiveSize(symlinkToEmpty))

        let symlinkToDir = temp.appendingPathComponent("link-dir.safetensors")
        try fm.createSymbolicLink(at: symlinkToDir, withDestinationURL: dirNamedWeights)
        #expect(!ModelStore.regularFileHasPositiveSize(symlinkToDir))

        let brokenSymlink = temp.appendingPathComponent("link-broken.safetensors")
        try fm.createSymbolicLink(
            at: brokenSymlink,
            withDestinationURL: temp.appendingPathComponent("does-not-exist"))
        #expect(!ModelStore.regularFileHasPositiveSize(brokenSymlink))
    }
}

// MARK: - Support, Views, Rendering & Engine Stream Audit Fixes (AUDIT-12..13, 15..21, 26..31)

@Suite struct AuditSupportRenderingAndEngineTests {

    @Test("AUDIT-12: FileTextExtractor propagates read errors and rejects whitespace-only PDFs")
    func audit12FileTextExtractorReadErrorsAndWhitespacePDF() throws {
        let temp = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        // Read error in chunk loop throws ExtractError.unreadable instead of silent truncation.
        #expect(throws: FileTextExtractor.ExtractError.unreadable) {
            _ = try FileTextExtractor.readBoundedUTF8(
                readChunk: { throw POSIXError(.EIO) },
                lossyFallback: { "" })
        }

        // A directory with a .txt extension triggers EISDIR on handle.read and throws unreadable.
        let dirWithTxtExt = temp.appendingPathComponent("folder.txt", isDirectory: true)
        try FileManager.default.createDirectory(at: dirWithTxtExt, withIntermediateDirectories: true)
        #expect(throws: FileTextExtractor.ExtractError.unreadable) {
            _ = try FileTextExtractor.extract(url: dirWithTxtExt)
        }

        // Whitespace-only PDF throws ExtractError.emptyPDF.
        let pdfURL = temp.appendingPathComponent("blank.pdf")
        try makePDF(text: "   \n\t  ").write(to: pdfURL)
        #expect(throws: FileTextExtractor.ExtractError.emptyPDF) {
            _ = try FileTextExtractor.extract(url: pdfURL)
        }
    }

    @Test("AUDIT-13: EditPromptDraft stages temperature and writes UserDefaults only on Save")
    func audit13EditPromptDraftStagesTemperatureUntilSave() throws {
        let suiteName = "org.mlxchat.audit13.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        defaults.set(0.4, forKey: Keys.personalizationTemperature)
        var draft = EditPromptDraft(preset: .general, defaults: defaults)
        #expect(draft.temperature == 0.4)

        // Mutate temperature (Cancel path): UserDefaults must stay unchanged.
        draft.temperature = 1.0
        #expect(defaults.double(forKey: Keys.personalizationTemperature) == 0.4)

        // Save path: persists staged temperature.
        draft.save(defaults: defaults)
        #expect(defaults.double(forKey: Keys.personalizationTemperature) == 1.0)
    }

    @Test("AUDIT-15: CursorPushTracker balances NSCursor push and pop across hover, disable, and disappear")
    func audit15CursorPushTrackerBalancesPushAndPop() {
        var tracker = CursorPushTracker()
        var pushes = 0
        var pops = 0
        let push = { pushes += 1 }
        let pop = { pops += 1 }

        // Exit without prior push must not pop.
        tracker.setHovering(false, enabled: true, push: push, pop: pop)
        #expect(pushes == 0 && pops == 0)

        // Hover while disabled must not push; exit must not pop.
        tracker.setHovering(true, enabled: false, push: push, pop: pop)
        tracker.setHovering(false, enabled: false, push: push, pop: pop)
        #expect(pushes == 0 && pops == 0)

        // Hover while enabled pushes once; repeated enter is idempotent; disabling pops once.
        tracker.setHovering(true, enabled: true, push: push, pop: pop)
        tracker.setHovering(true, enabled: true, push: push, pop: pop)
        #expect(pushes == 1 && pops == 0)
        tracker.setHovering(true, enabled: false, push: push, pop: pop)
        #expect(pushes == 1 && pops == 1)
        tracker.setHovering(false, enabled: false, push: push, pop: pop)
        #expect(pushes == 1 && pops == 1)

        // Disappear while hovered pops once and stays balanced.
        tracker.setHovering(true, enabled: true, push: push, pop: pop)
        #expect(pushes == 2 && pops == 1)
        tracker.popIfNeeded(pop: pop)
        tracker.popIfNeeded(pop: pop)
        #expect(pushes == 2 && pops == 2)
    }

    @Test("AUDIT-16: Studio.cascadedNSFont deduplicates fallback descriptors by PostScript name")
    func audit16CascadedFontDeduplicatesByPostScriptName() {
        let font = Studio.cascadedNSFont(base: NSFont.systemFont(ofSize: 14), size: 14)
        let descriptors = font.fontDescriptor.object(forKey: .cascadeList) as? [NSFontDescriptor] ?? []
        let names = descriptors.compactMap(\.postscriptName)
        #expect(!names.isEmpty)
        #expect(names.count == Set(names).count)
    }

    @Test("AUDIT-17: Inkling2Processor rejects empty and infinite CIImages without crashing")
    func audit17Inkling2VisionGuardsEmptyAndInfiniteImages() {
        let processor = Inkling2Processor(
            Inkling2ProcessorConfiguration(),
            tokenizer: MinimalAuditTokenizer())

        #expect(throws: VLMError.self) {
            _ = try processor.preprocess(
                image: CIImage.empty(),
                processing: nil)
        }
        #expect(throws: VLMError.self) {
            _ = try processor.preprocess(
                image: CIImage(color: .red),
                processing: nil)
        }
        #expect(throws: VLMError.self) {
            _ = try processor.renderRGBA(CIImage.empty())
        }
        #expect(throws: VLMError.self) {
            _ = try processor.renderRGBA(CIImage(color: .red))
        }
    }

    @Test("AUDIT-18: StreamedMoEEngine.withTemporaryImageFile always cleans up temp file and checks cancellation")
    func audit18StreamedMoETempImageCleanupAndCancellation() async throws {
        let payload = Data([0xFF, 0xD8, 0xFF, 0xD9])
        var capturedURL: URL?

        let result = try StreamedMoEEngine.withTemporaryImageFile(payload) { url -> Int in
            capturedURL = url
            #expect(FileManager.default.fileExists(atPath: url.path))
            return try Data(contentsOf: url).count
        }
        #expect(result == payload.count)
        let normalURL = try #require(capturedURL)
        #expect(!FileManager.default.fileExists(atPath: normalURL.path))

        // Cleanup also runs when body throws.
        var throwingURL: URL?
        #expect(throws: AuditTestError.self) {
            _ = try StreamedMoEEngine.withTemporaryImageFile(payload) { url -> Int in
                throwingURL = url
                #expect(FileManager.default.fileExists(atPath: url.path))
                throw AuditTestError(message: "encode failed")
            }
        }
        let failedURL = try #require(throwingURL)
        #expect(!FileManager.default.fileExists(atPath: failedURL.path))

        // Cancelled task fails immediately with CancellationError.
        let cancelTask = Task<Void, Error> {
            withUnsafeCurrentTask { $0?.cancel() }
            _ = try StreamedMoEEngine.withTemporaryImageFile(payload) { _ in 0 }
        }
        await #expect(throws: CancellationError.self) {
            try await cancelTask.value
        }
    }

    @Test("AUDIT-19: AttachmentImage composites transparent PNGs over an opaque white sRGB matte")
    func audit19AttachmentImageWhiteMatteForTransparency() throws {
        let rep = try #require(NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: 16,
            pixelsHigh: 16,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0))
        // Leave all pixels fully transparent (0, 0, 0, 0).
        if let bytes = rep.bitmapData {
            memset(bytes, 0, rep.bytesPerRow * rep.pixelsHigh)
        }
        let transparent = NSImage(size: NSSize(width: 16, height: 16))
        transparent.addRepresentation(rep)

        let jpeg = try #require(AttachmentImage.jpegData(from: transparent, maxDimension: 64, quality: 0.95))
        let decoded = try #require(NSBitmapImageRep(data: jpeg))
        let color = try #require(decoded.colorAt(x: 8, y: 8)?.usingColorSpace(.sRGB))
        #expect(color.redComponent > 0.95)
        #expect(color.greenComponent > 0.95)
        #expect(color.blueComponent > 0.95)
    }

    @MainActor
    @Test("AUDIT-20: WebHTMLView.Coordinator fails closed when rules fail and tears down on dismantle")
    func audit20WebHTMLViewFailClosedAndDismantle() {
        let webView = WKWebView(frame: .zero)
        let failedCoordinator = WebHTMLView.Coordinator()
        failedCoordinator.enqueueHTML("<p>untrusted</p>", in: webView)
        failedCoordinator.applyContentRulesResult(installed: false)
        failedCoordinator.loadPendingHTML(in: webView)
        #expect(!failedCoordinator.canLoadPendingHTML)
        #expect(failedCoordinator.lastHTML == nil)

        let okCoordinator = WebHTMLView.Coordinator()
        webView.navigationDelegate = okCoordinator
        okCoordinator.applyContentRulesResult(installed: true)
        okCoordinator.enqueueHTML("<p>safe</p>", in: webView)
        #expect(okCoordinator.lastHTML == "<p>safe</p>")

        WebHTMLView.dismantleNSView(webView, coordinator: okCoordinator)
        #expect(okCoordinator.isDismantled)
        #expect(webView.navigationDelegate == nil)
        okCoordinator.enqueueHTML("<p>after dismantle</p>", in: webView)
        #expect(okCoordinator.lastHTML == "<p>safe</p>")
    }

    @Test("AUDIT-21: MathInlineProvider returns placeholder for non-swiftmath URLs and swatexColor handles non-RGB")
    func audit21MathInlineFallbackAndSwatexColorSafety() async throws {
        let provider = MathInlineProvider(fontSize: 14, color: .labelColor)
        let nonMathURL = try #require(URL(string: "https://example.com/diagram.png"))
        _ = try await provider.image(with: nonMathURL, label: "diagram")

        let patternColor = NSColor(patternImage: NSImage(size: NSSize(width: 2, height: 2)))
        let ink = MathRenderer.swatexColor(from: patternColor)
        #expect(ink.r == 0 && ink.g == 0 && ink.b == 0 && ink.a == 1)
    }

    @Test("AUDIT-26: ArtifactNavigationPolicy allows same-document fragment links inside artifacts")
    func audit26ArtifactNavigationPolicySameDocumentFragments() {
        #expect(ArtifactNavigationPolicy.decision(
            for: URL(string: "about:blank#section-2"),
            isLinkActivated: true) == .allowInWebView)
        #expect(ArtifactNavigationPolicy.decision(
            for: URL(string: "#top"),
            isLinkActivated: true) == .allowInWebView)
        #expect(ArtifactNavigationPolicy.decision(
            for: URL(string: "about:blank"),
            isLinkActivated: true) == .cancel)
    }

    @Test("AUDIT-27: ChatCodeBlockView.displayLanguage falls back to 'code' for blank languages")
    func audit27CodeBlockDisplayLanguageBlankFallback() {
        #expect(ChatCodeBlockView.displayLanguage(for: nil) == "code")
        #expect(ChatCodeBlockView.displayLanguage(for: "") == "code")
        #expect(ChatCodeBlockView.displayLanguage(for: "  \t\n ") == "code")
        #expect(ChatCodeBlockView.displayLanguage(for: " Swift ") == "swift")
    }

    @Test("AUDIT-28: ThinkTagFilter.flushPartialOnCancel flushes safe carry without leaking markers or reclassifying")
    func audit28ThinkTagFilterFlushPartialOnCancel() {
        // Safe carry inside thinking stays thinking (never reclassified).
        var filter1 = ThinkTagFilter(open: "<think>", close: "</think>", startsInside: true, mayEmit: false)
        _ = filter1.consume("step one")
        let flushed1 = filter1.flushPartialOnCancel()
        #expect(flushed1.thinking == "step one")
        #expect(flushed1.content.isEmpty)

        // Trailing partial close marker is stripped; safe prefix is emitted.
        var filter2 = ThinkTagFilter(open: "<think>", close: "</think>", startsInside: true, mayEmit: false)
        let first2 = filter2.consume("reasoning done</thi")
        let flushed2 = filter2.flushPartialOnCancel()
        #expect(first2.thinking + flushed2.thinking == "reasoning done")
        #expect(flushed2.content.isEmpty)

        // Partial open marker in pending mode is dropped cleanly.
        var filter3 = ThinkTagFilter(open: "<think>", close: "</think>", startsInside: false, mayEmit: true)
        _ = filter3.consume("<thi")
        let flushed3 = filter3.flushPartialOnCancel()
        #expect(flushed3.thinking.isEmpty)
        #expect(flushed3.content.isEmpty)

        // Routers emit .thinkingDelta only and never .thinkingReclassified on cancel.
        var router = ThinkingEventRouter(
            filter: ThinkTagFilter(open: "<think>", close: "</think>", startsInside: true, mayEmit: false))
        var events: [EngineEvent] = []
        router.consume("halfway</th") { events.append($0) }
        router.flushOnCancel { events.append($0) }
        var combinedThinking = ""
        for event in events {
            switch event {
            case .thinkingDelta(let text): combinedThinking += text
            case .thinkingReclassified: Issue.record("Cancel flush must never reclassify thinking")
            case .delta(let text): Issue.record("Unexpected content delta on cancel: \(text)")
            case .prefill, .completed: break
            }
        }
        #expect(combinedThinking == "halfway")
    }

    @Test("AUDIT-29: AppleIntelligenceEngine.snapshotDelta preserves combining scalars and ZWJ emoji")
    func audit29AppleIntelligenceSnapshotDeltaUnicodeScalars() {
        // Combining acute accent added after base 'e'.
        var offset = 0
        var assembled = ""
        for snapshot in ["e", "e\u{0301}", "e\u{0301}!"] {
            let step = AppleIntelligenceEngine.snapshotDelta(previousUTF8Offset: offset, current: snapshot)
            assembled += step.delta
            offset = step.nextUTF8Offset
        }
        #expect(assembled == "e\u{0301}!")

        // ZWJ emoji sequence built incrementally across cumulative snapshots.
        offset = 0
        assembled = ""
        for snapshot in ["👩", "👩\u{200D}", "👩\u{200D}💻", "👩\u{200D}💻 ok"] {
            let step = AppleIntelligenceEngine.snapshotDelta(previousUTF8Offset: offset, current: snapshot)
            assembled += step.delta
            offset = step.nextUTF8Offset
        }
        #expect(assembled == "👩\u{200D}💻 ok")
    }

    @Test("AUDIT-30: PromptBuilder.trimmed drops leading non-user messages while keeping the final message")
    func audit30PromptBuilderDropsLeadingNonUserMessages() {
        let chunk = String(repeating: "abcd", count: 500)
        // Uneven history where pair trimming would leave a leading assistant message.
        let overBudget: [EngineMessage] = [
            EngineMessage(role: .user, text: chunk),
            EngineMessage(role: .user, text: chunk),
            EngineMessage(role: .assistant, text: chunk),
            EngineMessage(role: .user, text: "keep user"),
            EngineMessage(role: .assistant, text: "keep assistant"),
            EngineMessage(role: .user, text: "final question"),
        ]
        let trimmed = PromptBuilder.trimmed(history: overBudget, ctxTokens: 2_000)
        #expect(trimmed.first?.role == .user)
        #expect(trimmed.last?.text == "final question")

        // Leading assistant under budget is also dropped when a trailing user exists.
        let underBudget: [EngineMessage] = [
            EngineMessage(role: .assistant, text: "orphan"),
            EngineMessage(role: .user, text: "hello"),
        ]
        let normalized = PromptBuilder.trimmed(history: underBudget, ctxTokens: 8_192)
        #expect(normalized.count == 1)
        #expect(normalized[0].role == .user)
        #expect(normalized[0].text == "hello")

        // A single message is never dropped.
        let single = [EngineMessage(role: .assistant, text: "only")]
        #expect(PromptBuilder.trimmed(history: single, ctxTokens: 8_192).count == 1)
    }

    @Test("AUDIT-31: MessageSegmenter requires '# ' space for hash comments so C preprocessor directives stay intact")
    func audit31MessageSegmenterHashSpaceRule() throws {
        let cBody = (["#include <stdio.h>", "#define MAX 10"] + (1...18).map { "int x\($0) = \($0);" })
            .joined(separator: "\n")
        let cBlocks = MessageSegmenter.segment("```c\n\(cBody)\n```")
        guard case .artifact(let cArtifact) = cBlocks.first else {
            Issue.record("Expected C code artifact")
            return
        }
        #expect(cArtifact.title == "C code")

        let pyBody = (["# Compute Fibonacci"] + (1...18).map { "x\($0) = \($0)" })
            .joined(separator: "\n")
        let pyBlocks = MessageSegmenter.segment("```python\n\(pyBody)\n```")
        guard case .artifact(let pyArtifact) = pyBlocks.first else {
            Issue.record("Expected Python code artifact")
            return
        }
        #expect(pyArtifact.title == "Compute Fibonacci")

        let shBody = (["#!/usr/bin/env bash"] + (1...18).map { "echo \($0)" })
            .joined(separator: "\n")
        let shBlocks = MessageSegmenter.segment("```bash\n\(shBody)\n```")
        guard case .artifact(let shArtifact) = shBlocks.first else {
            Issue.record("Expected shell script artifact")
            return
        }
        #expect(shArtifact.title == "/usr/bin/env bash")
    }
}
