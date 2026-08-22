import Foundation
import SwiftData
import Testing
@testable import MLXChat

@Suite struct ChatHistoryTests {
    @MainActor
    @Test func modelSwitchStartsASeparateTranscriptSection() {
        let firstUser = ChatMessage(role: "user", content: "First", sortIndex: 0)
        let appleAnswer = ChatMessage(
            role: "assistant", content: "```json\n{}\n```", sortIndex: 1,
            modelID: appleIntelligenceEngineID)
        let pendingUser = ChatMessage(role: "user", content: "Follow-up", sortIndex: 2)

        let history = ChatController.historyMessages(
            [firstUser, appleAnswer, pendingUser],
            for: "mlx-community/Ornith-1.0-9B-4bit")

        #expect(history.map(\.content) == ["Follow-up"])
    }

    @MainActor
    @Test func sameModelKeepsItsUninterruptedTranscript() {
        let user = ChatMessage(role: "user", content: "Question", sortIndex: 0)
        let answer = ChatMessage(
            role: "assistant", content: "Answer", sortIndex: 1,
            modelID: appleIntelligenceEngineID)
        let followUp = ChatMessage(role: "user", content: "Follow-up", sortIndex: 2)

        let history = ChatController.historyMessages(
            [user, answer, followUp], for: appleIntelligenceEngineID)

        #expect(history.map(\.content) == ["Question", "Answer", "Follow-up"])
    }
}

@Suite struct EngineStateTests {
    @Test func preparingCoversDownloadAndLoadOfTheSameModel() {
        let loading = EngineRuntime.EngineState.loading(id: "ornith-ai/Ornith-1.5-9B-MLX-4bit", fraction: 0.4)
        let downloading = EngineRuntime.EngineState.downloading(id: "ornith-ai/Ornith-1.5-9B-MLX-4bit")

        #expect(loading.isPreparing("ornith-ai/Ornith-1.5-9B-MLX-4bit"))
        #expect(downloading.isPreparing("ornith-ai/Ornith-1.5-9B-MLX-4bit"))
        #expect(!loading.isPreparing("apple-intelligence"))
        #expect(!downloading.isPreparing("mlx-community/Qwen3.5-9B-4bit"))
    }

    @Test func idleAndGeneratingStatesAreNotPreparing() {
        #expect(!EngineRuntime.EngineState.empty.isPreparing("any"))
        #expect(!EngineRuntime.EngineState.ready(id: "any").isPreparing("any"))
        #expect(!EngineRuntime.EngineState.generating(id: "any").isPreparing("any"))
    }
}

private struct PersistFailure: Error, LocalizedError {
    var errorDescription: String? { "simulated persist failure" }
}

@MainActor
private func makeInMemoryStore() throws -> (container: ModelContainer, store: ConversationStore) {
    let schema = Schema([Conversation.self, ChatMessage.self])
    let configuration = ModelConfiguration(
        "persist-tests-\(UUID().uuidString)",
        schema: schema,
        isStoredInMemoryOnly: true)
    let container = try ModelContainer(for: schema, configurations: [configuration])
    return (container, ConversationStore(context: container.mainContext))
}

/// Serialized: SwiftData @Model inits bind to the test-host app container. These
/// tests never insert Conversation/ChatMessage; they only persist() via save().
@Suite(.serialized)
struct PersistenceTests {
    @MainActor
    @Test func throwingSaveExecutorReportsEachPersist() throws {
        let fixture = try makeInMemoryStore()
        let store = fixture.store
        store.saveExecutor = { throw PersistFailure() }
        var errors: [Error] = []
        store.onSaveError = { errors.append($0) }

        store.save()
        store.save()
        store.save()
        #expect(errors.count == 3)
    }

    @MainActor
    @Test func persistRetriesOnceThenReportsSuccess() throws {
        let fixture = try makeInMemoryStore()
        let store = fixture.store
        var attempts = 0
        store.saveExecutor = {
            attempts += 1
            if attempts == 1 { throw PersistFailure() }
        }
        var errors = 0
        var successes = 0
        store.onSaveError = { _ in errors += 1 }
        store.onSaveSuccess = { successes += 1 }

        store.save()
        #expect(attempts == 2)
        #expect(errors == 0)
        #expect(successes == 1)
    }

    @MainActor
    @Test func persistSuccessClearsOnlyHistorySaveBanner() throws {
        let fixture = try makeInMemoryStore()
        let store = fixture.store
        let chat = ChatController()
        chat.conversationStore = store
        chat.errorBanner = "Couldn't save chat history: simulated persist failure"
        store.save()
        #expect(chat.errorBanner == nil)

        chat.errorBanner = "Couldn't load that image."
        store.save()
        #expect(chat.errorBanner == "Couldn't load that image.")
    }

    @MainActor
    @Test func defaultExecutorPersistsInMemory() throws {
        let fixture = try makeInMemoryStore()
        let store = fixture.store
        var saveErrors = 0
        store.onSaveError = { _ in saveErrors += 1 }

        store.save()
        #expect(saveErrors == 0)
        withExtendedLifetime(fixture) {}
    }

    @MainActor
    @Test func assigningStoreSurfacesPersistErrorsOnTheBanner() throws {
        let fixture = try makeInMemoryStore()
        let store = fixture.store
        let chat = ChatController()
        chat.conversationStore = store
        #expect(chat.errorBanner == nil)

        store.onSaveError?(PersistFailure())
        #expect(chat.errorBanner == "Couldn't save chat history: simulated persist failure")
    }
}

@Suite struct EngineRuntimeLoadTests {
    @MainActor
    @Test func currentLoadFailureClearsTheSlot() {
        let runtime = EngineRuntime(modelStore: ModelStore())
        let x = FakeChatEngine(id: "model-x")
        runtime.installSlot(x)
        runtime.loadDidFail(x)
        #expect(runtime.state == .empty)
        #expect(runtime.loadedModelID == nil)
    }

    @MainActor
    @Test func staleLoadFailureLeavesTheSuccessorSlot() {
        let runtime = EngineRuntime(modelStore: ModelStore())
        let x = FakeChatEngine(id: "model-x")
        let y = FakeChatEngine(id: "model-y")
        runtime.installSlot(y)
        runtime.loadDidFail(x)
        #expect(runtime.state == .loading(id: "model-y", fraction: 0))
        #expect(runtime.loadedModelID == "model-y")
    }

    @MainActor
    @Test func staleSuccessDoesNotMarkTheSuccessorReadyForThePriorModel() {
        let runtime = EngineRuntime(modelStore: ModelStore())
        let x = FakeChatEngine(id: "model-x")
        let y = FakeChatEngine(id: "model-y")
        runtime.installSlot(y)
        #expect(throws: CancellationError.self) {
            try runtime.loadDidSucceed(x, id: "model-x")
        }
        #expect(runtime.state == .loading(id: "model-y", fraction: 0))
    }
}

private final class FakeChatEngine: ChatEngine {
    let id: String
    var isLoaded = false

    init(id: String) { self.id = id }

    func load(progress: @escaping @Sendable (Double) -> Void) async throws {}
    func unload() {}

    func stream(
        history: [EngineMessage],
        systemPrompt: String,
        temperature: Double,
        conversationID: UUID
    ) -> AsyncThrowingStream<EngineEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
}
