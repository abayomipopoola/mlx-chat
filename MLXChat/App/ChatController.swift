import Foundation
import Observation
import SwiftData

/// Orchestrates a chat turn: engine resolution (download + load), streaming,
/// think-phase routing, throttled UI flushes, errors, stop/regenerate.
@MainActor
@Observable
final class ChatController {
    let modelStore: ModelStore
    let runtime: EngineRuntime
    /// Wired by RootView once the SwiftData context exists.
    var conversationStore: ConversationStore?

    /// Selected engine/model id ("apple-intelligence" or an HF repo id).
    /// A persisted local selection whose files are gone (fresh install, moved
    /// streamed install, or model deleted outside the app) falls back to Apple
    /// Intelligence silently.
    var selectedModelID: String {
        didSet {
            UserDefaults.standard.set(selectedModelID, forKey: Keys.selectedModelID)
            guard oldValue != selectedModelID else { return }
            warmSwitchedModel()
        }
    }

    /// Transient "Switched to …" confirmation shown after a model switch loads.
    private(set) var switchToast: String?
    private var toastTask: Task<Void, Never>?

    private(set) var isStreaming = false
    private(set) var streamingMessageID: UUID?
    private(set) var isInThinkPhase = false
    var errorBanner: String?
    /// Conversation eligible for Retry after a failed (empty) turn.
    private var retryConversation: Conversation?

    private var generationTask: Task<Void, Never>?
    private var flushTask: Task<Void, Never>?
    private var activeMessage: ChatMessage?
    private var contentBuffer = ""
    private var thinkingBuffer = ""
    private var contentCharacterCount = 0
    private var thinkingCharacterCount = 0

    /// A cheap scroll trigger. ChatScreen observes this instead of counting the
    /// entire growing String on every SwiftUI body evaluation.
    private(set) var transcriptRevision = 0

    init() {
        let store = ModelStore()
        modelStore = store
        runtime = EngineRuntime(modelStore: store)
        let saved = UserDefaults.standard.string(forKey: Keys.selectedModelID)
            ?? Keys.Defaults.selectedModelID
        selectedModelID = saved == appleIntelligenceEngineID || store.isUsableLocalSelection(saved)
            ? saved : appleIntelligenceEngineID
    }

    var selectedModelDisplayName: String {
        if selectedModelID == appleIntelligenceEngineID { return "Apple Intelligence" }
        return modelStore.model(for: selectedModelID)?.pickerDisplayName ?? selectedModelID
    }

    /// Whether the header shows the thinking on/off bulb for the selected model.
    var selectedModelSupportsThinking: Bool {
        modelStore.model(for: selectedModelID)?.supportsThinkingToggle == true
    }

    /// Whether the composer shows the attachment control for the selected model.
    var selectedModelSupportsVision: Bool {
        modelStore.model(for: selectedModelID)?.supportsVision == true
    }

    var canRetry: Bool { retryConversation != nil }

    // MARK: - Sending

    func send(
        _ text: String,
        in conversation: Conversation,
        image: Data? = nil,
        attachment: (name: String, text: String)? = nil
    ) {
        guard !isStreaming, let store = conversationStore else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || image != nil || attachment != nil else { return }
        errorBanner = nil
        retryConversation = nil
        store.appendMessage(
            role: "user",
            content: trimmed,
            to: conversation,
            imageData: image,
            attachmentName: attachment?.name,
            attachmentText: attachment?.text)
        startAssistantTurn(in: conversation)
    }

    func retry() {
        guard let conversation = retryConversation else { return }
        errorBanner = nil
        retryConversation = nil
        startAssistantTurn(in: conversation)
    }

    /// Regenerate the LAST assistant message.
    func regenerate(_ message: ChatMessage, in conversation: Conversation) {
        guard !isStreaming, message.role == "assistant",
              conversation.orderedMessages.last?.id == message.id,
              let store = conversationStore
        else { return }
        errorBanner = nil
        store.deleteMessage(message)
        startAssistantTurn(in: conversation)
    }

    func stop() {
        generationTask?.cancel()
    }

    /// Warms the selected local model at launch when its files are already on disk,
    /// so the first message streams immediately. No-op for Apple Intelligence.
    func preloadSelectedModelIfAvailable() {
        let id = selectedModelID
        guard id != appleIntelligenceEngineID, modelStore.isAvailable(id) else { return }
        guard runtime.loadedModelID != id else { return }
        Task { [weak self] in
            _ = try? await self?.runtime.engine(for: id)
        }
    }

    /// Loads the newly selected model right away (instead of lazily on first
    /// send) so the picker can show progress, then confirms with a toast.
    private func warmSwitchedModel() {
        let id = selectedModelID
        let name = selectedModelDisplayName
        if id == appleIntelligenceEngineID {
            // Release any resident local weights right away — previously they
            // stayed in RAM until the idle auto-unload fired. Never yank
            // mid-stream: the in-flight generation still needs them.
            if !isStreaming { runtime.unloadCurrent() }
            showSwitchToast("Switched to \(name)")
            return
        }
        guard modelStore.isAvailable(id), !isStreaming else { return }
        if runtime.loadedModelID == id, case .ready = runtime.state {
            showSwitchToast("Switched to \(name)")
            return
        }
        Task { [weak self] in
            guard let self else { return }
            if (try? await self.runtime.engine(for: id)) != nil, self.selectedModelID == id {
                self.showSwitchToast("Switched to \(name)")
            }
        }
    }

    private func showSwitchToast(_ text: String) {
        toastTask?.cancel()
        switchToast = text
        toastTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(2.2))
            if !Task.isCancelled { self?.switchToast = nil }
        }
    }

    func conversationWillBeDeleted(_ conversation: Conversation) {
        if activeMessage?.conversation?.id == conversation.id {
            stop()
        }
        if retryConversation?.id == conversation.id {
            retryConversation = nil
            errorBanner = nil
        }
    }

    // MARK: - Turn lifecycle

    private func startAssistantTurn(in conversation: Conversation) {
        guard let store = conversationStore else { return }
        guard conversation.orderedMessages.last?.isUser == true else { return }

        let assistant = store.appendMessage(
            role: "assistant", content: "", to: conversation, modelID: selectedModelID)
        activeMessage = assistant
        streamingMessageID = assistant.id
        isStreaming = true
        isInThinkPhase = false
        contentBuffer = ""
        thinkingBuffer = ""
        contentCharacterCount = 0
        thinkingCharacterCount = 0

        let defaults = UserDefaults.standard
        // The preset (including Custom, which owns the old "custom instructions"
        // storage) is the single source of prompt customization.
        let systemPrompt = PromptBuilder.systemPrompt(
            presetText: PromptPresets.effectiveText(for: PromptPresets.selected),
            personalizationEnabled: false,
            customInstructions: "")
        let temperature = defaults.object(forKey: Keys.personalizationTemperature) as? Double
            ?? Keys.Defaults.personalizationTemperature

        let model = modelStore.model(for: selectedModelID)
        let supportsVision = model?.supportsVision == true
        let prior = conversation.orderedMessages
            .filter { $0.id != assistant.id }
            .filter { !($0.role == "assistant" && $0.content.isEmpty) }
        let pendingUserID = prior.last(where: { $0.isUser })?.id
        let history: [EngineMessage] = prior.map { message in
            if message.isUser {
                let text = PromptBuilder.userText(
                    content: message.content,
                    attachmentName: message.attachmentName,
                    attachmentText: message.attachmentText)
                // Images only on the final pending user turn, and only for VLMs.
                let images: [Data]
                if supportsVision,
                   message.id == pendingUserID,
                   let data = message.imageData {
                    images = [data]
                } else {
                    images = []
                }
                return EngineMessage(role: .user, text: text, images: images)
            }
            return EngineMessage(role: .assistant, text: message.content)
        }
        let modelID = selectedModelID
        let conversationID = conversation.id

        generationTask = Task { [weak self] in
            guard let self else { return }
            var thinkStartedAt: Date?
            var sawContent = false
            do {
                let engine = try await self.runtime.engine(for: modelID)
                self.runtime.beginGenerating(id: modelID)
                self.startFlushLoop()

                for try await event in engine.stream(
                    history: history, systemPrompt: systemPrompt,
                    temperature: temperature, conversationID: conversationID
                ) {
                    switch event {
                    case .delta(let delta):
                        self.contentBuffer += delta
                        self.contentCharacterCount += delta.count
                        if !sawContent {
                            sawContent = true
                            self.isInThinkPhase = false
                            if let start = thinkStartedAt {
                                assistant.thinkingSeconds = Date.now.timeIntervalSince(start)
                            }
                        }
                    case .thinkingDelta(let delta):
                        self.thinkingBuffer += delta
                        self.thinkingCharacterCount += delta.count
                        if thinkStartedAt == nil {
                            thinkStartedAt = .now
                            self.isInThinkPhase = true
                        }
                    case .thinkingReclassified:
                        self.thinkingBuffer = ""
                        self.thinkingCharacterCount = 0
                        assistant.thinkingSeconds = nil
                        self.isInThinkPhase = false
                    case .completed(let tokensPerSecond):
                        assistant.tokensPerSecond = tokensPerSecond
                    }
                }
            } catch is CancellationError {
                // Stop button: keep partial text, no banner.
            } catch {
                self.errorBanner = error.localizedDescription
            }
            self.finishTurn(in: conversation)
        }
    }

    private func finishTurn(in conversation: Conversation) {
        flushTask?.cancel()
        flushTask = nil
        flushBuffers()

        if let message = activeMessage,
           message.content.isEmpty, (message.thinking ?? "").isEmpty {
            // Nothing arrived (load/download failure or instant error): drop the
            // empty assistant message and offer Retry.
            conversationStore?.deleteMessage(message)
            if errorBanner != nil { retryConversation = conversation }
        }

        conversationStore?.save()
        runtime.endGenerating()
        isStreaming = false
        streamingMessageID = nil
        isInThinkPhase = false
        activeMessage = nil
        generationTask = nil
    }

    // MARK: - Throttled UI flush

    private func startFlushLoop() {
        flushTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                self.flushBuffers()
                let visibleCharacters = max(
                    self.contentCharacterCount,
                    self.thinkingCharacterCount)
                try? await Task.sleep(
                    for: StreamingRenderPolicy.flushInterval(
                        characterCount: visibleCharacters))
            }
        }
    }

    private func flushBuffers() {
        guard let message = activeMessage else { return }
        var changed = false
        if message.content != contentBuffer {
            message.content = contentBuffer
            changed = true
        }
        let thinking = thinkingBuffer.isEmpty ? nil : thinkingBuffer
        if message.thinking != thinking {
            message.thinking = thinking
            changed = true
        }
        if changed { transcriptRevision &+= 1 }
    }
}

/// Parsing and laying out Markdown gets more expensive as an answer grows.
/// Short replies retain the original low-latency cadence; long replies coalesce
/// more tokens into each update instead of reparsing the whole transcript at 10 Hz.
enum StreamingRenderPolicy {
    static func flushInterval(characterCount: Int) -> Duration {
        switch characterCount {
        case ..<2_000: .milliseconds(100)
        case ..<8_000: .milliseconds(180)
        case ..<20_000: .milliseconds(260)
        default: .milliseconds(350)
        }
    }
}
