import Foundation
import Observation

/// Orchestrates a chat turn: engine resolution (download + load), streaming,
/// think-phase routing, throttled UI flushes, errors, stop/regenerate.
@MainActor
@Observable
final class ChatController {
    let modelStore: ModelStore
    let runtime: EngineRuntime
    /// Wired by RootView once the SwiftData context exists.
    var conversationStore: ConversationStore? {
        didSet {
            conversationStore?.onSaveError = { [weak self] error in
                self?.setErrorBanner("Couldn't save chat history: \(error.localizedDescription)")
            }
            conversationStore?.onSaveSuccess = { [weak self] in
                guard let self, self.errorBanner?.hasPrefix("Couldn't save chat history:") == true else {
                    return
                }
                self.dismissErrorBanner()
            }
        }
    }

    /// Selected engine/model id ("apple-intelligence" or an HF repo id).
    /// A persisted local selection whose files are gone (fresh install, moved
    /// streamed install, or model deleted outside the app) falls back to Apple
    /// Intelligence silently.
    private(set) var selectedModelID: String {
        didSet {
            UserDefaults.standard.set(selectedModelID, forKey: Keys.selectedModelID)
            guard oldValue != selectedModelID else { return }
            warmSwitchedModel()
        }
    }

    enum ToastStyle: Equatable {
        case modelSwitch
        case thinking(enabled: Bool)
    }

    /// Transient confirmation shown after a model switch or thinking-mode tap.
    private(set) var switchToast: String?
    private(set) var switchToastStyle: ToastStyle = .modelSwitch

    /// UI-verification hook: hold the confirmation banner for screenshots.
    var effectiveSwitchToast: String? {
#if DEBUG
        let arguments = ProcessInfo.processInfo.arguments
        if let index = arguments.firstIndex(of: "--fake-switch-toast"),
           index + 1 < arguments.count {
            return "Switched to \(arguments[index + 1])"
        }
#endif
        return switchToast
    }
    private var toastTask: Task<Void, Never>?

    private(set) var isStreaming = false
    private(set) var streamingMessageID: UUID?
    private(set) var isInThinkPhase = false
    /// Share of the prompt the engine has read, before any output arrives;
    /// nil when it does not report or once output starts.
    private(set) var prefillProgress: Double?
    private var storedErrorBanner: String?
    /// Conversation eligible for Retry after a failed (empty) turn.
    private var retryConversation: Conversation?

    /// Single source of truth for the banner + optional retry target. Direct
    /// writes via `errorBanner = ...` always clear any stale `retryConversation`.
    var errorBanner: String? {
        get { storedErrorBanner }
        set { setErrorBanner(newValue, retryConversation: nil) }
    }

    func setErrorBanner(_ message: String?, retryConversation: Conversation? = nil) {
        storedErrorBanner = message
        self.retryConversation = message == nil ? nil : retryConversation
    }

    func dismissErrorBanner() {
        setErrorBanner(nil, retryConversation: nil)
    }

    private var generationTask: Task<Void, Never>?
    private var flushTask: Task<Void, Never>?
    private var warmLoadTask: Task<Void, Never>?
    private var activeTurnToken: UInt64 = 0
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
        if saved == appleIntelligenceEngineID || store.isUsableLocalSelection(saved) {
            selectedModelID = saved
        } else {
            let usable = store.allModels
                .filter { store.isUsableLocalSelection($0.id) }
                .map(\.id)
            selectedModelID = Self.resolveFallbackModelID(
                excluding: saved,
                appleIntelligenceAvailable: AppleIntelligenceEngine.isAvailable,
                usableLocalModelIDs: usable)
        }
    }

    /// Fallback engine when the selected local model is missing, deleted, or ejected:
    /// Apple Intelligence if available, else the first usable local model not matching
    /// `excludedID`, else Apple Intelligence (which shows the setup hint).
    nonisolated static func resolveFallbackModelID(
        excluding excludedID: String? = nil,
        appleIntelligenceAvailable: Bool,
        usableLocalModelIDs: [String]
    ) -> String {
        if appleIntelligenceAvailable {
            return appleIntelligenceEngineID
        }
        if let first = usableLocalModelIDs.first(where: { $0 != excludedID }) {
            return first
        }
        return appleIntelligenceEngineID
    }

    func fallbackModelID(excluding excludedID: String? = nil) -> String {
        let usable = modelStore.allModels
            .filter { $0.id != excludedID && modelStore.isUsableLocalSelection($0.id) }
            .map(\.id)
        return Self.resolveFallbackModelID(
            excluding: excludedID,
            appleIntelligenceAvailable: AppleIntelligenceEngine.isAvailable,
            usableLocalModelIDs: usable)
    }

    func handleModelDeleted(_ id: String) {
        runtime.modelDeleted(id)
        let selectionMatchesDeleted = selectedModelID == id
            || MLXStreamedInstall.variant(forStreamedModelID: selectedModelID)?.sourceModelID == id
        if selectionMatchesDeleted, !isStreaming {
            selectedModelID = fallbackModelID(excluding: selectedModelID)
        }
    }

    /// Frees the resident local model and falls back per A1 policy.
    func ejectResidentModel() {
        guard !isStreaming else { return }
        let current = selectedModelID
        runtime.unloadCurrent()
        if current != appleIntelligenceEngineID {
            selectedModelID = fallbackModelID(excluding: current)
        }
    }

    var selectedModelDisplayName: String {
        modelDisplayName(for: selectedModelID)
    }

    func modelDisplayName(for id: String) -> String {
        if id == appleIntelligenceEngineID { return "Apple Intelligence" }
        return modelStore.model(for: id)?.pickerDisplayName ?? id
    }

    /// The active engine cannot change halfway through a turn. All model-selection
    /// surfaces use this method so the header always names the engine generating.
    func selectModel(_ id: String) {
        guard !isStreaming else { return }
        selectedModelID = id
    }

    /// Whether the header shows a thinking bulb for the selected model.
    var selectedModelSupportsThinking: Bool {
        modelStore.model(for: selectedModelID)?.supportsThinking == true
    }

    /// Whether the selected model's chat template can disable thinking per turn.
    var selectedModelSupportsThinkingToggle: Bool {
        modelStore.model(for: selectedModelID)?.supportsThinkingToggle == true
    }

    /// Whether the composer shows the attachment control for the selected model.
    var selectedModelSupportsVision: Bool {
        modelStore.model(for: selectedModelID)?.supportsVision == true
    }

    /// True while the selected local model is downloading or loading into RAM.
    /// The composer stays inert until that finishes so a send cannot race load.
    var isSelectedModelPreparing: Bool {
        runtime.effectiveState.isPreparing(selectedModelID)
    }

    var canRetry: Bool { retryConversation != nil }

    // MARK: - Sending

    func send(
        _ text: String,
        in conversation: Conversation,
        image: Data? = nil,
        attachment: (name: String, text: String)? = nil
    ) {
        guard !isStreaming, !isSelectedModelPreparing, let store = conversationStore else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || image != nil || attachment != nil else { return }
        dismissErrorBanner()
        store.appendMessage(
            role: "user",
            content: trimmed,
            to: conversation,
            imageData: image,
            attachmentName: attachment?.name,
            attachmentText: attachment?.text)

#if DEBUG
        // The release-demo recorder still exercises the real composer and
        // send path, but avoids requiring a 14 GB checkpoint during capture.
        // Keep this hook debug-only and opt-in so production builds always use
        // the selected engine.
        if ProcessInfo.processInfo.arguments.contains("--real-recording-demo")
            || ProcessInfo.processInfo.arguments.contains("--capture-release-video") {
            playRecordingDemoTurn(in: conversation)
            return
        }
#endif
        startAssistantTurn(in: conversation)
    }

    func retry() {
        guard !isStreaming, !isSelectedModelPreparing,
              let conversation = retryConversation,
              conversationStore != nil
        else { return }
        dismissErrorBanner()
        startAssistantTurn(in: conversation)
    }

    /// Regenerate the LAST assistant message.
    func regenerate(_ message: ChatMessage, in conversation: Conversation) {
        guard !isStreaming, !isSelectedModelPreparing,
              message.role == "assistant",
              conversation.orderedMessages.last?.id == message.id,
              let store = conversationStore
        else { return }
        dismissErrorBanner()
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
        scheduleWarmLoad { [weak self] in
            _ = try? await self?.runtime.engine(for: id)
        }
    }

    /// Loads the newly selected model right away (instead of lazily on first
    /// send) so the picker can show progress, then confirms with a toast.
    private func warmSwitchedModel() {
        let id = selectedModelID
        let name = selectedModelDisplayName
        warmLoadTask?.cancel()
        warmLoadTask = nil
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
        scheduleWarmLoad { [weak self] in
            guard let self else { return }
            if (try? await self.runtime.engine(for: id)) != nil,
               !Task.isCancelled,
               self.selectedModelID == id {
                self.showSwitchToast("Switched to \(name)")
            }
        }
    }

    var currentWarmLoadTask: Task<Void, Never>? { warmLoadTask }

    func scheduleWarmLoad(_ work: @escaping @MainActor () async -> Void) {
        warmLoadTask?.cancel()
        warmLoadTask = Task {
            await work()
        }
    }

    func showThinkingToast(_ text: String, enabled: Bool = true) {
        presentToast(text, style: .thinking(enabled: enabled))
    }

    private func showSwitchToast(_ text: String) {
        presentToast(text, style: .modelSwitch)
    }

    private func presentToast(_ text: String, style: ToastStyle) {
        toastTask?.cancel()
        switchToastStyle = style
        switchToast = text
        toastTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(2.2))
            if !Task.isCancelled { self?.switchToast = nil }
        }
    }

    func conversationWillBeDeleted(_ conversation: Conversation) {
        if activeMessage?.conversation?.id == conversation.id {
            // Invalidate the in-flight turn immediately so async cancellation
            // cannot mutate, delete, or save already-deleted SwiftData rows.
            activeTurnToken &+= 1
            flushTask?.cancel()
            flushTask = nil
            generationTask?.cancel()
            generationTask = nil
            activeMessage = nil
            prefillProgress = nil
            isStreaming = false
            streamingMessageID = nil
            isInThinkPhase = false
            contentBuffer = ""
            thinkingBuffer = ""
            contentCharacterCount = 0
            thinkingCharacterCount = 0
            runtime.endGenerating()
        }
        if retryConversation?.id == conversation.id {
            dismissErrorBanner()
        }
    }

    // MARK: - Turn lifecycle

#if DEBUG
    /// Plays a short deterministic streamed-MoE turn after a real send. This
    /// is only used by the native screen recorder: the user still clicks the
    /// picker, types into the actual TextField, and submits through the normal
    /// ChatScreen -> ChatController path.
    private func playRecordingDemoTurn(in conversation: Conversation) {
        guard let store = conversationStore,
              conversation.orderedMessages.last?.isUser == true
        else { return }

        let assistant = store.appendMessage(
            role: "assistant", content: "", to: conversation,
            modelID: selectedModelID)
        assistant.thinking = DemoSeed.streamedMoEThinking
        assistant.thinkingSeconds = 2.8
        assistant.content = DemoSeed.streamedMoEAnswer
        assistant.tokensPerSecond = 18.4
        store.save()
        transcriptRevision &+= 1
    }
#endif

    private func startAssistantTurn(in conversation: Conversation) {
        guard let store = conversationStore else { return }
        guard conversation.orderedMessages.last?.isUser == true else { return }

        let modelID = selectedModelID
        warmLoadTask?.cancel()
        warmLoadTask = nil
        activeTurnToken &+= 1
        let turnToken = activeTurnToken

        let assistant = store.appendMessage(
            role: "assistant", content: "", to: conversation, modelID: modelID)
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
        let systemPrompt = PromptBuilder.currentSystemPrompt()
        let temperature = defaults.object(forKey: Keys.personalizationTemperature) as? Double
            ?? Keys.Defaults.personalizationTemperature

        let model = modelStore.model(for: modelID)
        let supportsVision = model?.supportsVision == true
        let allPrior = conversation.orderedMessages
            .filter { $0.id != assistant.id }
            .filter { !($0.role == "assistant" && $0.content.isEmpty) }
        let prior = Self.historyMessages(allPrior, for: modelID)
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
        let conversationID = conversation.id

        generationTask = Task { [weak self] in
            guard let self else { return }
            var thinkStartedAt: Date?
            var sawContent = false
            do {
                let engine = try await self.runtime.engine(for: modelID)
                guard self.activeTurnToken == turnToken else { return }
                self.runtime.beginGenerating(id: modelID)
                self.startFlushLoop(turnToken: turnToken)

                for try await event in engine.stream(
                    history: history, systemPrompt: systemPrompt,
                    temperature: temperature, conversationID: conversationID
                ) {
                    guard self.activeTurnToken == turnToken else { return }
                    switch event {
                    case .prefill(let done, let total):
                        if !sawContent, thinkStartedAt == nil, total > 0 {
                            self.prefillProgress = Double(done) / Double(total)
                        }
                    case .delta(let delta):
                        self.prefillProgress = nil
                        self.contentBuffer += delta
                        self.contentCharacterCount += delta.utf8.count
                        if !sawContent {
                            sawContent = true
                            self.isInThinkPhase = false
                            if let start = thinkStartedAt {
                                assistant.thinkingSeconds = Date.now.timeIntervalSince(start)
                            }
                        }
                    case .thinkingDelta(let delta):
                        self.prefillProgress = nil
                        self.thinkingBuffer += delta
                        self.thinkingCharacterCount += delta.utf8.count
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
                if self.activeTurnToken == turnToken {
                    self.setErrorBanner(error.localizedDescription)
                }
            }
            self.finishTurn(in: conversation, turnToken: turnToken)
        }
    }

    /// Keep only the current model's uninterrupted section of a mixed-model chat.
    /// This prevents another model's response format and leaked control tokens from
    /// becoming examples in the next engine's transcript.
    static func historyMessages(_ messages: [ChatMessage], for modelID: String) -> [ChatMessage] {
        guard let boundary = messages.lastIndex(where: {
            !$0.isUser && $0.modelID != modelID
        }) else { return messages }
        return Array(messages.suffix(from: messages.index(after: boundary)))
    }

    private func finishTurn(in conversation: Conversation, turnToken: UInt64) {
        guard activeTurnToken == turnToken else { return }
        prefillProgress = nil
        flushTask?.cancel()
        flushTask = nil
        flushBuffers()

        if let message = activeMessage,
           message.content.isEmpty, (message.thinking ?? "").isEmpty {
            // Nothing arrived (load/download failure or instant error): drop the
            // empty assistant message and offer Retry.
            conversationStore?.deleteMessage(message)
            if let banner = errorBanner {
                setErrorBanner(banner, retryConversation: conversation)
            }
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

    private func startFlushLoop(turnToken: UInt64) {
        flushTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self, self.activeTurnToken == turnToken else { return }
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
