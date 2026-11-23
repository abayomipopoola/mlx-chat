import CoreImage
import Foundation
import HuggingFace
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

/// One MLX model held in RAM, with a KV-cached ChatSession per conversation.
final class MLXEngine: ChatEngine, @unchecked Sendable {
    struct PriorMessageKey: Equatable, Sendable {
        let role: EngineMessage.Role
        let text: String
    }

    let model: CatalogModel
    private let streamingSettings: StreamedMoERuntimeSettings?
    var id: String { model.id }

    private let stateLock = NSLock()
    private var container: ModelContainer?
    private var loadedCapabilities: CatalogModel?
    private var streamingCleanup: (@Sendable () -> Void)?
    private var streamingErrorPoller: (@Sendable () -> Error?)?
    private var session: ChatSession?
    private var sessionConversationID: UUID?
    private var sessionSystemPrompt: String?
    /// Exact prior (non-pending) message signature the session has incorporated.
    /// Any mismatch (regenerate, delete, edit, cancel mid-stream) forces a rebuild.
    private var sessionKnownPriorSignature: [PriorMessageKey] = []
    /// Deduplicates concurrent load requests (switch-warming + first send) so the
    /// multi-GB weights are never materialized twice.
    private var inFlightLoad: Task<Void, Error>?
    /// Monotonic generation bumped on `unload()` so a racing load cannot
    /// resurrect a container after unload.
    private var loadGeneration: UInt64 = 0
    /// Serializes overlapping `stream()` turns on the shared `ChatSession`.
    private var activeStreamTask: Task<Void, Never>?
    private var activeStreamToken: UInt64 = 0

    init(model: CatalogModel, streamingSettings: StreamedMoERuntimeSettings? = nil) {
        self.model = model
        self.streamingSettings = streamingSettings
    }

    var isLoaded: Bool { stateLock.withLock { container != nil } }

    var currentLoadGeneration: UInt64 { stateLock.withLock { loadGeneration } }

    func isLoadGenerationCurrent(_ generation: UInt64) -> Bool {
        stateLock.withLock { loadGeneration == generation }
    }

    nonisolated static func shouldRebuildSession(
        hasSession: Bool,
        sessionConversationID: UUID?,
        sessionSystemPrompt: String?,
        sessionKnownPriorSignature: [PriorMessageKey],
        conversationID: UUID,
        systemPrompt: String,
        prior: [EngineMessage]
    ) -> Bool {
        guard hasSession,
              sessionConversationID == conversationID,
              sessionSystemPrompt == systemPrompt
        else { return true }
        let priorSignature = prior.map { PriorMessageKey(role: $0.role, text: $0.text) }
        return sessionKnownPriorSignature != priorSignature
    }

    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        enum Action {
            case alreadyLoaded
            case join(Task<Void, Error>)
            case start(Task<Void, Error>, UInt64)
        }
        let action: Action = stateLock.withLock {
            if container != nil { return .alreadyLoaded }
            if let existing = inFlightLoad { return .join(existing) }
            let generation = loadGeneration
            let task = Task { try await self.performLoad(generation: generation, progress: progress) }
            inFlightLoad = task
            return .start(task, generation)
        }
        switch action {
        case .alreadyLoaded:
            return
        case .join(let existing):
            try await existing.value
        case .start(let task, let generation):
            defer {
                stateLock.withLock {
                    if loadGeneration == generation {
                        inFlightLoad = nil
                    }
                }
            }
            try await task.value
        }
    }

    /// Buffer-cache limit while a model is loaded and generating: 2% of physical
    /// RAM, floored at 64 MB and capped at 512 MB so MLX reuses Metal buffers
    /// across decode steps without holding excessive memory.
    nonisolated static func activeBufferCacheLimit(
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Int {
        let floorBytes = 64 * 1024 * 1024
        let capBytes = 512 * 1024 * 1024
        let scaled = Int(min(physicalMemoryBytes / 50, UInt64(capBytes)))
        return max(floorBytes, min(scaled, capBytes))
    }

    nonisolated static func effectiveBufferCacheLimit(
        streaming: Bool,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Int {
        let standard = activeBufferCacheLimit(physicalMemoryBytes: physicalMemoryBytes)
        return streaming ? min(64 * 1024 * 1024, standard) : standard
    }

    private func performLoad(
        generation: UInt64,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws {
        // Local-first: a complete on-disk copy (plain ~/Models/<org>/<name> layout or a
        // hub snapshot) loads offline. A `.directory` configuration resolves without the
        // downloader AND keeps extraEOSTokens — the bare `loadContainer(from: URL, …)`
        // overload would drop them and Qwen models would run past <|im_end|>.
        guard let localDirectory = ModelStore.localMLXModelDirectory(for: model.huggingFaceRepoID) else {
            throw EngineError.unavailable(
                "This model is not downloaded or its files are missing.")
        }
        do {
            try ModelWeightNormalizer.normalizeIfNeeded(in: localDirectory)
        } catch {
            throw EngineError.generationFailed(
                "Couldn't prepare the model weights: \(error.localizedDescription)")
        }
        let configData = try? Data(contentsOf: localDirectory.appendingPathComponent("config.json"))
        let capabilities = ModelCatalog.applyingKnownCapabilities(to: model, configJSON: configData)
        let configuration = ModelConfiguration(
            directory: localDirectory, extraEOSTokens: capabilities.extraEOSTokens)
        let prismPolicy: HadamardQuantizedCheckpointPolicy?
        if let configData, PrismHadamardQwen35.isPrismConfig(configData) {
            do {
                prismPolicy = try PrismHadamardQwen35.policy(configJSON: configData)
            } catch {
                throw EngineError.generationFailed(
                    "Couldn't load the model: \(error.localizedDescription)")
            }
            await PrismHadamardQwen35.register()
        } else {
            prismPolicy = nil
        }
        await LFM3.register()
        await Spark25.register()
        await Inkling2.register()

        let streamingConfig: Inkling2StreamingConfig? = streamingSettings.map { settings in
            Inkling2StreamingConfig(
                modelDirectory: localDirectory,
                expertCacheSlots: settings.expertCacheSlots)
        }
        Memory.cacheLimit = Self.effectiveBufferCacheLimit(streaming: streamingConfig != nil)
        let loadProgress = LoadProgressHandlers.weights { weightProgress in
            progress(weightProgress.fractionCompleted)
        }

        do {
            // Route mlx C-level errors (e.g. affine quantize rejection) into
            // thrown MLXError.caught instead of the default fatalError/SIGTRAP.
            // Vision models load via VLMModelFactory; text-only via LLMModelFactory.
            let loaded = try await Inkling2.withStreamingLoad(streamingConfig) {
                try await withError {
                    if model.supportsVision {
                        try await VLMModelFactory.shared.loadContainer(
                            from: HubDownloader(ModelStore.hubClient),
                            using: HuggingFaceTokenizerLoader(),
                            configuration: configuration,
                            progress: loadProgress)
                    } else {
                        try await LLMModelFactory.shared.loadContainer(
                            from: HubDownloader(ModelStore.hubClient),
                            using: HuggingFaceTokenizerLoader(),
                            configuration: configuration,
                            progress: loadProgress)
                    }
                }
            }
            if streamingConfig != nil {
                let active = await loaded.perform { context in
                    (context.model as? Inkling2)?.isStreamingExpertsActive == true
                }
                guard active else {
                    await loaded.perform { context in
                        (context.model as? Inkling2)?.releaseStreamingResources()
                    }
                    throw EngineError.generationFailed(
                        "SSD-streamed expert loader did not activate for \(model.displayName).")
                }
            }
            if let prismPolicy {
                try await loaded.perform { context in
                    try prismPolicy.prepareLoaded(context.model)
                }
            }
            let hooks = await loaded.perform { context -> (
                cleanup: (@Sendable () -> Void)?,
                errorPoller: (@Sendable () -> Error?)?
            ) in
                guard let inkling = context.model as? Inkling2 else { return (nil, nil) }
                let cleanup: @Sendable () -> Void = { [weak inkling] in
                    inkling?.releaseStreamingResources()
                }
                let poller: @Sendable () -> Error? = { [weak inkling] in
                    inkling?.consumeStreamingError()
                }
                return (cleanup, poller)
            }
            // Unloaded (model switch) while we were loading: drop the weights
            // instead of resurrecting a container nobody owns.
            try Task.checkCancellation()
            let committed = stateLock.withLock { () -> Bool in
                guard loadGeneration == generation else { return false }
                container = loaded
                loadedCapabilities = capabilities
                streamingCleanup = hooks.cleanup
                streamingErrorPoller = hooks.errorPoller
                return true
            }
            guard committed else {
                hooks.cleanup?()
                throw CancellationError()
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw Self.mapLoadError(error)
        }
    }

    func unload() {
        let (loadToCancel, streamToCancel, cleanup) = stateLock.withLock {
            () -> (Task<Void, Error>?, Task<Void, Never>?, (@Sendable () -> Void)?) in
            loadGeneration &+= 1
            activeStreamToken &+= 1
            let loadTask = inFlightLoad
            let streamTask = activeStreamTask
            let releaseHook = streamingCleanup
            inFlightLoad = nil
            activeStreamTask = nil
            streamingCleanup = nil
            streamingErrorPoller = nil
            session = nil
            sessionConversationID = nil
            sessionSystemPrompt = nil
            sessionKnownPriorSignature = []
            container = nil
            loadedCapabilities = nil
            return (loadTask, streamTask, releaseHook)
        }
        loadToCancel?.cancel()
        streamToCancel?.cancel()
        cleanup?()
        Memory.cacheLimit = 0
        Memory.clearCache()
    }

    private static func mapLoadError(_ error: Error) -> Error {
        let text = String(describing: error)
        if text.contains("[quantize]") || text.localizedCaseInsensitiveContains("not supported") {
            return EngineError.generationFailed(
                "This model's quantization isn't supported by the MLX runtime in this build.")
        }
        if text.localizedCaseInsensitiveContains("unsupported") || text.localizedCaseInsensitiveContains("model type") {
            return EngineError.generationFailed("This model's architecture isn't supported by the MLX runtime.")
        }
        return EngineError.generationFailed("Couldn't load the model: \(error.localizedDescription)")
    }

    func stream(
        history: [EngineMessage],
        systemPrompt: String,
        temperature: Double,
        conversationID: UUID
    ) -> AsyncThrowingStream<EngineEvent, Error> {
        AsyncThrowingStream { continuation in
            let (streamToken, previousTask): (UInt64, Task<Void, Never>?) = stateLock.withLock {
                activeStreamToken &+= 1
                let prior = activeStreamTask
                return (activeStreamToken, prior)
            }
            previousTask?.cancel()

            let task = Task { [self] in
                if let previousTask {
                    await previousTask.value
                }
                defer {
                    stateLock.withLock {
                        if activeStreamToken == streamToken {
                            activeStreamTask = nil
                        }
                    }
                }

                var eventRouter: ThinkingEventRouter?
                var emittedAssistantText = ""
                do {
                    try Task.checkCancellation()
                    let snapshot = stateLock.withLock {
                        (
                            isCurrent: activeStreamToken == streamToken,
                            generation: loadGeneration,
                            container: container,
                            capabilities: loadedCapabilities,
                            session: session,
                            conversationID: sessionConversationID,
                            systemPrompt: sessionSystemPrompt,
                            priorSignature: sessionKnownPriorSignature,
                            streamingErrorPoller: streamingErrorPoller
                        )
                    }
                    guard snapshot.isCurrent else { throw CancellationError() }
                    guard let container = snapshot.container else {
                        throw EngineError.generationFailed("Model is not loaded.")
                    }
                    let effectiveModel = snapshot.capabilities
                        ?? ModelCatalog.applyingKnownCapabilities(to: model)
                    let trimmed = PromptBuilder.trimmed(
                        history: history, ctxTokens: effectiveModel.runtimeContextTokens)
                    guard let pending = trimmed.last, pending.role == .user else {
                        throw EngineError.generationFailed("No pending user message.")
                    }
                    let prior = Array(trimmed.dropLast())
                    let priorSignature = prior.map { PriorMessageKey(role: $0.role, text: $0.text) }

                    let needsRebuild = Self.shouldRebuildSession(
                        hasSession: snapshot.session != nil,
                        sessionConversationID: snapshot.conversationID,
                        sessionSystemPrompt: snapshot.systemPrompt,
                        sessionKnownPriorSignature: snapshot.priorSignature,
                        conversationID: conversationID,
                        systemPrompt: systemPrompt,
                        prior: prior)

                    let activeSession: ChatSession
                    if needsRebuild {
                        var params = GenerateParameters(temperature: Float(temperature))
                        // Safety cap: bound runaway generation. Small quantized reasoning
                        // models (e.g. Gemma 4 12B 4-bit) can loop in their thinking phase
                        // for thousands of tokens without closing it. Normal replies are far
                        // shorter, so this only stops a true runaway from generating forever.
                        params.maxTokens = 8192
                        if let streamingSettings {
                            params.maxKVSize = streamingSettings.contextTokens
                            params.prefill = PrefillParameters(
                                stepSize: Inkling2StreamingConfig.defaultPrefillChunkSize)
                        }
                        let rebuilt = ChatSession(
                            container,
                            instructions: systemPrompt,
                            history: prior.map {
                                Chat.Message(role: $0.role == .user ? .user : .assistant, content: $0.text)
                            },
                            generateParameters: params)
                        stateLock.withLock {
                            guard activeStreamToken == streamToken,
                                  loadGeneration == snapshot.generation
                            else { return }
                            session = rebuilt
                            sessionConversationID = conversationID
                            sessionSystemPrompt = systemPrompt
                            sessionKnownPriorSignature = priorSignature
                        }
                        activeSession = rebuilt
                    } else if let existing = snapshot.session {
                        existing.generateParameters.temperature = Float(temperature)
                        activeSession = existing
                    } else {
                        throw EngineError.generationFailed("Chat session unavailable.")
                    }

                    Memory.cacheLimit = Self.effectiveBufferCacheLimit(
                        streaming: streamingSettings != nil)
                    let modelTraits = await container.perform { context in
                        (
                            isInkling2: context.model is Inkling2,
                            runtimeSplitsReasoning: context.configuration.reasoningConfig != nil
                        )
                    }

                    // Reasoning routing: Inkling 2 takes `reasoning_effort` ("high" when always-on);
                    // Qwen3 / Ternary Bonsai / Gemma 4 take `enable_thinking`. When mlx-swift-lm
                    // already splits reasoning via `reasoningConfig`, `.reasoning` events pass
                    // straight through as `.thinkingDelta` and `.chunk` is not re-filtered.
                    let thinkingEnabled = ThinkingPreference.isEnabled
                    let thinking = effectiveModel.thinking
                    if let context = Self.additionalContext(
                        isInkling2: modelTraits.isInkling2,
                        thinking: thinking,
                        thinkingEnabled: thinkingEnabled)
                    {
                        activeSession.additionalContext = context
                    }

                    if !modelTraits.isInkling2, !modelTraits.runtimeSplitsReasoning {
                        var filter = ThinkTagFilter()
                        if let thinking {
                            let active = !thinking.toggleable || thinkingEnabled
                            // With thinking off the template pre-closes the span, so the stream
                            // is a plain answer — passthrough (the default filter) is correct.
                            filter = thinking.makeFilter(enabled: active)
                        }
                        eventRouter = ThinkingEventRouter(filter: filter)
                    }
                    var inklingSanitizer = InklingFrameSanitizer()
                    var tokensPerSecond: Double?

                    let emitEvent: (EngineEvent) -> Void = { event in
                        if case .delta(let text) = event {
                            emittedAssistantText += text
                        }
                        _ = continuation.yield(event)
                    }

                    // Images go only with the pending user turn; prior history
                    // stays text-only (old images are not re-fed to the KV cache).
                    let stream: AsyncThrowingStream<Generation, Error>
                    if effectiveModel.supportsVision, !pending.images.isEmpty {
                        let images: [UserInput.Image] = pending.images.compactMap { data in
                            guard let ciImage = CIImage(data: data) else { return nil }
                            return .ciImage(ciImage)
                        }
                        stream = activeSession.streamDetails(to: pending.text, images: images)
                    } else {
                        stream = activeSession.streamDetails(to: pending.text)
                    }

                    for try await event in stream {
                        try Task.checkCancellation()
                        if let streamingError = snapshot.streamingErrorPoller?() {
                            throw streamingError
                        }
                        switch event {
                        case .reasoning(let text):
                            if !text.isEmpty {
                                emitEvent(.thinkingDelta(text))
                            }
                        case .chunk(let text):
                            if modelTraits.isInkling2 {
                                let clean = inklingSanitizer.consume(text)
                                if !clean.isEmpty {
                                    emitEvent(.delta(clean))
                                }
                            } else if modelTraits.runtimeSplitsReasoning {
                                if !text.isEmpty {
                                    emitEvent(.delta(text))
                                }
                            } else {
                                eventRouter?.consume(text, emit: emitEvent)
                            }
                        case .info(let info):
                            tokensPerSecond = info.tokensPerSecond
                        case .toolCall, .rejectedToolCall, .steering:
                            break
                        }
                    }

                    if let streamingError = snapshot.streamingErrorPoller?() {
                        throw streamingError
                    }

                    if modelTraits.isInkling2 {
                        let tail = inklingSanitizer.finish()
                        if !tail.isEmpty {
                            emitEvent(.delta(tail))
                        }
                    } else if !modelTraits.runtimeSplitsReasoning {
                        eventRouter?.finish(emit: emitEvent)
                    }

                    // Session has now incorporated this user/assistant turn.
                    let updatedSignature = priorSignature + [
                        PriorMessageKey(role: .user, text: pending.text),
                        PriorMessageKey(role: .assistant, text: emittedAssistantText),
                    ]
                    stateLock.withLock {
                        guard activeStreamToken == streamToken,
                              loadGeneration == snapshot.generation
                        else { return }
                        sessionKnownPriorSignature = updatedSignature
                    }
                    continuation.yield(.completed(tokensPerSecond: tokensPerSecond))
                    continuation.finish()
                } catch is CancellationError {
                    // Flush safe buffered carry without leaking partial close tags or reclassifying.
                    eventRouter?.flushOnCancel { _ = continuation.yield($0) }
                    // Clear unconditionally even when superseded: the successor awaits this
                    // task before inspecting `session`, so it will cleanly rebuild instead of
                    // reusing a half-updated ChatSession.
                    stateLock.withLock {
                        session = nil
                        sessionKnownPriorSignature = []
                    }
                    continuation.yield(.completed(tokensPerSecond: nil))
                    continuation.finish()
                } catch let error as EngineError {
                    stateLock.withLock {
                        session = nil
                        sessionKnownPriorSignature = []
                    }
                    continuation.finish(throwing: error)
                } catch {
                    stateLock.withLock {
                        session = nil
                        sessionKnownPriorSignature = []
                    }
                    continuation.finish(throwing: EngineError.generationFailed(error.localizedDescription))
                }
            }
            stateLock.withLock {
                if activeStreamToken == streamToken {
                    activeStreamTask = task
                } else {
                    task.cancel()
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    nonisolated static func additionalContext(
        isInkling2: Bool,
        thinking: ThinkingConfig?,
        thinkingEnabled: Bool
    ) -> [String: any Sendable]? {
        if isInkling2 || thinking == .inklingFrames {
            let active = (thinking?.toggleable != true) || thinkingEnabled
            return Inkling2.additionalContext(thinkingEnabled: active)
        }
        if let thinking, thinking.toggleable {
            return ["enable_thinking": thinkingEnabled]
        }
        return nil
    }
}
