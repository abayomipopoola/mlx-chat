import Foundation
import TurboFieldfare

/// Text-only MoE chat backend powered by TurboFieldfare.
///
/// This is intentionally separate from `MLXEngine`: dense and ordinary MLX
/// models keep their existing loading and cached-session behavior. The actor
/// below owns TurboFieldfare's resident core, FP16 KV cache, Metal state, and
/// SSD-backed expert streamers.
final class StreamedMoEEngine: ChatEngine {
    let model: CatalogModel
    let modelDirectory: URL
    var id: String { model.id }

    private let runtimeSettings: StreamedMoERuntimeSettings
    private var session: StreamedMoESession?
    private var inFlightLoad: Task<Void, Error>?

    init(
        model: CatalogModel,
        modelDirectory: URL,
        runtimeSettings: StreamedMoERuntimeSettings = .current()
    ) {
        precondition(model.backend == .streamedMoE)
        self.model = model
        self.modelDirectory = modelDirectory.standardizedFileURL
        self.runtimeSettings = runtimeSettings
    }

    var isLoaded: Bool { session != nil }

    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        guard session == nil else { return }
        if let inFlightLoad {
            try await inFlightLoad.value
            return
        }
        let task = Task { try await self.performLoad(progress: progress) }
        inFlightLoad = task
        defer { inFlightLoad = nil }
        try await task.value
    }

    private func performLoad(progress: @escaping @Sendable (Double) -> Void) async throws {
        guard let variant = StreamedMoEInstall.variant(
            forStreamedModelID: model.id) else {
            throw EngineError.unavailable("This SSD-streamed model is not supported.")
        }
        let candidate = StreamedMoESession()
        do {
            try await candidate.load(
                directory: modelDirectory,
                variant: variant,
                settings: runtimeSettings,
                progress: progress)
            try Task.checkCancellation()
            session = candidate
            // The first chat then starts after the system prompt.
            await candidate.prewarm(systemPrompt: PromptBuilder.currentSystemPrompt())
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw EngineError.generationFailed(
                "Couldn’t load the SSD-streamed model: \(String(describing: error))")
        }
    }

    func unload() {
        inFlightLoad?.cancel()
        inFlightLoad = nil
        if let session {
            Task { await session.cancelPrewarm() }
        }
        session = nil
    }

    func stream(
        history: [EngineMessage],
        systemPrompt: String,
        temperature: Double,
        conversationID: UUID
    ) -> AsyncThrowingStream<EngineEvent, Error> {
        AsyncThrowingStream { continuation in
            guard let session else {
                continuation.finish(throwing: EngineError.generationFailed("Model is not loaded."))
                return
            }
            let task = Task {
                do {
                    let tokensPerSecond = try await session.generate(
                        history: history,
                        systemPrompt: systemPrompt,
                        temperature: temperature,
                        conversationID: conversationID,
                        onEvent: { continuation.yield($0) })
                    continuation.yield(.completed(tokensPerSecond: tokensPerSecond))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.yield(.completed(tokensPerSecond: nil))
                    continuation.finish()
                } catch let error as EngineError {
                    continuation.finish(throwing: error)
                } catch {
                    continuation.finish(throwing: EngineError.generationFailed(
                        "SSD-streamed generation failed: \(String(describing: error))"))
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

/// Serial owner of TurboFieldfare's non-copyable Metal and streaming state.
private actor StreamedMoESession {
    private struct LoadedState {
        let directory: URL
        let architecture: StreamedMoEArchitecture
        let context: MetalContext
        let tokenizer: GFTokenizer
        let model: Model
        let runner: any ContinuableLogitProducer
        let maxContext: Int
        let scratch: RawCompletionScratch
        let settings: StreamedMoERuntimeSettings
        var vision: VisionRuntime?
    }

    private var loadedState: LoadedState?
    private var kvCache: StreamedMoEKVCacheState?
    /// Feeds the system prompt after load. `generate` awaits it first: the
    /// actor is reentrant, and both drive the same runner.
    private var prewarmTask: Task<Void, Never>?

    func load(
        directory: URL,
        variant: StreamedMoEVariant,
        settings: StreamedMoERuntimeSettings,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws {
        guard loadedState == nil else { return }
        if let error = StreamedMoEInstall.validationError(
            at: directory,
            for: variant) {
            throw EngineError.unavailable(error)
        }

        progress(0.05)
        let context = try MetalContext()
        try Task.checkCancellation()

        progress(0.30)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: directory)
        try Task.checkCancellation()

        progress(0.45)
        let runtimeConfiguration = RuntimeConfiguration(
            expertCacheSlots: settings.expertCacheSlots,
            prefillEnabled: true,
            prefillChunkTokens: 128,
            forceLogitsHead: true)
        let model = try Model.load(
            directoryURL: directory,
            device: context.device,
            streamingMode: .pread(slotCount: settings.expertCacheSlots),
            expertCachePolicy: runtimeConfiguration.modelExpertCachePolicy,
            integrityPolicy: .sizeCheckTrustedReceipt)
        try Task.checkCancellation()

        progress(0.80)
        let runner: any ContinuableLogitProducer
        if variant.architecture == .flint {
            runner = try FlintForwardRunner(
                model: model,
                context: context,
                maxContext: settings.contextTokens)
        } else if variant.architecture == .lfm2moe {
            runner = try LFM2MoEForwardRunner(
                model: model,
                context: context,
                maxContext: settings.contextTokens)
        } else {
            runner = try RealForwardRunner(
                model: model,
                context: context,
                maxContext: settings.contextTokens,
                runtimeConfiguration: runtimeConfiguration)
        }
        let scratch = try RawCompletionScratch(
            context: context,
            vocab: model.config.vocabSize,
            logitSoftcap: Float(model.config.finalLogitSoftcap))
        try Task.checkCancellation()

        loadedState = LoadedState(
            directory: directory,
            architecture: variant.architecture,
            context: context,
            tokenizer: tokenizer,
            model: model,
            runner: runner,
            maxContext: settings.contextTokens,
            scratch: scratch,
            settings: settings,
            vision: nil)
        progress(1.0)
    }

    func generate(
        history: [EngineMessage],
        systemPrompt: String,
        temperature: Double,
        conversationID: UUID,
        onEvent: @escaping @Sendable (EngineEvent) -> Void
    ) async throws -> Double? {
        if let prewarmTask {
            await prewarmTask.value
            self.prewarmTask = nil
        }
        guard let loadedState else {
            throw EngineError.generationFailed("Model is not loaded.")
        }
        let context = loadedState.context
        let tokenizer = loadedState.tokenizer
        let runner = loadedState.runner
        let scratch = loadedState.scratch
        let settings = loadedState.settings
        let trimmed = PromptBuilder.trimmed(
            history: history,
            ctxTokens: settings.contextTokens)
        guard trimmed.last?.role == .user else {
            throw EngineError.generationFailed("No pending user message.")
        }

        let flint = loadedState.architecture == .flint
        let thinkingEnabled = Self.thinkingEnabled(loadedState)
        let wantsImages = trimmed.contains { !$0.images.isEmpty }
        let multimodalInput: MultimodalPrefillInput?
        let promptIDs: [Int32]
        if wantsImages {
            let input = try encodeMultimodalPrompt(
                history: trimmed,
                systemPrompt: systemPrompt,
                tokenizer: tokenizer)
            multimodalInput = input
            promptIDs = input.effectiveTokenIDs
        } else {
            multimodalInput = nil
            promptIDs = try Self.textPromptIDs(loadedState, history: trimmed,
                                               systemPrompt: systemPrompt,
                                               thinkingEnabled: thinkingEnabled)
        }
        let maxNewTokens = min(8_192, loadedState.maxContext - promptIDs.count)
        guard maxNewTokens > 0 else {
            throw EngineError.contextExceeded(
                "This conversation exceeds the streamed model’s \(settings.contextTokens / 1_024)K in-memory context. Start a new chat, shorten the prompt, or raise Context in the model’s SSD settings.")
        }

        var eventRouter = StreamedMoEEventRouter(
            tokenizer: tokenizer,
            thinkingEnabled: flint ? true : thinkingEnabled)
        let config = GenerationConfig(
            maxNewTokens: maxNewTokens,
            temperature: Float(max(temperature, 0)),
            topK: flint ? 20 : 64,
            topP: 0.95)

        let allowsRewind = Self.allowsRewind(runner)
        var start = StreamedMoEKVReusePolicy.start(
            newPromptIDs: promptIDs,
            cache: kvCache,
            conversationID: conversationID,
            systemPrompt: systemPrompt,
            thinkingEnabled: thinkingEnabled,
            allowsRewind: allowsRewind)
        if multimodalInput != nil { start = .reset }
        // Flint's sliding layers keep a bounded ring, so a deep rewind must
        // start over instead.
        if case .resume(let cached) = start, let flint = runner as? FlintForwardRunner,
           flint.continuationPosition - cached > flint.maxRewindDistance {
            start = .reset
        }
        // Flint and LFM2 feed the prompt one token at a time inside their
        // chunked prefill, running the output head only on the last token.
        let prefillConfig = PrefillRuntimeConfig.defaultChunked

        let result: RawDecodeResult
        var lastPrefillEvent = Date.distantPast
        do {
            result = try await runRawCompletion(
                producer: runner,
                tokenizer: tokenizer,
                promptIds: promptIDs,
                multimodalInput: multimodalInput,
                config: config,
                context: context,
                scratch: scratch,
                prefillConfig: prefillConfig,
                start: start
            ) { progress in
                switch progress {
                case .prefill(let done, let total):
                    let now = Date()
                    if done >= total || now.timeIntervalSince(lastPrefillEvent) >= 0.1 {
                        lastPrefillEvent = now
                        onEvent(.prefill(done: done, total: total))
                    }
                case .token(_, let tokenID, let delta):
                    eventRouter.consume(tokenID: tokenID, delta: delta, emit: onEvent)
                case .tail(let text):
                    eventRouter.consumeTail(text, emit: onEvent)
                }
            }
        } catch is CancellationError where allowsRewind && multimodalInput == nil {
            // Stop lands between tokens, so a rewindable runner still holds a
            // valid prefix of this prompt. Keep it for the next turn.
            let fed = min(runner.continuationPosition, promptIDs.count)
            kvCache = fed > 0
                ? StreamedMoEKVCacheState(
                    conversationID: conversationID,
                    systemPrompt: systemPrompt,
                    thinkingEnabled: thinkingEnabled,
                    kvTokenIDs: Array(promptIDs.prefix(fed)))
                : nil
            throw CancellationError()
        } catch {
            kvCache = nil
            throw error
        }

        kvCache = StreamedMoEKVCacheState(
            conversationID: conversationID,
            systemPrompt: systemPrompt,
            thinkingEnabled: thinkingEnabled,
            kvTokenIDs: result.kvBackedTokenIDs)

        eventRouter.finish(emit: onEvent)
        return result.decodeSeconds > 0
            ? Double(result.newTokens) / result.decodeSeconds
            : nil
    }

    /// Starts feeding the part of every text prompt that comes before the
    /// user's words (the system prompt and the user-turn header), so the first
    /// chat resumes after it. Only runners that can rewind take part: their
    /// cache serves any conversation that shares those tokens.
    func prewarm(systemPrompt: String) {
        guard let state = loadedState, Self.allowsRewind(state.runner),
              prewarmTask == nil else { return }
        let thinkingEnabled = Self.thinkingEnabled(state)
        func ids(_ text: String) throws -> [Int32] {
            try Self.textPromptIDs(state, history: [EngineMessage(role: .user, text: text)],
                                   systemPrompt: systemPrompt,
                                   thinkingEnabled: thinkingEnabled)
        }
        guard let first = try? ids("a"), let second = try? ids("z") else { return }
        let shared = zip(first, second).prefix { $0 == $1 }.count
        guard shared > 0 else { return }
        let prefix = Array(first.prefix(shared))
        if let kvCache, kvCache.kvTokenIDs.starts(with: prefix) { return }
        prewarmTask = Task {
            await self.runPrewarm(prefix: prefix, state: state, systemPrompt: systemPrompt,
                                  thinkingEnabled: thinkingEnabled)
        }
    }

    /// Stops warming between prompt tokens, so an unloaded model frees soon.
    func cancelPrewarm() {
        prewarmTask?.cancel()
    }

    private func runPrewarm(prefix: [Int32], state: LoadedState, systemPrompt: String,
                            thinkingEnabled: Bool) async {
        do {
            try await runRawPrefill(producer: state.runner, promptIds: prefix,
                                    scratch: state.scratch)
            kvCache = StreamedMoEKVCacheState(
                conversationID: UUID(),
                systemPrompt: systemPrompt,
                thinkingEnabled: thinkingEnabled,
                kvTokenIDs: prefix)
        } catch {
            kvCache = nil
        }
    }

    private static func allowsRewind(_ runner: any ContinuableLogitProducer) -> Bool {
        runner is LFM2MoEForwardRunner || runner is FlintForwardRunner
    }

    private static func thinkingEnabled(_ state: LoadedState) -> Bool {
        state.architecture == .flint ? false : (
            state.architecture.thinkingIsOptional ? ThinkingPreference.isEnabled : true)
    }

    private static func textPromptIDs(_ state: LoadedState,
                                      history: [EngineMessage],
                                      systemPrompt: String,
                                      thinkingEnabled: Bool) throws -> [Int32] {
        let rendered = state.architecture == .flint
            ? GFTokenizer.flintPrompt(
                turns: history.map {
                    ($0.role == .assistant ? "assistant" : "user", $0.text)
                },
                system: systemPrompt)
            : try StreamedMoEPrompt.render(
                tokenizer: state.tokenizer,
                history: history,
                systemPrompt: systemPrompt,
                thinkingEnabled: thinkingEnabled)
        return state.tokenizer.encode(rendered, addBOS: false)
    }

    private func encodeMultimodalPrompt(
        history: [EngineMessage],
        systemPrompt: String,
        tokenizer: GFTokenizer
    ) throws -> MultimodalPrefillInput {
        guard let state = loadedState else {
            throw EngineError.generationFailed("Model is not loaded.")
        }
        guard state.architecture == .gemma4 else {
            throw EngineError.generationFailed(
                "This SSD-streamed model cannot use images.")
        }
        let vision: VisionRuntime
        if let existing = state.vision {
            vision = existing
        } else {
            do {
                vision = try VisionRuntime.open(
                    textModelURL: state.directory,
                    context: state.context)
            } catch {
                throw EngineError.generationFailed(
                    "Couldn’t open the vision pack: \(String(describing: error))")
            }
            loadedState?.vision = vision
        }

        var featuresByID: [UUID: VisionFeatures] = [:]
        var messages: [MultimodalMessage] = []
        let system = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !system.isEmpty {
            messages.append(MultimodalMessage(role: .system, content: [.text(system)]))
        }
        for message in history {
            var parts: [MultimodalContentPart] = []
            let text = message.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                parts.append(.text(text))
            }
            for data in message.images {
                let id = UUID()
                featuresByID[id] = try encodeAttachedImage(
                    data,
                    vision: vision,
                    languageModel: state.model)
                parts.append(.image(id: id))
            }
            if parts.isEmpty {
                parts = [.text("")]
            }
            messages.append(MultimodalMessage(
                role: message.role == .assistant ? .assistant : .user,
                content: parts))
        }
        do {
            return try MultimodalPromptRenderer.render(
                messages: messages,
                featuresByID: featuresByID,
                tokenizer: tokenizer)
        } catch {
            throw EngineError.generationFailed(
                "Couldn’t prepare the image prompt: \(String(describing: error))")
        }
    }

    private func encodeAttachedImage(
        _ data: Data,
        vision: VisionRuntime,
        languageModel: Model
    ) throws -> VisionFeatures {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-chat-vision-\(UUID().uuidString).jpg")
        do {
            try data.write(to: url, options: .atomic)
            return try vision.encodeImage(
                at: url,
                languageModel: languageModel)
        } catch {
            throw EngineError.generationFailed(
                "Couldn’t encode the attached image: \(String(describing: error))")
        }
    }
}

/// Bridges TurboFieldfare's token-aware decode stream into the app's thinking
/// filter. The detokenizer strips special tokens, so Gemma's `<channel|>` and
/// Qwen's `</think>` have empty text deltas. Detect the family-specific close
/// token by ID and synthesize its marker for the shared text filter.
struct StreamedMoEEventRouter {
    private var router: ThinkingEventRouter
    private let thoughtOpenID: Int32
    private let thoughtOpenMarker: String
    private let thoughtCloseID: Int32
    private let thoughtCloseMarker: String
    private var awaitingThoughtClose: Bool
    private var leadingOpenBuffer: String?

    init(tokenizer: GFTokenizer, thinkingEnabled: Bool) {
        switch tokenizer.dialect {
        case .gemma:
            self.init(
                thinkingConfig: .gemmaChannel,
                thinkingEnabled: thinkingEnabled,
                thoughtCloseID: tokenizer.channelEndID)
        case .chatml:
            self.init(
                thinkingConfig: .thinkTags(startsInside: true, toggleable: true),
                thinkingEnabled: thinkingEnabled,
                thoughtCloseID: tokenizer.thinkEndID ?? tokenizer.channelEndID)
        case .flint:
            self.init(
                thinkingConfig: .thinkTags(startsInside: false, toggleable: false),
                thinkingEnabled: true,
                thoughtOpenID: tokenizer.thinkStartID ?? -1,
                thoughtCloseID: tokenizer.thinkEndID ?? -1,
                preopened: true)
        }
    }

    init(
        thinkingConfig: ThinkingConfig,
        thinkingEnabled: Bool,
        thoughtOpenID: Int32 = -1,
        thoughtCloseID: Int32,
        preopened: Bool? = nil
    ) {
        let opened = preopened ?? thinkingEnabled
        self.thoughtOpenID = thoughtOpenID
        self.thoughtOpenMarker = thinkingConfig.open
        self.thoughtCloseID = thoughtCloseID
        self.thoughtCloseMarker = thinkingConfig.close
        router = ThinkingEventRouter(filter: thinkingConfig.makeFilter(
            enabled: thinkingEnabled,
            preopened: opened))
        awaitingThoughtClose = opened
        leadingOpenBuffer = (opened && !thinkingConfig.startsInside && !thinkingConfig.open.isEmpty) ? "" : nil
    }

    mutating func consume(
        tokenID: Int32,
        delta: String,
        emit: (EngineEvent) -> Void
    ) {
        var effectiveDelta = delta
        if let pendingOpen = leadingOpenBuffer {
            if !thoughtOpenMarker.isEmpty,
               thoughtOpenID >= 0,
               tokenID == thoughtOpenID {
                leadingOpenBuffer = nil
                let combined = pendingOpen + delta
                let trimmed = String(combined.drop(while: { $0.isWhitespace }))
                if trimmed.hasPrefix(thoughtOpenMarker) {
                    effectiveDelta = String(trimmed.dropFirst(thoughtOpenMarker.count))
                } else {
                    effectiveDelta = combined
                }
            } else if !thoughtCloseMarker.isEmpty,
                      thoughtCloseID >= 0,
                      tokenID == thoughtCloseID {
                leadingOpenBuffer = nil
                if !pendingOpen.isEmpty {
                    router.consume(pendingOpen, emit: emit)
                }
            } else {
                let combined = pendingOpen + delta
                let trimmed = String(combined.drop(while: { $0.isWhitespace }))
                if trimmed.isEmpty {
                    leadingOpenBuffer = combined
                    return
                } else if trimmed.hasPrefix(thoughtOpenMarker) {
                    leadingOpenBuffer = nil
                    effectiveDelta = String(trimmed.dropFirst(thoughtOpenMarker.count))
                } else if thoughtOpenMarker.hasPrefix(trimmed) {
                    leadingOpenBuffer = combined
                    return
                } else {
                    leadingOpenBuffer = nil
                    effectiveDelta = combined
                }
            }
        }

        if router.isPending,
           !thoughtOpenMarker.isEmpty,
           thoughtOpenID >= 0,
           tokenID == thoughtOpenID {
            let piece = effectiveDelta.contains(thoughtOpenMarker)
                ? effectiveDelta : effectiveDelta + thoughtOpenMarker
            router.consume(piece, emit: emit)
            awaitingThoughtClose = router.isThinking
        } else if (awaitingThoughtClose || router.isThinking),
                  !thoughtCloseMarker.isEmpty,
                  thoughtCloseID >= 0,
                  tokenID == thoughtCloseID {
            awaitingThoughtClose = false
            let piece = effectiveDelta.contains(thoughtCloseMarker)
                ? effectiveDelta : effectiveDelta + thoughtCloseMarker
            router.consume(piece, emit: emit)
        } else {
            router.consume(effectiveDelta, emit: emit)
            if router.isThinking {
                awaitingThoughtClose = true
            }
        }
    }

    mutating func consumeTail(_ text: String, emit: (EngineEvent) -> Void) {
        if let pendingOpen = leadingOpenBuffer {
            leadingOpenBuffer = nil
            router.consume(pendingOpen + text, emit: emit)
            return
        }
        router.consume(text, emit: emit)
    }

    mutating func finish(emit: (EngineEvent) -> Void) {
        if let pendingOpen = leadingOpenBuffer, !pendingOpen.isEmpty {
            leadingOpenBuffer = nil
            router.consume(pendingOpen, emit: emit)
        }
        leadingOpenBuffer = nil
        router.finish(emit: emit)
    }
}

/// Uses TurboFieldfare's public family-specific chat template when thinking is
/// off. Thinking prompts leave the model's thought span open so streamed output
/// can be split into thought and answer events.
enum StreamedMoEPrompt {
    private enum Role: String {
        case system
        case user
        case model
    }

    private enum Token {
        static let bos = "<bos>"
        static let turnOpen = "<|turn>"
        static let turnClose = "<turn|>"
        static let think = "<|think|>"
        static let thoughtChannel = "<|channel>thought"
        static let chatMLTurnOpen = "<|im_start|>"
        static let chatMLTurnClose = "<|im_end|>"
        static let qwenThink = "<think>"
        static let lfmBOS = "<|startoftext|>"
    }

    static func render(
        tokenizer: GFTokenizer,
        history: [EngineMessage],
        systemPrompt: String,
        thinkingEnabled: Bool
    ) throws -> String {
        if thinkingEnabled {
            switch tokenizer.dialect {
            case .gemma:
                return renderGemmaThinkingPrompt(
                    history: history,
                    systemPrompt: systemPrompt)
            case .chatml:
                return renderChatMLThinkingPrompt(
                    history: history,
                    systemPrompt: systemPrompt,
                    family: tokenizer.chatMLFamily ?? .qwen)
            case .flint:
                return GFTokenizer.flintPrompt(
                    turns: history.map {
                        ($0.role == .assistant ? "assistant" : "user", $0.text)
                    },
                    system: systemPrompt)
            }
        }

        var messages: [GFTokenizer.Message] = []
        let system = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !system.isEmpty {
            messages.append(GFTokenizer.Message(role: .system, content: system))
        }
        for message in history {
            messages.append(GFTokenizer.Message(
                role: message.role == .assistant ? .assistant : .user,
                content: message.text))
        }
        return try tokenizer.applyChatTemplate(messages)
    }

    static func renderGemmaThinkingPrompt(
        history: [EngineMessage],
        systemPrompt: String
    ) -> String {
        var prompt = Token.bos
        let rawSystem = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let system = rawSystem.isEmpty ? Token.think : Token.think + "\n" + rawSystem
        appendTurn(role: .system, content: system, to: &prompt)
        for message in history {
            appendTurn(
                role: message.role == .assistant ? .model : .user,
                content: message.text.trimmingCharacters(in: .whitespacesAndNewlines),
                to: &prompt)
        }
        prompt += Token.turnOpen + Role.model.rawValue + "\n" + Token.thoughtChannel + "\n"
        return prompt
    }

    static func renderChatMLThinkingPrompt(
        history: [EngineMessage],
        systemPrompt: String,
        family: ChatMLFamily = .qwen
    ) -> String {
        var prompt = family == .lfm ? Token.lfmBOS : ""
        let system = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !system.isEmpty {
            appendChatMLTurn(role: "system", content: system, to: &prompt)
        }
        for message in history {
            appendChatMLTurn(
                role: message.role == .assistant ? "assistant" : "user",
                content: message.text.trimmingCharacters(in: .whitespacesAndNewlines),
                to: &prompt)
        }
        prompt += Token.chatMLTurnOpen + "assistant\n" + Token.qwenThink
            + (family == .lfm ? "" : "\n")
        return prompt
    }

    private static func appendTurn(role: Role, content: String, to prompt: inout String) {
        prompt += Token.turnOpen + role.rawValue + "\n" + content + Token.turnClose + "\n"
    }

    private static func appendChatMLTurn(
        role: String,
        content: String,
        to prompt: inout String
    ) {
        prompt += Token.chatMLTurnOpen + role + "\n" + content
            + Token.chatMLTurnClose + "\n"
    }
}
