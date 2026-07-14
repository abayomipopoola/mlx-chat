import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// One MLX model held in RAM, with a KV-cached ChatSession per conversation.
final class MLXEngine: ChatEngine {
    let model: CatalogModel
    var id: String { model.id }

    private var container: ModelContainer?
    private var session: ChatSession?
    private var sessionConversationID: UUID?
    private var sessionSystemPrompt: String?
    /// Number of prior (non-pending) messages the session has incorporated.
    /// A mismatch (regenerate, delete, cancel mid-stream) forces a session rebuild.
    private var sessionKnownPriorCount = 0
    /// Deduplicates concurrent load requests (switch-warming + first send) so the
    /// multi-GB weights are never materialized twice.
    private var inFlightLoad: Task<Void, Error>?

    init(model: CatalogModel) {
        self.model = model
    }

    var isLoaded: Bool { container != nil }

    func load(progress: @escaping @Sendable (Double) -> Void) async throws {
        guard container == nil else { return }
        // A load is already running (e.g. switch-warming): join it instead of
        // loading the same weights a second time.
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
        // Local-first: a complete on-disk copy (plain ~/Models/<org>/<name> layout or a
        // hub snapshot) loads offline. A `.directory` configuration resolves without the
        // downloader AND keeps extraEOSTokens — the bare `loadContainer(from: URL, …)`
        // overload would drop them and Qwen models would run past <|im_end|>.
        let configuration: ModelConfiguration
        if let localDirectory = ModelStore.localModelDirectory(for: model.id) {
            configuration = ModelConfiguration(
                directory: localDirectory, extraEOSTokens: model.extraEOSTokens)
        } else {
            configuration = ModelConfiguration(id: model.id, extraEOSTokens: model.extraEOSTokens)
        }

        do {
            let loaded = try await LLMModelFactory.shared.loadContainer(
                from: HubDownloader(ModelStore.hubClient),
                using: #huggingFaceTokenizerLoader(),
                configuration: configuration,
                progressHandler: { progress($0.fractionCompleted) })
            // Unloaded (model switch) while we were loading: drop the weights
            // instead of resurrecting a container nobody owns.
            try Task.checkCancellation()
            container = loaded
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw Self.mapLoadError(error)
        }
        // Keep MLX's buffer cache small so idle RAM stays low.
        Memory.cacheLimit = 20 * 1024 * 1024
    }

    func unload() {
        inFlightLoad?.cancel()
        inFlightLoad = nil
        session = nil
        sessionConversationID = nil
        sessionSystemPrompt = nil
        sessionKnownPriorCount = 0
        container = nil
        Memory.clearCache()
    }

    private static func mapLoadError(_ error: Error) -> Error {
        let text = String(describing: error)
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
            let task = Task { [self] in
                do {
                    guard let container else {
                        throw EngineError.generationFailed("Model is not loaded.")
                    }
                    let trimmed = PromptBuilder.trimmed(history: history, ctxTokens: model.ctxTokens)
                    guard let pending = trimmed.last, pending.role == .user else {
                        throw EngineError.generationFailed("No pending user message.")
                    }
                    let prior = Array(trimmed.dropLast())

                    let needsRebuild = session == nil
                        || sessionConversationID != conversationID
                        || sessionSystemPrompt != systemPrompt
                        || sessionKnownPriorCount != prior.count
                    if needsRebuild {
                        var params = GenerateParameters(temperature: Float(temperature))
                        // Safety cap: bound runaway generation. Small quantized reasoning
                        // models (e.g. Gemma 4 12B 4-bit) can loop in their thinking phase
                        // for thousands of tokens without closing it. Normal replies are far
                        // shorter, so this only stops a true runaway from generating forever.
                        params.maxTokens = 8192
                        session = ChatSession(
                            container,
                            instructions: systemPrompt,
                            history: prior.map {
                                Chat.Message(role: $0.role == .user ? .user : .assistant, content: $0.text)
                            },
                            generateParameters: params)
                        sessionConversationID = conversationID
                        sessionSystemPrompt = systemPrompt
                        sessionKnownPriorCount = prior.count
                    } else {
                        session?.generateParameters.temperature = Float(temperature)
                    }
                    guard let session else {
                        throw EngineError.generationFailed("Chat session unavailable.")
                    }

                    // Reasoning routing (Qwen3 / Ternary Bonsai / DeepSeek-R1 / Gemma 4). For toggleable
                    // models `enable_thinking: false` makes the chat template pre-close the
                    // reasoning span so the model answers directly — applied per turn via
                    // the template context, no session rebuild needed.
                    let thinkingEnabled = UserDefaults.standard.object(forKey: Keys.thinkingEnabled)
                        as? Bool ?? Keys.Defaults.thinkingEnabled
                    var filter = ThinkTagFilter()
                    if let thinking = model.thinking {
                        let active = !thinking.toggleable || thinkingEnabled
                        if thinking.toggleable {
                            session.additionalContext = ["enable_thinking": thinkingEnabled]
                        }
                        // With thinking off the template pre-closes the span, so the stream
                        // is a plain answer — passthrough (the default filter) is correct.
                        if active {
                            filter = ThinkTagFilter(
                                open: thinking.open, close: thinking.close,
                                startsInside: thinking.startsInside, mayEmit: !thinking.startsInside)
                        }
                    }
                    var tokensPerSecond: Double?

                    for try await event in session.streamDetails(to: pending.text) {
                        try Task.checkCancellation()
                        switch event {
                        case .chunk(let text):
                            let routed = filter.consume(text)
                            if !routed.thinking.isEmpty { continuation.yield(.thinkingDelta(routed.thinking)) }
                            if !routed.content.isEmpty { continuation.yield(.delta(routed.content)) }
                        case .info(let info):
                            tokensPerSecond = info.tokensPerSecond
                        case .toolCall:
                            break
                        }
                    }

                    let flush = filter.finish()
                    if flush.reclassified {
                        // Think block never closed: everything is the answer.
                        if !flush.content.isEmpty { continuation.yield(.delta(flush.content)) }
                        continuation.yield(.thinkingReclassified)
                    } else {
                        if !flush.thinking.isEmpty { continuation.yield(.thinkingDelta(flush.thinking)) }
                        if !flush.content.isEmpty { continuation.yield(.delta(flush.content)) }
                    }

                    // Session has now incorporated this user/assistant turn.
                    sessionKnownPriorCount += 2
                    continuation.yield(.completed(tokensPerSecond: tokensPerSecond))
                    continuation.finish()
                } catch is CancellationError {
                    // Partial output stays; the count mismatch rebuilds the session next turn.
                    session = nil
                    continuation.yield(.completed(tokensPerSecond: nil))
                    continuation.finish()
                } catch let error as EngineError {
                    session = nil
                    continuation.finish(throwing: error)
                } catch {
                    session = nil
                    continuation.finish(throwing: EngineError.generationFailed(error.localizedDescription))
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
