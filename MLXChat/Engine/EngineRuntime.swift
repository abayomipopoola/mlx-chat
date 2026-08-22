import Foundation
import Observation

/// Owns the single in-RAM model slot: lazy download + load, engine switching,
/// generation state, and idle auto-unload.
@MainActor
@Observable
final class EngineRuntime {
    enum EngineState: Equatable {
        case empty
        case downloading(id: String)
        case loading(id: String, fraction: Double)
        case ready(id: String)
        case generating(id: String)

        /// True while this model is still being fetched or pulled into RAM.
        func isPreparing(_ modelID: String) -> Bool {
            switch self {
            case .downloading(let id), .loading(let id, _):
                return id == modelID
            case .empty, .ready, .generating:
                return false
            }
        }
    }

    private(set) var state: EngineState = .empty

    /// UI-verification hook: hold a mid-load state for screenshots.
    var effectiveState: EngineState {
#if DEBUG
        let arguments = ProcessInfo.processInfo.arguments
        if let index = arguments.firstIndex(of: "--fake-loading-model"),
           index + 1 < arguments.count {
            return .loading(id: arguments[index + 1], fraction: 0.45)
        }
#endif
        return state
    }
    private(set) var lastUsedAt = Date.now

    private let modelStore: ModelStore
    private let appleEngine = AppleIntelligenceEngine()
    private var modelEngine: ChatEngine?
    /// nonisolated(unsafe): written once in init, read in (nonisolated) deinit.
    private nonisolated(unsafe) var unloadTimer: Timer?

    init(modelStore: ModelStore) {
        self.modelStore = modelStore
        unloadTimer = Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in self?.autoUnloadTick() }
        }
    }

    deinit {
        // The run loop retains scheduled timers; without this a discarded
        // runtime would leave its timer firing forever.
        unloadTimer?.invalidate()
    }

    /// Loaded local model id, if any.
    var loadedModelID: String? {
        switch state {
        case .empty: return nil
        case .downloading(let id), .loading(let id, _), .ready(let id), .generating(let id): return id
        }
    }

    /// Resolve an engine, downloading and loading lazily. Blocks until ready.
    func engine(for id: String) async throws -> ChatEngine {
        if id == appleIntelligenceEngineID {
            if let message = AppleIntelligenceEngine.availabilityMessage() {
                throw EngineError.unavailable(message)
            }
            // Switching to Apple Intelligence releases any resident local weights
            // — except mid-generation, where the stream still needs them (the
            // idle auto-unload or a later resolve reclaims them after).
            if modelEngine != nil, !isGenerating { unloadCurrent() }
            return appleEngine
        }

        guard let catalogModel = modelStore.model(for: id) else {
            throw EngineError.generationFailed("Unknown model: \(id)")
        }

        // One local model backend in RAM at a time.
        if let current = modelEngine, current.id != id {
            unloadCurrent()
        }

        if !modelStore.isAvailable(id) {
            state = .downloading(id: id)
            do {
                try await modelStore.downloadAndWait(id)
            } catch {
                state = .empty
                throw error
            }
        }

        // MLX-only preflight: the streamed MoE backend validates its `.gturbo`
        // manifest instead. Dense and imported MLX behavior stays unchanged.
        if catalogModel.backend == .mlx,
           let reason = ModelSupport.unsupportedReason(modelID: id) {
            state = .empty
            throw EngineError.unavailable("\(catalogModel.displayName) can't run in this build: \(reason).")
        }

        let engine: ChatEngine
        if let current = modelEngine {
            engine = current
        } else {
            switch catalogModel.backend {
            case .mlx:
                engine = MLXEngine(model: catalogModel)
            case .streamedMoE:
                guard let directory = modelStore.availableDirectory(for: id) else {
                    state = .empty
                    throw EngineError.unavailable(
                        "Choose this model’s completed .gturbo directory in Manage Models.")
                }
                engine = StreamedMoEEngine(model: catalogModel, modelDirectory: directory)
            }
            modelEngine = engine
        }

        if !engine.isLoaded {
            state = .loading(id: id, fraction: 0)
            do {
                try await engine.load { [weak self] fraction in
                    Task { @MainActor [weak self] in
                        if case .loading = self?.state {
                            self?.state = .loading(id: id, fraction: fraction)
                        }
                    }
                }
            } catch {
                loadDidFail(engine)
                throw error
            }
        }

        try loadDidSucceed(engine, id: id)
        return engine
    }

    /// Places `engine` in the single RAM slot without loading weights.
    func installSlot(_ engine: ChatEngine) {
        modelEngine = engine
        state = .loading(id: engine.id, fraction: 0)
    }

    /// A cancelled or failed load clears the slot only while it still owns it.
    func loadDidFail(_ engine: ChatEngine) {
        if modelEngine === engine {
            modelEngine = nil
            state = .empty
        }
    }

    /// Stale success must not stamp `.ready` for a prior model over a successor.
    func loadDidSucceed(_ engine: ChatEngine, id: String) throws {
        guard modelEngine === engine else { throw CancellationError() }
        state = .ready(id: id)
        lastUsedAt = .now
    }

    func beginGenerating(id: String) {
        if id != appleIntelligenceEngineID { state = .generating(id: id) }
    }

    func endGenerating() {
        if case .generating(let id) = state { state = .ready(id: id) }
        lastUsedAt = .now
    }

    /// True while a local model owns the in-RAM slot for an active stream.
    private var isGenerating: Bool {
        if case .generating = state { return true }
        return false
    }

    func unloadCurrent() {
        modelEngine?.unload()
        modelEngine = nil
        state = .empty
    }

    /// Runtime settings shape the KV buffers and expert cache at load time.
    /// Drop only a resident streamed engine so its next load uses new values;
    /// dense MLX and Apple Intelligence remain untouched.
    func streamedModelSettingsDidChange() {
        guard !isGenerating,
              let id = loadedModelID,
              modelStore.model(for: id)?.backend == .streamedMoE else { return }
        unloadCurrent()
    }

    /// Called when a model's files are deleted from disk.
    func modelDeleted(_ id: String) {
        if loadedModelID == id { unloadCurrent() }
    }

    /// Frees RAM after the configured idle interval (0 = never).
    private func autoUnloadTick() {
        let minutes = UserDefaults.standard.object(forKey: Keys.autoUnloadMinutes) as? Int
            ?? Keys.Defaults.autoUnloadMinutes
        guard minutes > 0, case .ready = state else { return }
        guard Date.now.timeIntervalSince(lastUsedAt) > Double(minutes) * 60 else { return }
        unloadCurrent()
    }
}
