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
    private var activeRequestToken: UInt64 = 0
    @ObservationIgnored
    var localEngineFactory: ((CatalogModel) -> ChatEngine)?
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

    /// Loaded local model id, if any (includes downloading/loading).
    var loadedModelID: String? {
        switch state {
        case .empty: return nil
        case .downloading(let id), .loading(let id, _), .ready(let id), .generating(let id): return id
        }
    }

    /// Resident local model id currently held in RAM (`.ready` or `.generating` only).
    var residentModelID: String? {
        switch state {
        case .ready(let id), .generating(let id): return id
        case .empty, .downloading, .loading: return nil
        }
    }

    /// Resolve an engine, downloading and loading lazily. Blocks until ready.
    func engine(for id: String) async throws -> ChatEngine {
        if id != appleIntelligenceEngineID,
           isGenerating,
           let current = modelEngine,
           current.id != id {
            throw EngineError.unavailable("Stop the current response before switching models.")
        }

        activeRequestToken &+= 1
        let requestToken = activeRequestToken

        if id == appleIntelligenceEngineID {
            if let message = AppleIntelligenceEngine.availabilityMessage() {
                throw EngineError.unavailable(message)
            }
            // Switching to Apple Intelligence releases any resident local weights
            // — except mid-generation, where the stream still needs them (the
            // idle auto-unload or a later resolve reclaims them after).
            if modelEngine != nil, !isGenerating { unloadSlotPreservingRequestToken() }
            return appleEngine
        }

        guard let catalogModel = modelStore.model(for: id) else {
            throw EngineError.generationFailed("Unknown model: \(id)")
        }

        // One local model backend in RAM at a time. Preserve `requestToken`
        // (already bumped above) so this in-call unload does not cancel itself.
        if let current = modelEngine, current.id != id, !isGenerating {
            unloadSlotPreservingRequestToken()
        }

        if !modelStore.isAvailable(id) {
            state = .downloading(id: id)
            do {
                try await modelStore.downloadAndWait(id)
            } catch {
                if activeRequestToken == requestToken {
                    state = .empty
                }
                throw error
            }
            guard activeRequestToken == requestToken else {
                throw CancellationError()
            }
        }

        // Re-check after suspension so an active local stream cannot be clobbered.
        if isGenerating, let current = modelEngine, current.id != id {
            throw EngineError.unavailable("Stop the current response before switching models.")
        }

        // MLX-only & MLX-streamed preflight: the TurboFieldfare backend validates its `.gturbo`
        // manifest instead. Dense and imported MLX behavior stays unchanged.
        if (catalogModel.backend == .mlx || catalogModel.backend == .mlxStreamed),
           let reason = modelStore.unsupportedReason(for: id) {
            if activeRequestToken == requestToken {
                state = .empty
            }
            throw EngineError.unavailable("\(catalogModel.displayName) can't run in this build: \(reason).")
        }

        let engine: ChatEngine
        if let current = modelEngine, current.id == id {
            engine = current
        } else {
            if modelEngine != nil, !isGenerating {
                unloadSlotPreservingRequestToken()
            }
            if let localEngineFactory {
                engine = localEngineFactory(catalogModel)
            } else {
                switch catalogModel.backend {
                case .mlx:
                    engine = MLXEngine(model: catalogModel)
                case .mlxStreamed:
                    engine = MLXEngine(model: catalogModel, streamingSettings: .current())
                case .streamedMoE:
                    guard let directory = modelStore.availableDirectory(for: id) else {
                        if activeRequestToken == requestToken {
                            state = .empty
                        }
                        throw EngineError.unavailable(
                            "Choose this model’s completed .gturbo directory in Manage Models.")
                    }
                    engine = StreamedMoEEngine(model: catalogModel, modelDirectory: directory)
                }
            }
            modelEngine = engine
        }

        if !engine.isLoaded {
            if case .loading(let loadingID, _) = state, loadingID == id {
                // Keep existing progress when another caller awaits the same load.
            } else {
                state = .loading(id: id, fraction: 0)
            }
            do {
                try await engine.load { [weak self] fraction in
                    Task { @MainActor [weak self] in
                        guard let self,
                              self.activeRequestToken == requestToken,
                              self.modelEngine === engine,
                              case .loading(let currentID, _) = self.state,
                              currentID == id
                        else { return }
                        self.state = .loading(id: id, fraction: fraction)
                    }
                }
            } catch {
                loadDidFail(engine)
                throw error
            }
        }

        guard activeRequestToken == requestToken else {
            throw CancellationError()
        }
        try loadDidSucceed(engine, id: id)
        return engine
    }

    /// Places `engine` in the single RAM slot without loading weights.
    func installSlot(_ engine: ChatEngine) {
        activeRequestToken &+= 1
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
        guard modelEngine === engine, engine.id == id else { throw CancellationError() }
        state = .ready(id: id)
        lastUsedAt = .now
    }

    func beginGenerating(id: String) {
        guard id != appleIntelligenceEngineID else { return }
        if modelEngine?.id == id {
            state = .generating(id: id)
        }
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
        activeRequestToken &+= 1
        unloadSlotPreservingRequestToken()
    }

    private func unloadSlotPreservingRequestToken() {
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
              modelStore.model(for: id)?.backend.isSSDStreamed == true else { return }
        unloadCurrent()
    }

    /// Called when a model's files are deleted from disk.
    func modelDeleted(_ id: String) {
        guard let loaded = loadedModelID else { return }
        if loaded == id
            || MLXStreamedInstall.variant(forStreamedModelID: loaded)?.sourceModelID == id
        {
            unloadCurrent()
        }
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
