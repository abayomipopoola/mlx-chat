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
    }

    private(set) var state: EngineState = .empty
    private(set) var lastUsedAt = Date.now

    private let modelStore: ModelStore
    private let appleEngine = AppleIntelligenceEngine()
    private var mlxEngine: MLXEngine?
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

    /// Loaded MLX model id, if any.
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
            return appleEngine
        }

        guard let catalogModel = modelStore.model(for: id) else {
            throw EngineError.generationFailed("Unknown model: \(id)")
        }

        // One MLX model in RAM at a time.
        if let current = mlxEngine, current.id != id {
            unloadCurrent()
        }

        if !modelStore.isDownloaded(id) {
            state = .downloading(id: id)
            do {
                try await modelStore.downloadAndWait(id)
            } catch {
                state = .empty
                throw error
            }
        }

        let engine = mlxEngine ?? MLXEngine(model: catalogModel)
        mlxEngine = engine

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
                mlxEngine = nil
                state = .empty
                throw error
            }
        }

        state = .ready(id: id)
        lastUsedAt = .now
        return engine
    }

    func beginGenerating(id: String) {
        if id != appleIntelligenceEngineID { state = .generating(id: id) }
    }

    func endGenerating() {
        if case .generating(let id) = state { state = .ready(id: id) }
        lastUsedAt = .now
    }

    func unloadCurrent() {
        mlxEngine?.unload()
        mlxEngine = nil
        state = .empty
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
