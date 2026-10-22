import Darwin
import Foundation
import Testing
@testable import MLXChat

/// Live app-path benchmark for the SSD-streamed engines: load time, then two
/// new conversations with the app's real system prompt, each timed to the
/// first streamed event and reporting the engine's decode rate. Opt-in
/// (`MLXCHAT_RUN_SSD_APP_BENCH=1`); `MLXCHAT_SSD_BENCH_MODELS` picks variants
/// by key (flint, lfm25, qat, ornith15, standard, qwen36), default all installed;
/// `MLXCHAT_SSD_BENCH_IDLE` waits that many seconds after load, as typing would.
@Suite(.serialized) struct StreamedMoEAppBenchmarkTests {
    private static let environment = ProcessInfo.processInfo.environment
    private static let variants: [String: StreamedMoEVariant] = [
        "flint": StreamedMoEInstall.flint, "lfm25": StreamedMoEInstall.lfm25,
        "qat": StreamedMoEInstall.qat, "ornith15": StreamedMoEInstall.ornith15,
        "standard": StreamedMoEInstall.standard, "qwen36": StreamedMoEInstall.qwen36,
    ]

    /// Physical footprint now and the process peak, in GB (what Activity
    /// Monitor's Memory column shows).
    static func footprintGB() -> (now: Double, peak: Double) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return (0, 0) }
        return (Double(info.phys_footprint) / 1e9, Double(info.ledger_phys_footprint_peak) / 1e9)
    }

    private struct Turn {
        var firstEventSeconds: Double?
        var events = 0
        var tokensPerSecond: Double?
        var answer = ""
        var prefillEvents = 0
        var prefillRises = true
        var lastPrefill = (done: 0, total: 0)
    }

    private static func run(_ engine: StreamedMoEEngine, question: String) async throws -> Turn {
        var turn = Turn()
        let started = Date()
        for try await event in engine.stream(
            history: [EngineMessage(role: .user, text: question)],
            systemPrompt: PromptBuilder.currentSystemPrompt(),
            temperature: 0,
            conversationID: UUID()
        ) {
            switch event {
            case .delta(let text):
                turn.answer += text
                fallthrough
            case .thinkingDelta:
                turn.events += 1
                if turn.firstEventSeconds == nil {
                    turn.firstEventSeconds = Date().timeIntervalSince(started)
                }
            case .thinkingReclassified:
                break
            case .prefill(let done, let total):
                turn.prefillRises = turn.prefillRises && done >= turn.lastPrefill.done && done <= total
                turn.lastPrefill = (done, total)
                turn.prefillEvents += 1
            case .completed(let rate):
                turn.tokensPerSecond = rate
            }
        }
        return turn
    }

    @Test(.timeLimit(.minutes(60)))
    func loadFirstTokenAndDecodePerModel() async throws {
        guard Self.environment["MLXCHAT_RUN_SSD_APP_BENCH"] == "1" else { return }
        let keys = (Self.environment["MLXCHAT_SSD_BENCH_MODELS"] ?? "flint,lfm25,qat,ornith15,standard,qwen36")
            .split(separator: ",").map(String.init)
        for key in keys {
            guard let variant = Self.variants[key],
                  FileManager.default.fileExists(atPath: variant.managedDirectory
                    .appendingPathComponent("manifest.json").path),
                  let catalog = ModelCatalog.model(for: variant.modelID)
            else { continue }
            // `MLXCHAT_SSD_BENCH_USER_SETTINGS=1` loads with the app's saved
            // context and cache settings instead of 4K and 16 slots.
            // `MLXCHAT_SSD_BENCH_SLOTS` overrides the cache slot count.
            let saved = Self.environment["MLXCHAT_SSD_BENCH_USER_SETTINGS"] == "1"
                ? StreamedMoERuntimeSettings.current()
                : StreamedMoERuntimeSettings(contextTokens: 4_096, expertCacheSlots: 16)
            let settings = StreamedMoERuntimeSettings(
                contextTokens: saved.contextTokens,
                expertCacheSlots: Int(Self.environment["MLXCHAT_SSD_BENCH_SLOTS"] ?? "")
                    ?? saved.expertCacheSlots)
            let before = Self.footprintGB()
            let engine = StreamedMoEEngine(
                model: catalog, modelDirectory: variant.managedDirectory,
                runtimeSettings: settings)
            let loadStart = Date()
            try await engine.load(progress: { _ in })
            let loadSeconds = Date().timeIntervalSince(loadStart)
            // Seconds a person would spend typing before the first send.
            if let idle = Double(Self.environment["MLXCHAT_SSD_BENCH_IDLE"] ?? ""), idle > 0 {
                try await Task.sleep(for: .seconds(idle))
            }
            let loaded = Self.footprintGB()
            let first = try await Self.run(engine, question: "What is 17 plus 25? Answer with the number only.")
            let second = try await Self.run(engine, question: "What is the capital of France? Answer in one word.")
            let used = Self.footprintGB()
            // `MLXCHAT_SSD_BENCH_HOLD` keeps the loaded model this many seconds
            // (and as long again after unload) for outside memory tools.
            let hold = Double(Self.environment["MLXCHAT_SSD_BENCH_HOLD"] ?? "") ?? 0
            if hold > 0 {
                print("SSD_APP_BENCH hold pid \(getpid()) loaded")
                try await Task.sleep(for: .seconds(hold))
            }
            engine.unload()
            try await Task.sleep(for: .seconds(max(1, hold)))
            let after = Self.footprintGB()
            func line(_ turn: Turn) -> String {
                String(format: "first event %.2f s, %d events, %.2f tok/s, prefill events %d (last %d/%d, rising %@)",
                       turn.firstEventSeconds ?? -1, turn.events, turn.tokensPerSecond ?? -1,
                       turn.prefillEvents, turn.lastPrefill.done, turn.lastPrefill.total,
                       turn.prefillRises ? "yes" : "no")
            }
            print("""
                SSD_APP_BENCH \(key): load \(String(format: "%.2f", loadSeconds)) s, \
                context \(settings.contextTokens) slots \(settings.expertCacheSlots), \
                footprint GB before \(String(format: "%.2f", before.now)) loaded \(String(format: "%.2f", loaded.now)) \
                after chats \(String(format: "%.2f", used.now)) unloaded \(String(format: "%.2f", after.now)) \
                process peak \(String(format: "%.2f", used.peak))
                SSD_APP_BENCH \(key) chat 1: \(line(first)) answer \(first.answer.prefix(60).debugDescription)
                SSD_APP_BENCH \(key) chat 2: \(line(second)) answer \(second.answer.prefix(60).debugDescription)
                """)
            #expect(first.events > 0 && second.events > 0)
            #expect(first.prefillRises && second.prefillRises)
        }
    }
}
