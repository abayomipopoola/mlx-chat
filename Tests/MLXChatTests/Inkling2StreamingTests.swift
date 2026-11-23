import Darwin
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import XCTest
@testable import MLXChat

final class Inkling2StreamingTests: XCTestCase {

    private final class SyncBox<Value>: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Value
        init(_ value: Value) { self.value = value }
        func withLock<R>(_ body: (inout Value) -> R) -> R {
            lock.withLock { body(&value) }
        }
        func read() -> Value {
            lock.withLock { value }
        }
    }

    // MARK: - Cycle 1 Offline Unit Tests (Routing Observer Hook)

    func testRoutingObserverVisitOrder() throws {
        // 1. Default nil, parameter-key invariance, fused GPU kernel prefill + decode branches.
        let model = try Inkling2ModelTests.makeModel(bfloat16: true)
        XCTAssertNil(model.routingObserver)

        let baselineKeys = Set(model.parameters().flattened().map(\.0))
        model.routingObserver = { _, _, _, _, _ in }
        let observedKeys = Set(model.parameters().flattened().map(\.0))
        XCTAssertEqual(baselineKeys, observedKeys)
        model.routingObserver = nil

        let textConfig = model.config.textConfiguration
        let expectedVisits = Self.expectedVisitSchedule(textConfig)
        XCTAssertEqual(expectedVisits.count, textConfig.layerVisits.count)
        XCTAssertEqual(expectedVisits.map(\.layer), textConfig.layerVisits)

        let promptTokens = MLXArray((0 ..< 6).map { Int32(($0 * 5 + 1) % 48) }).reshaped(1, 6)
        let decodeTokens = [Int32(7), Int32(19), Int32(31)]

        // Baseline run with observer == nil
        let nilCache = try model.newCache(parameters: nil)
        let nilPrefill = model(promptTokens, cache: nilCache)
        eval(nilPrefill)
        var nilDecodeLogits: [MLXArray] = []
        for tok in decodeTokens {
            let step = model(MLXArray([tok]).reshaped(1, 1), cache: nilCache)
            eval(step)
            nilDecodeLogits.append(step)
        }

        // Observed run calling eval(indices) on every MoE invocation
        struct RecordedCall: Sendable {
            let visit: Int
            let layer: Int
            let pass: Int
            let isDecode: Bool
            let lastDim: Int
            let values: [UInt32]
        }
        let recordedBox = SyncBox<[RecordedCall]>([])
        model.routingObserver = { visit, layer, pass, indices, isDecode in
            eval(indices)
            let vals = indices.asType(.uint32).asArray(UInt32.self)
            let lastDim = indices.dim(-1)
            recordedBox.withLock { arr in
                arr.append(
                    RecordedCall(
                        visit: visit, layer: layer, pass: pass, isDecode: isDecode,
                        lastDim: lastDim, values: vals))
            }
        }
        defer { model.routingObserver = nil }

        let obsCache = try model.newCache(parameters: nil)
        let obsPrefill = model(promptTokens, cache: obsCache)
        eval(obsPrefill)
        var obsDecodeLogits: [MLXArray] = []
        for tok in decodeTokens {
            let step = model(MLXArray([tok]).reshaped(1, 1), cache: obsCache)
            eval(step)
            obsDecodeLogits.append(step)
        }
        let recorded = recordedBox.read()

        // Total calls = 1 prefill + 3 decode steps = 4 * expectedVisits.count
        XCTAssertEqual(recorded.count, 4 * expectedVisits.count)

        let prefillCalls = Array(recorded.prefix(expectedVisits.count))
        for (idx, call) in prefillCalls.enumerated() {
            XCTAssertEqual(call.visit, expectedVisits[idx].visit)
            XCTAssertEqual(call.layer, expectedVisits[idx].layer)
            XCTAssertEqual(call.pass, expectedVisits[idx].pass)
            XCTAssertFalse(call.isDecode)
            XCTAssertEqual(call.lastDim, textConfig.expertsPerToken)
            XCTAssertEqual(call.values.count, 6 * textConfig.expertsPerToken)
            XCTAssertTrue(call.values.allSatisfy { Int($0) < textConfig.experts })
        }

        for stepIdx in 0 ..< decodeTokens.count {
            let sliceStart = (stepIdx + 1) * expectedVisits.count
            let stepCalls = Array(recorded[sliceStart ..< (sliceStart + expectedVisits.count)])
            for (idx, call) in stepCalls.enumerated() {
                XCTAssertEqual(call.visit, expectedVisits[idx].visit)
                XCTAssertEqual(call.layer, expectedVisits[idx].layer)
                XCTAssertEqual(call.pass, expectedVisits[idx].pass)
                XCTAssertTrue(call.isDecode)
                XCTAssertEqual(call.lastDim, textConfig.expertsPerToken)
                XCTAssertEqual(call.values.count, textConfig.expertsPerToken)
                XCTAssertTrue(call.values.allSatisfy { Int($0) < textConfig.experts })
            }
        }

        let prefillDiff = abs(nilPrefill.asType(.float32) - obsPrefill.asType(.float32)).max().item(Float.self)
        XCTAssertEqual(prefillDiff, 0)
        for (a, b) in zip(nilDecodeLogits, obsDecodeLogits) {
            let diff = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
            XCTAssertEqual(diff, 0)
        }

        // 2. Non-kernel prefill branch (`guard convolution.usesKernel(mask: mask)` else path)
        // triggered via sconv_kernel_size = 2 while still running on GPU.
        let nonKernelModel = try Self.makeTinyModel(bfloat16: true, convKernelSize: 2)
        let nkNilCache = try nonKernelModel.newCache(parameters: nil)
        let nkNilPrefill = nonKernelModel(promptTokens, cache: nkNilCache)
        eval(nkNilPrefill)

        let nkRecordedBox = SyncBox<[RecordedCall]>([])
        nonKernelModel.routingObserver = { visit, layer, pass, indices, isDecode in
            eval(indices)
            let vals = indices.asType(.uint32).asArray(UInt32.self)
            let lastDim = indices.dim(-1)
            nkRecordedBox.withLock { arr in
                arr.append(
                    RecordedCall(
                        visit: visit, layer: layer, pass: pass, isDecode: isDecode,
                        lastDim: lastDim, values: vals))
            }
        }
        defer { nonKernelModel.routingObserver = nil }

        let nkObsCache = try nonKernelModel.newCache(parameters: nil)
        let nkObsPrefill = nonKernelModel(promptTokens, cache: nkObsCache)
        eval(nkObsPrefill)
        let nkRecorded = nkRecordedBox.read()

        XCTAssertEqual(nkRecorded.count, expectedVisits.count)
        for (idx, call) in nkRecorded.enumerated() {
            XCTAssertEqual(call.visit, expectedVisits[idx].visit)
            XCTAssertEqual(call.layer, expectedVisits[idx].layer)
            XCTAssertEqual(call.pass, expectedVisits[idx].pass)
            XCTAssertFalse(call.isDecode)
            XCTAssertEqual(call.lastDim, textConfig.expertsPerToken)
            XCTAssertTrue(call.values.allSatisfy { Int($0) < textConfig.experts })
        }
        let nkDiff = abs(nkNilPrefill.asType(.float32) - nkObsPrefill.asType(.float32)).max().item(Float.self)
        XCTAssertEqual(nkDiff, 0)

        print("INKLING2_OBSERVER_TEST: kernelPrefill=true nonKernelPrefill=true decode=true visits=\(expectedVisits.count) maxDiff=\(prefillDiff)")
    }

    // MARK: - Cycle 1 Opt-In Check (a) Routing & I/O Benchmark

    func testCheckARoutingBenchmark() async throws {
        guard let benchDir = Self.envValue("INKLING2_BENCH_DIR"), !benchDir.isEmpty else {
            return
        }
        let modelURL = URL(fileURLWithPath: benchDir, isDirectory: true)
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: modelURL.appendingPathComponent("config.json").path),
            "Missing config.json at \(modelURL.path)")

        let nTokens = Self.envValue("INKLING2_BENCH_TOKENS").flatMap(Int.init) ?? 300
        let overheadTokens = Self.envValue("INKLING2_BENCH_OVERHEAD_TOKENS").flatMap(Int.init) ?? 32
        let traceBackendLabel = "ssd_streamed_32_chunk512"
        let outPath = Self.envValue("INKLING2_BENCH_OUT")
            ?? FileManager.default.temporaryDirectory
                .appendingPathComponent("inkling2-check-a.json")
                .path

        let safetensorsIndex = try Self.parseSafetensorsExperts(in: modelURL)
        let configData = try Data(contentsOf: modelURL.appendingPathComponent("config.json"))
        let modelConfig = try JSONDecoder.json5().decode(Inkling2Configuration.self, from: configData)
        let textConfig = modelConfig.textConfiguration
        let visitSchedule = Self.expectedVisitSchedule(textConfig)
        let visitCount = visitSchedule.count

        await Inkling2.register()

        let previousCacheLimit = Memory.cacheLimit
        defer {
            Memory.cacheLimit = previousCacheLimit
            Memory.clearCache()
        }

        let benchStartNs = DispatchTime.now().uptimeNanoseconds
        func logProgress(_ message: String) {
            let elapsedSec = Double(DispatchTime.now().uptimeNanoseconds &- benchStartNs) / 1_000_000_000.0
            let activeMB = Memory.activeMemory / 1_048_576
            print("INKLING2_CHECK_A progress t=\(String(format: "%.1fs", elapsedSec)) activeMB=\(activeMB) \(message)")
            fflush(stdout)
        }

        let configuration = ModelConfiguration(
            directory: modelURL,
            extraEOSTokens: ["<|content_model_end_sampling|>"])

        // Phase 1: Collect prefill & decode routing traces on SSD-streamed mode (slots=32, chunk=512).
        // Running streamed before resident avoids host swap thrashing AND keeps expert tensors out of OS page cache.
        var prefillChunkReports: [PrefillUniqueReport] = []
        var rawDecodeTraces: [(name: String, decodeTokens: [[[UInt32]]])] = []
        var prosePromptIDs: [Int] = []

        do {
            Memory.cacheLimit = MLXEngine.effectiveBufferCacheLimit(streaming: true)
            let streamCfg = Inkling2StreamingConfig(
                modelDirectory: modelURL,
                expertCacheSlots: 32,
                evictionPolicy: .sharedSLRU,
                prefillChunkSize: 512,
                maxConcurrentReads: 4,
                loopWeightedSlots: true)
            logProgress("phase1 loading streamed container (slots=32, chunk=512)...")
            let streamedContainer = try await Inkling2.withStreamingLoad(streamCfg) {
                try await VLMModelFactory.shared.loadContainer(
                    from: HubDownloader(ModelStore.hubClient),
                    using: HuggingFaceTokenizerLoader(),
                    configuration: configuration)
            }
            logProgress("phase1 loaded streamed container")

            let phase1Result = try await streamedContainer.perform { context -> ([PrefillUniqueReport], [(name: String, decodeTokens: [[[UInt32]]])], [Int]) in
                guard let inkling = context.model as? Inkling2 else {
                    XCTFail("Expected Inkling2 model instance, got \(type(of: context.model))")
                    throw BenchmarkError.unexpectedModelType
                }
                defer { inkling.routingObserver = nil }

                let workloads = Self.buildCheckAWorkloads(tokenizer: context.tokenizer)
                for workload in workloads {
                    XCTAssertGreaterThanOrEqual(workload.promptIDs.count, 1200, "Prompt \(workload.name) too short")
                    XCTAssertLessThanOrEqual(workload.promptIDs.count, 1800, "Prompt \(workload.name) too long")
                }

                var prefillReports: [PrefillUniqueReport] = []
                var traces: [(name: String, decodeTokens: [[[UInt32]]])] = []
                let prefillParams = PrefillParameters(stepSize: 512)

                for workload in workloads {
                    let promptArray = MLXArray(workload.promptIDs.map(Int32.init)).reshaped(1, workload.promptIDs.count)

                    struct TraceRecorderState {
                        var prefillPerVisit: [[UInt32]]
                        var decodeTokensTrace: [[[UInt32]]] = []
                        var currentDecodeToken: [[UInt32]]
                        var decodeActive = false
                        var sawWrapCheck = false
                    }
                    let traceBox = SyncBox(TraceRecorderState(
                        prefillPerVisit: Array(repeating: [UInt32](), count: visitCount),
                        currentDecodeToken: Array(repeating: [UInt32](), count: visitCount)))

                    inkling.routingObserver = { visit, _, _, indices, isDecode in
                        eval(indices)
                        let flat = indices.asType(.uint32).asArray(UInt32.self)
                        traceBox.withLock { state in
                            if !isDecode {
                                state.prefillPerVisit[visit].append(contentsOf: flat)
                            } else {
                                if state.decodeActive, visit == 0 {
                                    state.decodeTokensTrace.append(state.currentDecodeToken)
                                    state.currentDecodeToken = Array(repeating: [UInt32](), count: visitCount)
                                }
                                state.decodeActive = true
                                state.sawWrapCheck = true
                                state.currentDecodeToken[visit] = flat
                            }
                        }
                    }

                    var tokenCount = 0
                    var prefillMs = 0.0
                    var firstNs: UInt64 = 0
                    var lastNs: UInt64 = 0
                    do {
                        let cache = try inkling.newCache(parameters: GenerateParameters(maxKVSize: 4_096))
                        var iter = try TokenIterator(
                            input: LMInput(tokens: promptArray),
                            model: context.model,
                            cache: cache,
                            processor: nil,
                            sampler: ArgMaxSampler(),
                            prefill: prefillParams,
                            maxTokens: nTokens)
                        prefillMs = iter.promptPrefillTime * 1000.0
                        while iter.next() != nil {
                            let now = DispatchTime.now().uptimeNanoseconds
                            if firstNs == 0 { firstNs = now }
                            lastNs = now
                            tokenCount += 1
                        }
                        Stream.gpu.synchronize()
                    }
                    inkling.routingObserver = nil
                    if let streamErr = inkling.consumeStreamingError() {
                        throw streamErr
                    }

                    let (prefillPerVisit, decodeTokensTrace) = traceBox.withLock { state -> ([[UInt32]], [[[UInt32]]]) in
                        if state.sawWrapCheck, !state.currentDecodeToken.allSatisfy(\.isEmpty) {
                            state.decodeTokensTrace.append(state.currentDecodeToken)
                        }
                        return (state.prefillPerVisit, state.decodeTokensTrace)
                    }
                    Memory.clearCache()

                    let decodeMs = Double(lastNs &- firstNs) / 1_000_000.0
                    let tps = (tokenCount > 1 && decodeMs > 0) ? Double(tokenCount - 1) / (decodeMs / 1000.0) : 0
                    logProgress("phase1 workload=\(workload.name) prompt=\(workload.promptIDs.count) prefillMs=\(String(format: "%.1f", prefillMs)) tok=\(tokenCount) tps=\(String(format: "%.2f", tps))")

                    prefillReports.append(
                        Self.analyzePrefillChunks(
                            workload: workload.name,
                            promptTokens: workload.promptIDs.count,
                            topK: textConfig.expertsPerToken,
                            perVisitFlat: prefillPerVisit,
                            schedule: visitSchedule))
                    traces.append((name: workload.name, decodeTokens: decodeTokensTrace))
                }

                let firstPrompt = workloads.first(where: { $0.name == "prose" })?.promptIDs ?? workloads[0].promptIDs
                return (prefillReports, traces, firstPrompt)
            }

            prefillChunkReports = phase1Result.0
            rawDecodeTraces = phase1Result.1
            prosePromptIDs = phase1Result.2
            await streamedContainer.perform { ctx in (ctx.model as? Inkling2)?.releaseStreamingResources() }
        }
        Memory.clearCache()
        logProgress("phase1 released streamed container")

        // Phase 2: Locality analysis + uncontaminated single-threaded F_NOCACHE SSD I/O replay (before loading resident).
        let tempTracesForLocality = rawDecodeTraces.map {
            WorkloadTrace(name: $0.name, nilDecodeMsPerToken: 0, decodeTokens: $0.decodeTokens)
        }
        let localityReport = Self.analyzeLocality(
            traces: tempTracesForLocality,
            schedule: visitSchedule,
            textConfig: textConfig)
        logProgress("phase2 completed locality analysis")

        let proseTraceForIO = tempTracesForLocality.first(where: { $0.name == "prose" }) ?? tempTracesForLocality[0]
        let ioReplayRows: [IOReplayRow]
        if safetensorsIndex.allContiguous {
            ioReplayRows = try Self.runIOReplay(
                trace: proseTraceForIO,
                schedule: visitSchedule,
                textConfig: textConfig,
                expertIndex: safetensorsIndex)
        } else {
            ioReplayRows = []
        }
        logProgress("phase2 completed uncontaminated F_NOCACHE IO replay rows=\(ioReplayRows.count)")

        // Phase 3: Compact 64-token Resident 48-sync eval(indices) overhead probe.
        var overheadReport: OverheadReport!
        do {
            Memory.cacheLimit = MLXEngine.activeBufferCacheLimit()
            logProgress("phase3 loading resident container for 64-token 48-sync overhead probe...")
            let residentContainer = try await Inkling2.withStreamingLoad(nil) {
                try await VLMModelFactory.shared.loadContainer(
                    from: HubDownloader(ModelStore.hubClient),
                    using: HuggingFaceTokenizerLoader(),
                    configuration: configuration)
            }
            logProgress("phase3 loaded resident container")

            overheadReport = try await residentContainer.perform { context -> OverheadReport in
                guard let inkling = context.model as? Inkling2 else {
                    XCTFail("Expected Inkling2 model instance, got \(type(of: context.model))")
                    throw BenchmarkError.unexpectedModelType
                }
                defer { inkling.routingObserver = nil }

                let probePromptIDs = Array(prosePromptIDs.prefix(min(64, prosePromptIDs.count)))
                let promptArray = MLXArray(probePromptIDs.map(Int32.init)).reshaped(1, probePromptIDs.count)

                // 1. 8-token warmup (nil observer)
                inkling.routingObserver = nil
                do {
                    let warmCache = try inkling.newCache(parameters: GenerateParameters(maxKVSize: 4_096))
                    var warmIter = try TokenIterator(
                        input: LMInput(tokens: promptArray),
                        model: context.model,
                        cache: warmCache,
                        processor: nil,
                        sampler: ArgMaxSampler(),
                        maxTokens: 8)
                    while warmIter.next() != nil {}
                    Stream.gpu.synchronize()
                }
                Memory.clearCache()
                logProgress("phase3 completed 8-token warmup prompt=\(probePromptIDs.count)")

                // 2. Nil-observer timed run
                var nilTokenIDs: [Int] = []
                nilTokenIDs.reserveCapacity(overheadTokens)
                var nilPrefillMs = 0.0
                var nilFirstNs: UInt64 = 0
                var nilLastNs: UInt64 = 0
                do {
                    let nilCache = try inkling.newCache(parameters: GenerateParameters(maxKVSize: 4_096))
                    var nilIter = try TokenIterator(
                        input: LMInput(tokens: promptArray),
                        model: context.model,
                        cache: nilCache,
                        processor: nil,
                        sampler: ArgMaxSampler(),
                        maxTokens: overheadTokens)
                    nilPrefillMs = nilIter.promptPrefillTime * 1000.0
                    while let tok = nilIter.next() {
                        let now = DispatchTime.now().uptimeNanoseconds
                        if nilFirstNs == 0 { nilFirstNs = now }
                        nilLastNs = now
                        nilTokenIDs.append(tok)
                    }
                    Stream.gpu.synchronize()
                }
                let nilDecodeTotalMs = Double(nilLastNs &- nilFirstNs) / 1_000_000.0
                let nilDecodeMsPerToken = nilTokenIDs.count > 1
                    ? nilDecodeTotalMs / Double(nilTokenIDs.count - 1)
                    : 0
                let nilTokPerSec = nilDecodeMsPerToken > 0 ? 1000.0 / nilDecodeMsPerToken : 0
                Memory.clearCache()
                logProgress("phase3 nil probe prefillMs=\(String(format: "%.1f", nilPrefillMs)) decodeMs=\(String(format: "%.2f", nilDecodeMsPerToken)) tps=\(String(format: "%.2f", nilTokPerSec))")

                // 3. Observed timed run calling eval(indices) on every visit
                let decodeObsTokens = SyncBox(0)
                inkling.routingObserver = { visit, _, _, indices, isDecode in
                    eval(indices)
                    _ = indices.asType(.uint32).asArray(UInt32.self)
                    if isDecode && visit == 0 {
                        decodeObsTokens.withLock { $0 += 1 }
                    }
                }

                var obsTokenIDs: [Int] = []
                obsTokenIDs.reserveCapacity(overheadTokens)
                var obsPrefillMs = 0.0
                var obsFirstNs: UInt64 = 0
                var obsLastNs: UInt64 = 0
                do {
                    let obsCache = try inkling.newCache(parameters: GenerateParameters(maxKVSize: 4_096))
                    var obsIter = try TokenIterator(
                        input: LMInput(tokens: promptArray),
                        model: context.model,
                        cache: obsCache,
                        processor: nil,
                        sampler: ArgMaxSampler(),
                        maxTokens: overheadTokens)
                    obsPrefillMs = obsIter.promptPrefillTime * 1000.0
                    while let tok = obsIter.next() {
                        let now = DispatchTime.now().uptimeNanoseconds
                        if obsFirstNs == 0 { obsFirstNs = now }
                        obsLastNs = now
                        obsTokenIDs.append(tok)
                    }
                    Stream.gpu.synchronize()
                }
                inkling.routingObserver = nil
                Memory.clearCache()

                let obsDecodeTotalMs = Double(obsLastNs &- obsFirstNs) / 1_000_000.0
                let obsDecodeMsPerToken = obsTokenIDs.count > 1
                    ? obsDecodeTotalMs / Double(obsTokenIDs.count - 1)
                    : 0
                let obsTokPerSec = obsDecodeMsPerToken > 0 ? 1000.0 / obsDecodeMsPerToken : 0
                let deltaMs = obsDecodeMsPerToken - nilDecodeMsPerToken
                logProgress("phase3 obs probe prefillMs=\(String(format: "%.1f", obsPrefillMs)) decodeMs=\(String(format: "%.2f", obsDecodeMsPerToken)) delta=\(String(format: "%+.2f", deltaMs))")

                return OverheadReport(
                    workload: "prose_resident_64",
                    promptTokens: probePromptIDs.count,
                    warmupTokens: 8,
                    generatedTokens: nilTokenIDs.count,
                    decodeTraceTokens: decodeObsTokens.read(),
                    nilPrefillMs: nilPrefillMs,
                    nilDecodeMsPerToken: nilDecodeMsPerToken,
                    nilTokPerSec: nilTokPerSec,
                    obsPrefillMs: obsPrefillMs,
                    obsDecodeMsPerToken: obsDecodeMsPerToken,
                    obsTokPerSec: obsTokPerSec,
                    deltaMsPerToken: deltaMs,
                    tokensIdentical: nilTokenIDs == obsTokenIDs)
            }
        }
        Memory.clearCache()
        logProgress("phase3 released resident container")

        // Phase 4: Cache Policy Simulation anchored to measured Resident nilDecodeMsPerToken.
        let workloadTraces = rawDecodeTraces.map {
            WorkloadTrace(
                name: $0.name,
                nilDecodeMsPerToken: overheadReport.nilDecodeMsPerToken,
                decodeTokens: $0.decodeTokens)
        }
        let simulationRows = Self.runCacheSimulations(
            traces: workloadTraces,
            schedule: visitSchedule,
            textConfig: textConfig,
            expertIndex: safetensorsIndex)
        logProgress("phase4 completed cache simulations rows=\(simulationRows.count)")

        let results = CheckABenchmarkReport(
            modelPath: modelURL.path,
            hostModel: Self.hardwareModel(),
            physicalRAMBytes: ProcessInfo.processInfo.physicalMemory,
            macOSVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            traceBackend: traceBackendLabel,
            configSummary: ConfigSummary(
                hiddenLayers: textConfig.hiddenLayers,
                coreStart: textConfig.loop?.coreStart ?? 0,
                coreEnd: textConfig.loop?.coreEnd ?? textConfig.hiddenLayers,
                passes: textConfig.loop?.passes ?? 1,
                visitCount: visitCount,
                numExperts: textConfig.experts,
                topK: textConfig.expertsPerToken,
                contiguityVerified: safetensorsIndex.allContiguous,
                perLayerExpertRawMB: (0 ..< textConfig.hiddenLayers).map {
                    Double(safetensorsIndex.rawExpertBytes(layer: $0)) / 1e6
                },
                perLayerExpertAlignedMB: (0 ..< textConfig.hiddenLayers).map {
                    Double(safetensorsIndex.alignedExpertReadBytes(layer: $0)) / 1e6
                },
                totalExpertRawBytes: safetensorsIndex.totalExpertRawBytes),
            overhead: [overheadReport],
            prefillUniqueExperts: prefillChunkReports,
            locality: localityReport,
            simulation: simulationRows,
            ioReplay: ioReplayRows)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let jsonData = try encoder.encode(results)
        try jsonData.write(to: URL(fileURLWithPath: outPath))

        Self.printCheckASummary(results, jsonPath: outPath)
    }

    // MARK: - Synthetic Tiny Looped Model Helper (Non-Kernel Branch)

    private static func makeTinyModel(bfloat16: Bool, convKernelSize: Int) throws -> Inkling2 {
        let json = Inkling2ModelTests.tinyJSON.replacingOccurrences(
            of: "\"sconv_kernel_size\": 4",
            with: "\"sconv_kernel_size\": \(convKernelSize)")
        let config = try JSONDecoder().decode(Inkling2Configuration.self, from: Data(json.utf8))
        let model = Inkling2(config)
        MLXRandom.seed(5)
        var parameters = [String: MLXArray]()
        for (key, value) in model.parameters().flattened() where value.dtype == .float32 {
            parameters[key] = MLXRandom.normal(value.shape) * 0.3
        }
        for (key, value) in model.parameters().flattened() where key.hasSuffix(".global_scale") {
            parameters[key] = MLXArray.ones(value.shape)
        }
        for (key, value) in model.parameters().flattened()
        where key.hasSuffix("_proj.weight") && value.dtype == .uint32 {
            let prefix = String(key.dropLast(".weight".count))
            let (rows, packed) = (value.dim(1), value.dim(2))
            let dense = MLXRandom.normal([value.dim(0), rows, packed * 8]) * 0.3
            let (weight, scales, _) = MLX.quantized(
                dense, groupSize: 16, bits: 4, mode: .nvfp4)
            parameters[key] = weight
            parameters["\(prefix).scales"] = scales
        }
        if bfloat16 {
            let float32Suffixes = [
                ".query", "mlp.gate.weight", "e_score_correction_bias",
                "forget_gate_proj.weight", "message_gate_proj.weight", ".global_scale",
            ]
            for (key, value) in parameters
            where value.dtype == .float32 && !float32Suffixes.contains(where: key.hasSuffix) {
                parameters[key] = value.asType(.bfloat16)
            }
        }
        try model.update(
            parameters: ModuleParameters.unflattened(parameters), verify: .noUnusedKeys)
        if bfloat16 {
            try model.prepare()
        }
        eval(model)
        return model
    }

    // MARK: - Benchmark Types & Helpers

    private enum BenchmarkError: Error {
        case unexpectedModelType
        case malformedSafetensors(String)
        case ioFailure(String)
    }

    private struct VisitEntry: Sendable {
        let visit: Int
        let layer: Int
        let pass: Int
    }

    private static func expectedVisitSchedule(_ config: Inkling2TextConfiguration) -> [VisitEntry] {
        guard let loop = config.loop else {
            return (0 ..< config.hiddenLayers).map { VisitEntry(visit: $0, layer: $0, pass: 0) }
        }
        var entries: [VisitEntry] = []
        var visit = 0
        for layer in 0 ..< loop.coreStart {
            entries.append(VisitEntry(visit: visit, layer: layer, pass: 0))
            visit += 1
        }
        for pass in 0 ..< loop.passes {
            for layer in loop.core {
                entries.append(VisitEntry(visit: visit, layer: layer, pass: pass))
                visit += 1
            }
        }
        for layer in loop.coreEnd ..< config.hiddenLayers {
            entries.append(VisitEntry(visit: visit, layer: layer, pass: 0))
            visit += 1
        }
        return entries
    }

    private static func envValue(_ name: String) -> String? {
        let env = ProcessInfo.processInfo.environment
        if let v = env[name], !v.isEmpty { return v }
        if let v = env["TEST_RUNNER_\(name)"], !v.isEmpty { return v }
        return nil
    }

    private static func hardwareModel() -> String {
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        guard size > 0 else { return "Apple Silicon Mac" }
        var buf = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &buf, &size, nil, 0)
        return String(cString: buf)
    }

    // MARK: - Safetensors Header Inspector

    private struct ExpertSliceSpan {
        let shardIndex: Int
        let rawOffset: UInt64
        let rawLength: UInt64
        var alignedOffset: UInt64 { rawOffset & ~UInt64(16383) }
        var alignedLength: UInt64 {
            let end = rawOffset + rawLength
            let alignedEnd = (end + 16383) & ~UInt64(16383)
            return alignedEnd - alignedOffset
        }
    }

    private struct ParsedSafetensorsExperts {
        let shardURLs: [URL]
        let layerCount: Int
        let expertCount: Int
        let allContiguous: Bool
        // [layer][tensorKind (0..3)] = (shardIndex, baseFileOffset, stridePerExpert)
        let layers: [[(shardIndex: Int, baseOffset: UInt64, perExpertBytes: UInt64)]]
        let totalExpertRawBytes: UInt64

        func spans(layer: Int, expert: Int) -> [ExpertSliceSpan] {
            layers[layer].map { entry in
                ExpertSliceSpan(
                    shardIndex: entry.shardIndex,
                    rawOffset: entry.baseOffset + UInt64(expert) * entry.perExpertBytes,
                    rawLength: entry.perExpertBytes)
            }
        }

        func rawExpertBytes(layer: Int) -> UInt64 {
            layers[layer].reduce(0) { $0 + $1.perExpertBytes }
        }

        func alignedExpertReadBytes(layer: Int) -> UInt64 {
            // Compute exact aligned span for expert 0 (conservative per-read estimate; simulation
            // uses exact expert index offsets per access).
            spans(layer: layer, expert: 0).reduce(0) { $0 + $1.alignedLength }
        }
    }

    private static func parseSafetensorsExperts(in directory: URL) throws -> ParsedSafetensorsExperts {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let cfg = try JSONDecoder.json5().decode(Inkling2Configuration.self, from: configData)
        let text = cfg.textConfiguration

        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let shardNames: [String]
        if FileManager.default.fileExists(atPath: indexURL.path) {
            struct Idx: Decodable {
                let weight_map: [String: String]
            }
            let decoded = try JSONDecoder().decode(Idx.self, from: Data(contentsOf: indexURL))
            shardNames = Array(Set(decoded.weight_map.values)).sorted()
        } else {
            shardNames = (try FileManager.default.contentsOfDirectory(atPath: directory.path))
                .filter { $0.hasSuffix(".safetensors") }
                .sorted()
        }
        let shardURLs = shardNames.map { directory.appendingPathComponent($0) }

        struct HeaderMeta {
            let shardIndex: Int
            let dataStart: UInt64
            let offset: UInt64
            let length: UInt64
            let shape0: Int
        }
        var tensorMeta: [String: HeaderMeta] = [:]

        for (shardIdx, url) in shardURLs.enumerated() {
            let fh = try FileHandle(forReadingFrom: url)
            defer { try? fh.close() }
            guard let lenData = try fh.read(upToCount: 8), lenData.count == 8 else {
                throw BenchmarkError.malformedSafetensors("Short header in \(url.lastPathComponent)")
            }
            let headerLen = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
            guard let headerData = try fh.read(upToCount: Int(headerLen)),
                  headerData.count == Int(headerLen),
                  let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
            else {
                throw BenchmarkError.malformedSafetensors("Invalid header JSON in \(url.lastPathComponent)")
            }
            let dataStart = 8 + headerLen
            for (name, val) in json where name != "__metadata__" {
                guard let dict = val as? [String: Any],
                      let shape = dict["shape"] as? [Int],
                      let offsets = dict["data_offsets"] as? [NSNumber], offsets.count == 2
                else { continue }
                let b = offsets[0].uint64Value
                let e = offsets[1].uint64Value
                tensorMeta[name] = HeaderMeta(
                    shardIndex: shardIdx,
                    dataStart: dataStart,
                    offset: b,
                    length: e - b,
                    shape0: shape.first ?? 0)
            }
        }

        let suffixes = [
            "gate_up_proj.weight",
            "gate_up_proj.scales",
            "down_proj.weight",
            "down_proj.scales",
        ]
        var allContiguous = true
        var layers: [[(shardIndex: Int, baseOffset: UInt64, perExpertBytes: UInt64)]] = []
        var totalRaw: UInt64 = 0

        for l in 0 ..< text.hiddenLayers {
            let prefixA = "model.language_model.layers.\(l).mlp.experts."
            let prefixB = "language_model.layers.\(l).mlp.experts."
            var entries: [(Int, UInt64, UInt64)] = []
            for sfx in suffixes {
                guard let meta = tensorMeta[prefixA + sfx] ?? tensorMeta[prefixB + sfx] else {
                    throw BenchmarkError.malformedSafetensors("Missing expert tensor layer \(l) \(sfx)")
                }
                if meta.shape0 != text.experts || meta.length % UInt64(text.experts) != 0 {
                    allContiguous = false
                }
                let perExpert = meta.length / UInt64(max(1, text.experts))
                entries.append((meta.shardIndex, meta.dataStart + meta.offset, perExpert))
                totalRaw += meta.length
            }
            layers.append(entries)
        }

        return ParsedSafetensorsExperts(
            shardURLs: shardURLs,
            layerCount: text.hiddenLayers,
            expertCount: text.experts,
            allContiguous: allContiguous,
            layers: layers,
            totalExpertRawBytes: totalRaw)
    }

    // MARK: - Deterministic Workloads

    private struct Workload {
        let name: String
        let promptIDs: [Int]
    }

    private static func buildCheckAWorkloads(tokenizer: Tokenizer) -> [Workload] {
        let proseBase = [
            "Distributed storage engines achieve predictable tail latency by separating control metadata from bulk sequential payloads across NVMe queues.",
            "Modern language models pair sliding local attention with periodic global layers so long technical manuals remain coherent without exhausting memory.",
            "When an operating system bypasses the unified page cache, application eviction policies determine whether repeated accesses strike SSD or silicon.",
            "Mixture-of-experts architectures route each token to a tiny subset of feed-forward parameters, turning sparse compute into an I/O scheduling problem.",
            "Apple Silicon unified memory removes CPU-to-GPU copies, yet unbounded allocator pools still inflate process footprint during streaming decoding.",
            "Careful synchronization barriers ensure compute kernels finish reading shared Metal buffers before host workers overwrite slots from flash.",
            "Looped transformers revisit a shared core multiple times per token, raising temporal locality when activations diverge gradually across passes.",
        ]
        let codeBase = [
            "func mergeIntervals(_ ranges: [Range<Int>]) -> [Range<Int>] {\n    let sorted = ranges.sorted { $0.lowerBound < $1.lowerBound }\n    var out: [Range<Int>] = []\n    for r in sorted {\n        if let last = out.last, last.upperBound >= r.lowerBound {\n            out[out.count - 1] = last.lowerBound ..< max(last.upperBound, r.upperBound)\n        } else {\n            out.append(r)\n        }\n    }\n    return out\n}",
            "struct RingIndex {\n    var head: Int = 0\n    let capacity: Int\n    mutating func advance(by delta: Int) {\n        head = (head + delta) % capacity\n    }\n}",
            "func alignedSpan(offset: UInt64, length: UInt64, page: UInt64 = 16384) -> (UInt64, UInt64, Int) {\n    let start = offset & ~(page - 1)\n    let lead = Int(offset - start)\n    let total = ((UInt64(lead) + length + page - 1) & ~(page - 1))\n    return (start, total, lead)\n}",
        ]
        let qaBase = [
            "Question: Why does NVFP4 use a two-level scale factor? Answer: Each 16-element group stores a quantized E4M3 micro-scale while a float32 per-tensor global scale preserves dynamic range.",
            "Question: How do sliding attention layers interact with full attention layers? Answer: Local layers capture short-range syntax inside a bounded window while periodic global layers exchange long-distance facts.",
            "Question: Why must host reads avoid F_NOCACHE alignment gaps? Answer: Direct block reads perform best when byte spans align to 16 KB page boundaries with minimal over-read waste.",
        ]

        return [
            buildWorkload(
                name: "prose",
                header: "Summarize the systems design themes in the following technical excerpt and continue with a detailed technical analysis:\n\n",
                units: proseBase,
                tokenizer: tokenizer),
            buildWorkload(
                name: "code",
                header: "Review the following Swift systems utilities, identify edge cases, and write an optimized implementation:\n\n",
                units: codeBase,
                tokenizer: tokenizer),
            buildWorkload(
                name: "qa",
                header: "Study these systems engineering notes and answer follow-up questions with precise reasoning:\n\n",
                units: qaBase,
                tokenizer: tokenizer),
        ]
    }

    private static func buildWorkload(
        name: String,
        header: String,
        units: [String],
        tokenizer: Tokenizer
    ) -> Workload {
        var text = header
        var step = 0
        var ids = tokenizer.encode(text: text, addSpecialTokens: true)
        while ids.count < 1450 {
            let unit = units[step % units.count]
            text += "[\(step + 1)] \(unit)\n\n"
            step += 1
            ids = tokenizer.encode(text: text, addSpecialTokens: true)
        }
        if ids.count > 1750 {
            ids = Array(ids.prefix(1550))
        }
        return Workload(name: name, promptIDs: ids)
    }

    // MARK: - Analysis: Prefill, Locality, Cache Sim, IO Replay

    private static func analyzePrefillChunks(
        workload: String,
        promptTokens: Int,
        topK: Int,
        perVisitFlat: [[UInt32]],
        schedule: [VisitEntry]
    ) -> PrefillUniqueReport {
        var fullVisitCounts: [Int] = []
        fullVisitCounts.reserveCapacity(schedule.count)
        for v in 0 ..< schedule.count {
            fullVisitCounts.append(Set(perVisitFlat[v]).count)
        }

        let chunkSizes = [64, 128, 256, 512]
        var chunkSummaries: [ChunkUniqueSummary] = []
        for c in chunkSizes {
            var allChunkUniques: [Int] = []
            for v in 0 ..< schedule.count {
                let arr = perVisitFlat[v]
                let nPos = arr.count / topK
                var startPos = 0
                while startPos < nPos {
                    let endPos = min(nPos, startPos + c)
                    let slice = arr[(startPos * topK) ..< (endPos * topK)]
                    allChunkUniques.append(Set(slice).count)
                    startPos = endPos
                }
            }
            let mean = allChunkUniques.isEmpty ? 0 : Double(allChunkUniques.reduce(0, +)) / Double(allChunkUniques.count)
            let mx = allChunkUniques.max() ?? 0
            chunkSummaries.append(ChunkUniqueSummary(chunkSize: c, meanUnique: mean, maxUnique: mx))
        }

        let meanFull = fullVisitCounts.isEmpty ? 0 : Double(fullVisitCounts.reduce(0, +)) / Double(fullVisitCounts.count)
        return PrefillUniqueReport(
            workload: workload,
            promptTokens: promptTokens,
            meanFullUnique: meanFull,
            maxFullUnique: fullVisitCounts.max() ?? 0,
            fullPerVisit: fullVisitCounts,
            chunks: chunkSummaries)
    }

    private struct WorkloadTrace {
        let name: String
        let nilDecodeMsPerToken: Double
        let decodeTokens: [[[UInt32]]] // [token][visit] -> [expertIDs (K)]
    }

    private static func analyzeLocality(
        traces: [WorkloadTrace],
        schedule: [VisitEntry],
        textConfig: Inkling2TextConfiguration
    ) -> LocalityReport {
        let topK = Double(textConfig.expertsPerToken)
        let loop = textConfig.loop
        let coreStart = loop?.coreStart ?? 0
        let coreEnd = loop?.coreEnd ?? textConfig.hiddenLayers
        let passes = loop?.passes ?? 1

        // Map (layer, pass) -> visit
        var visitMap: [Int: [Int: Int]] = [:]
        for entry in schedule {
            visitMap[entry.layer, default: [:]][entry.pass] = entry.visit
        }

        // Cross-pass overlap per looped layer (across all workloads pooled)
        var crossPassRows: [CrossPassLayerReport] = []
        for layer in coreStart ..< coreEnd where passes > 1 {
            var overlaps01: [Double] = []
            var overlaps12: [Double] = []
            var overlaps02: [Double] = []
            var unionSizes: [Int] = []
            for trace in traces {
                for tok in trace.decodeTokens {
                    var sets: [Set<UInt32>] = []
                    for p in 0 ..< passes {
                        if let v = visitMap[layer]?[p] {
                            sets.append(Set(tok[v]))
                        }
                    }
                    guard sets.count == passes else { continue }
                    if sets.count >= 2 {
                        overlaps01.append(Double(sets[0].intersection(sets[1]).count) / topK)
                    }
                    if sets.count >= 3 {
                        overlaps12.append(Double(sets[1].intersection(sets[2]).count) / topK)
                        overlaps02.append(Double(sets[0].intersection(sets[2]).count) / topK)
                    }
                    var u = Set<UInt32>()
                    for s in sets { u.formUnion(s) }
                    unionSizes.append(u.count)
                }
            }
            func avg(_ xs: [Double]) -> Double { xs.isEmpty ? 0 : xs.reduce(0, +) / Double(xs.count) }
            let meanUnion = unionSizes.isEmpty ? 0 : Double(unionSizes.reduce(0, +)) / Double(unionSizes.count)
            crossPassRows.append(
                CrossPassLayerReport(
                    layer: layer,
                    overlap01: avg(overlaps01),
                    overlap12: avg(overlaps12),
                    overlap02: avg(overlaps02),
                    meanUnionPerToken: meanUnion))
        }

        // Token-to-token overlap per visit
        var tokToTokPerVisit: [Double] = []
        for v in 0 ..< schedule.count {
            var vals: [Double] = []
            for trace in traces where trace.decodeTokens.count >= 2 {
                for t in 1 ..< trace.decodeTokens.count {
                    let prev = Set(trace.decodeTokens[t - 1][v])
                    let curr = Set(trace.decodeTokens[t][v])
                    vals.append(Double(prev.intersection(curr).count) / topK)
                }
            }
            tokToTokPerVisit.append(vals.isEmpty ? 0 : vals.reduce(0, +) / Double(vals.count))
        }

        // O(T) two-pointer sliding window unique experts per layer for windows {8, 16, 32, 64}.
        let windowSizes = [8, 16, 32, 64]
        var visitsByLayer: [[Int]] = Array(repeating: [], count: textConfig.hiddenLayers)
        for entry in schedule {
            visitsByLayer[entry.layer].append(entry.visit)
        }
        var layerWindows: [LayerWindowUniqueReport] = []
        let numExperts = textConfig.experts
        for layer in 0 ..< textConfig.hiddenLayers {
            let layerVisits = visitsByLayer[layer]
            var stats: [Int: Double] = [:]
            for w in windowSizes {
                var sum = 0
                var windowCount = 0
                for trace in traces where trace.decodeTokens.count >= w {
                    var freq = [Int](repeating: 0, count: numExperts)
                    var distinct = 0
                    @inline(__always)
                    func addToken(_ t: Int) {
                        for v in layerVisits {
                            for rawE in trace.decodeTokens[t][v] {
                                let e = Int(rawE)
                                if e >= 0 && e < numExperts {
                                    if freq[e] == 0 { distinct += 1 }
                                    freq[e] += 1
                                }
                            }
                        }
                    }
                    @inline(__always)
                    func removeToken(_ t: Int) {
                        for v in layerVisits {
                            for rawE in trace.decodeTokens[t][v] {
                                let e = Int(rawE)
                                if e >= 0 && e < numExperts {
                                    freq[e] -= 1
                                    if freq[e] == 0 { distinct -= 1 }
                                }
                            }
                        }
                    }
                    for t in 0 ..< w {
                        addToken(t)
                    }
                    sum += distinct
                    windowCount += 1
                    for t in w ..< trace.decodeTokens.count {
                        removeToken(t - w)
                        addToken(t)
                        sum += distinct
                        windowCount += 1
                    }
                }
                stats[w] = windowCount == 0 ? 0 : Double(sum) / Double(windowCount)
            }
            layerWindows.append(
                LayerWindowUniqueReport(
                    layer: layer,
                    isLooped: loop?.core.contains(layer) ?? false,
                    w8: stats[8] ?? 0,
                    w16: stats[16] ?? 0,
                    w32: stats[32] ?? 0,
                    w64: stats[64] ?? 0))
        }

        return LocalityReport(
            crossPassPerLoopedLayer: crossPassRows,
            meanCrossPassOverlap01: crossPassRows.map(\.overlap01).reduce(0, +) / Double(max(1, crossPassRows.count)),
            meanCrossPassOverlap12: crossPassRows.map(\.overlap12).reduce(0, +) / Double(max(1, crossPassRows.count)),
            meanCrossPassOverlap02: crossPassRows.map(\.overlap02).reduce(0, +) / Double(max(1, crossPassRows.count)),
            meanCrossPassUnion: crossPassRows.map(\.meanUnionPerToken).reduce(0, +) / Double(max(1, crossPassRows.count)),
            tokenToTokenOverlapPerVisit: tokToTokPerVisit,
            meanTokenToTokenOverlap: tokToTokPerVisit.reduce(0, +) / Double(max(1, tokToTokPerVisit.count)),
            windowUniquePerLayer: layerWindows)
    }

    // MARK: - Cache Policy Simulator

    private enum SimPolicy: String, CaseIterable {
        case sharedLRU
        case sharedSLRU
        case perPassLRU
    }

    private struct SimBudget {
        let name: String
        let loopedSlots: Int
        let outerSlots: Int
    }

    private final class LayerCacheSimulator {
        let capacity: Int
        let policy: SimPolicy
        let passes: Int

        private var recencyTick: UInt64 = 0
        // For sharedLRU: [expertID: tick]
        private var lruMap: [Int: UInt64] = [:]
        // For sharedSLRU:
        private var probation: [Int: UInt64] = [:]
        private var protected: [Int: UInt64] = [:]
        private let protectedCap: Int
        // For perPassLRU: [pass: [expertID: tick]]
        private var perPassMaps: [[Int: UInt64]] = []
        private let passCap: Int

        init(capacity: Int, policy: SimPolicy, passes: Int) {
            self.capacity = capacity
            self.policy = policy
            self.passes = passes
            self.protectedCap = capacity / 2
            self.passCap = capacity / max(1, passes)
            if policy == .perPassLRU {
                self.perPassMaps = Array(repeating: [:], count: max(1, passes))
            }
        }

        func access(experts: [Int], pass: Int) -> [Int] {
            var misses: [Int] = []
            let pinned = Set(experts)
            for e in experts {
                recencyTick &+= 1
                switch policy {
                case .sharedLRU:
                    if lruMap[e] != nil {
                        lruMap[e] = recencyTick
                    } else {
                        misses.append(e)
                        if lruMap.count >= capacity {
                            if let victim = lruMap
                                .filter({ !pinned.contains($0.key) })
                                .min(by: { $0.value < $1.value })?.key
                            {
                                lruMap.removeValue(forKey: victim)
                            }
                        }
                        lruMap[e] = recencyTick
                    }

                case .sharedSLRU:
                    if protected[e] != nil {
                        protected[e] = recencyTick
                    } else if probation[e] != nil {
                        probation.removeValue(forKey: e)
                        if protectedCap > 0 {
                            if protected.count >= protectedCap {
                                if let demote = protected.min(by: { $0.value < $1.value }) {
                                    protected.removeValue(forKey: demote.key)
                                    probation[demote.key] = demote.value
                                }
                            }
                            protected[e] = recencyTick
                        } else {
                            probation[e] = recencyTick
                        }
                    } else {
                        misses.append(e)
                        while probation.count + protected.count >= capacity {
                            if let victim = probation
                                .filter({ !pinned.contains($0.key) })
                                .min(by: { $0.value < $1.value })?.key
                            {
                                probation.removeValue(forKey: victim)
                            } else if let victim = protected
                                .filter({ !pinned.contains($0.key) })
                                .min(by: { $0.value < $1.value })?.key
                            {
                                protected.removeValue(forKey: victim)
                            } else {
                                break
                            }
                        }
                        probation[e] = recencyTick
                    }

                case .perPassLRU:
                    let p = min(pass, perPassMaps.count - 1)
                    if perPassMaps[p][e] != nil {
                        perPassMaps[p][e] = recencyTick
                    } else {
                        misses.append(e)
                        if perPassMaps[p].count >= passCap {
                            if let victim = perPassMaps[p]
                                .filter({ !pinned.contains($0.key) })
                                .min(by: { $0.value < $1.value })?.key
                            {
                                perPassMaps[p].removeValue(forKey: victim)
                            }
                        }
                        perPassMaps[p][e] = recencyTick
                    }
                }
            }
            return misses
        }
    }

    private static func simulationBudgets() -> [SimBudget] {
        [
            SimBudget(name: "uniform_8", loopedSlots: 8, outerSlots: 8),
            SimBudget(name: "uniform_12", loopedSlots: 12, outerSlots: 12),
            SimBudget(name: "uniform_16", loopedSlots: 16, outerSlots: 16),
            SimBudget(name: "uniform_24", loopedSlots: 24, outerSlots: 24),
            SimBudget(name: "uniform_32", loopedSlots: 32, outerSlots: 32),
            SimBudget(name: "uniform_48", loopedSlots: 48, outerSlots: 48),
            SimBudget(name: "weighted_16_8", loopedSlots: 16, outerSlots: 8),
            SimBudget(name: "weighted_24_12", loopedSlots: 24, outerSlots: 12),
            SimBudget(name: "weighted_32_16", loopedSlots: 32, outerSlots: 16),
        ]
    }

    private static func runCacheSimulations(
        traces: [WorkloadTrace],
        schedule: [VisitEntry],
        textConfig: Inkling2TextConfiguration,
        expertIndex: ParsedSafetensorsExperts
    ) -> [SimulationRow] {
        let budgets = simulationBudgets()
        let passes = textConfig.loop?.passes ?? 1
        let coreRange = textConfig.loop?.core ?? (0 ..< textConfig.hiddenLayers)
        let topK = textConfig.expertsPerToken

        var rows: [SimulationRow] = []
        for budget in budgets {
            for policy in SimPolicy.allCases {
                let effectiveLooped = policy == .perPassLRU && passes > 1
                    ? (budget.loopedSlots / passes) * passes
                    : budget.loopedSlots
                var residentBytes: UInt64 = 0
                for layer in 0 ..< textConfig.hiddenLayers {
                    let s = coreRange.contains(layer) ? effectiveLooped : budget.outerSlots
                    residentBytes += UInt64(s) * expertIndex.rawExpertBytes(layer: layer)
                }
                let residentMB = Double(residentBytes) / 1e6

                if policy == .perPassLRU && passes > 1 && (budget.loopedSlots / passes) < topK {
                    rows.append(
                        SimulationRow(
                            budget: budget.name,
                            loopedSlots: budget.loopedSlots,
                            outerSlots: budget.outerSlots,
                            policy: policy.rawValue,
                            infeasible: true,
                            hitRate: 0,
                            steadyHitRate: 0,
                            missesPerToken: 0,
                            alignedMissedMBPerToken: 0,
                            residentMB: residentMB,
                            projMsAt2GBps: 0,
                            projTokPerSec: 0))
                    continue
                }

                var totalReq = 0
                var totalHits = 0
                var steadyReq = 0
                var steadyHits = 0
                var totalMisses = 0
                var totalAlignedBytes: UInt64 = 0
                var totalTokens = 0

                for trace in traces {
                    let sims: [LayerCacheSimulator] = (0 ..< textConfig.hiddenLayers).map { layer in
                        let isLooped = coreRange.contains(layer)
                        let cap = isLooped ? budget.loopedSlots : budget.outerSlots
                        let layerPolicy: SimPolicy = (policy == .perPassLRU && !isLooped) ? .sharedLRU : policy
                        let layerPasses = isLooped ? passes : 1
                        return LayerCacheSimulator(capacity: cap, policy: layerPolicy, passes: layerPasses)
                    }

                    for (tokIdx, tok) in trace.decodeTokens.enumerated() {
                        totalTokens += 1
                        for entry in schedule {
                            let req = tok[entry.visit].map(Int.init)
                            let misses = sims[entry.layer].access(experts: req, pass: entry.pass)
                            let reqCount = req.count
                            let missCount = misses.count
                            totalReq += reqCount
                            totalHits += (reqCount - missCount)
                            totalMisses += missCount
                            if tokIdx >= 32 {
                                steadyReq += reqCount
                                steadyHits += (reqCount - missCount)
                            }
                            for e in misses {
                                for span in expertIndex.spans(layer: entry.layer, expert: e) {
                                    totalAlignedBytes += span.alignedLength
                                }
                            }
                        }
                    }
                }

                let meanNilDecodeMs = traces.map(\.nilDecodeMsPerToken).reduce(0, +) / Double(max(1, traces.count))
                let alignedMissedBytesPerToken = Double(totalAlignedBytes) / Double(max(1, totalTokens))
                let projMsAt2GBps = alignedMissedBytesPerToken / 2e9 * 1000.0
                let projTokPerSec = (meanNilDecodeMs + projMsAt2GBps) > 0
                    ? 1000.0 / (meanNilDecodeMs + projMsAt2GBps) : 0

                rows.append(
                    SimulationRow(
                        budget: budget.name,
                        loopedSlots: budget.loopedSlots,
                        outerSlots: budget.outerSlots,
                        policy: policy.rawValue,
                        infeasible: false,
                        hitRate: totalReq > 0 ? Double(totalHits) / Double(totalReq) : 0,
                        steadyHitRate: steadyReq > 0 ? Double(steadyHits) / Double(steadyReq) : 0,
                        missesPerToken: Double(totalMisses) / Double(max(1, totalTokens)),
                        alignedMissedMBPerToken: alignedMissedBytesPerToken / 1e6,
                        residentMB: residentMB,
                        projMsAt2GBps: projMsAt2GBps,
                        projTokPerSec: projTokPerSec))
            }
        }
        return rows
    }

    private static func runIOReplay(
        trace: WorkloadTrace,
        schedule: [VisitEntry],
        textConfig: Inkling2TextConfiguration,
        expertIndex: ParsedSafetensorsExperts
    ) throws -> [IOReplayRow] {
        // Open FDs with F_NOCACHE once per shard and record file sizes for EOF clamping.
        var fds: [Int32] = []
        var fileSizes: [UInt64] = []
        defer {
            for fd in fds where fd >= 0 { close(fd) }
        }
        for url in expertIndex.shardURLs {
            let fd = open(url.path, O_RDONLY)
            guard fd >= 0 else {
                throw BenchmarkError.ioFailure("open failed for \(url.path): errno=\(errno)")
            }
            fds.append(fd)
            var st = stat()
            guard fstat(fd, &st) == 0, st.st_size >= 0 else {
                throw BenchmarkError.ioFailure("fstat failed for \(url.path): errno=\(errno)")
            }
            fileSizes.append(UInt64(st.st_size))
            _ = fcntl(fd, F_NOCACHE, 1)
        }

        // Reusable 16 KB-aligned bounce buffer
        var maxSpan = 16384
        for layer in 0 ..< expertIndex.layerCount {
            for s in expertIndex.spans(layer: layer, expert: 0) {
                maxSpan = max(maxSpan, Int(s.alignedLength))
            }
        }
        var alignedPtr: UnsafeMutableRawPointer?
        let rc = posix_memalign(&alignedPtr, 16384, maxSpan + 16384)
        guard rc == 0, let bounce = alignedPtr else {
            throw BenchmarkError.ioFailure("posix_memalign failed: \(rc)")
        }
        defer { free(bounce) }

        let totalTokens = trace.decodeTokens.count
        let startToken = totalTokens >= 200 ? 100 : max(0, totalTokens - 100)
        let endToken = min(totalTokens, startToken + 100)
        let weightedBudgets = simulationBudgets().filter { $0.name.hasPrefix("weighted_") }
        let passes = textConfig.loop?.passes ?? 1
        let coreRange = textConfig.loop?.core ?? (0 ..< textConfig.hiddenLayers)
        let topK = textConfig.expertsPerToken

        var rows: [IOReplayRow] = []
        for budget in weightedBudgets {
            for policy in SimPolicy.allCases {
                if policy == .perPassLRU && passes > 1 && (budget.loopedSlots / passes) < topK {
                    rows.append(
                        IOReplayRow(
                            budget: budget.name, policy: policy.rawValue, infeasible: true,
                            tokensReplayed: 0, timedOut: false, meanMsPerToken: 0, p50MsPerToken: 0,
                            p95MsPerToken: 0, readsPerToken: 0, alignedMBPerToken: 0,
                            effectiveMBps: 0, suspectCached: false, projMsAt2GBps: 0))
                    continue
                }

                let sims: [LayerCacheSimulator] = (0 ..< textConfig.hiddenLayers).map { layer in
                    let isLooped = coreRange.contains(layer)
                    let cap = isLooped ? budget.loopedSlots : budget.outerSlots
                    let layerPolicy: SimPolicy = (policy == .perPassLRU && !isLooped) ? .sharedLRU : policy
                    let layerPasses = isLooped ? passes : 1
                    return LayerCacheSimulator(capacity: cap, policy: layerPolicy, passes: layerPasses)
                }

                // Warm cache state on tokens 0 ..< startToken
                for t in 0 ..< startToken {
                    let tok = trace.decodeTokens[t]
                    for entry in schedule {
                        let req = tok[entry.visit].map(Int.init)
                        _ = sims[entry.layer].access(experts: req, pass: entry.pass)
                    }
                }

                var tokenMsList: [Double] = []
                var totalReads = 0
                var totalAlignedBytes: UInt64 = 0
                var timedOut = false
                let replayStartNs = DispatchTime.now().uptimeNanoseconds

                for t in startToken ..< endToken {
                    let tokStartNs = DispatchTime.now().uptimeNanoseconds
                    let tok = trace.decodeTokens[t]
                    for entry in schedule {
                        let req = tok[entry.visit].map(Int.init)
                        let misses = sims[entry.layer].access(experts: req, pass: entry.pass)
                        for e in misses {
                            for span in expertIndex.spans(layer: entry.layer, expert: e) {
                                let fileSize = fileSizes[span.shardIndex]
                                let clampedEnd = min(span.alignedOffset + span.alignedLength, fileSize)
                                guard clampedEnd > span.alignedOffset else {
                                    throw BenchmarkError.ioFailure("span offset \(span.alignedOffset) >= fileSize \(fileSize)")
                                }
                                let readBytes = UInt64(clampedEnd - span.alignedOffset)
                                var remaining = Int(readBytes)
                                var offset = off_t(span.alignedOffset)
                                var dstOff = 0
                                let fd = fds[span.shardIndex]
                                while remaining > 0 {
                                    let got = pread(fd, bounce.advanced(by: dstOff), remaining, offset)
                                    if got < 0 {
                                        if errno == EINTR { continue }
                                        throw BenchmarkError.ioFailure("pread errno=\(errno)")
                                    }
                                    if got == 0 {
                                        throw BenchmarkError.ioFailure("short read at EOF")
                                    }
                                    remaining -= got
                                    offset += off_t(got)
                                    dstOff += got
                                }
                                totalReads += 1
                                totalAlignedBytes += readBytes
                            }
                        }
                    }
                    let tokEndNs = DispatchTime.now().uptimeNanoseconds
                    tokenMsList.append(Double(tokEndNs &- tokStartNs) / 1_000_000.0)
                    let elapsedSec = Double(tokEndNs &- replayStartNs) / 1_000_000_000.0
                    if elapsedSec >= 45.0 {
                        timedOut = true
                        break
                    }
                }

                let replayed = tokenMsList.count
                let sorted = tokenMsList.sorted()
                let meanMs = replayed > 0 ? tokenMsList.reduce(0, +) / Double(replayed) : 0
                let p50Ms = sorted.isEmpty ? 0 : sorted[min(sorted.count - 1, sorted.count / 2)]
                let p95Ms = sorted.isEmpty ? 0 : sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.95))]
                let totalSec = tokenMsList.reduce(0, +) / 1000.0
                let mbps = totalSec > 0 ? (Double(totalAlignedBytes) / 1e6) / totalSec : 0
                let alignedMBPerTok = replayed > 0 ? (Double(totalAlignedBytes) / Double(replayed)) / 1e6 : 0
                print("INKLING2_CHECK_A replay-progress budget=\(budget.name) policy=\(policy.rawValue) replayed=\(replayed) mean=\(String(format: "%.1f", meanMs))ms MB/s=\(String(format: "%.0f", mbps))")
                fflush(stdout)

                rows.append(
                    IOReplayRow(
                        budget: budget.name,
                        policy: policy.rawValue,
                        infeasible: false,
                        tokensReplayed: replayed,
                        timedOut: timedOut,
                        meanMsPerToken: meanMs,
                        p50MsPerToken: p50Ms,
                        p95MsPerToken: p95Ms,
                        readsPerToken: replayed > 0 ? Double(totalReads) / Double(replayed) : 0,
                        alignedMBPerToken: alignedMBPerTok,
                        effectiveMBps: mbps,
                        suspectCached: mbps > 8000.0,
                        projMsAt2GBps: (alignedMBPerTok * 1e6) / 2e9 * 1000.0))
            }
        }
        return rows
    }

    private static func printCheckASummary(_ r: CheckABenchmarkReport, jsonPath: String) {
        print("INKLING2_CHECK_A json=\(jsonPath)")
        print("INKLING2_CHECK_A host=\(r.hostModel) ramGB=\(String(format: "%.1f", Double(r.physicalRAMBytes) / 1_073_741_824.0)) macOS=\(r.macOSVersion)")
        print("INKLING2_CHECK_A traceBackend=\(r.traceBackend)")
        print("INKLING2_CHECK_A config layers=\(r.configSummary.hiddenLayers) core=\(r.configSummary.coreStart)..<\(r.configSummary.coreEnd) passes=\(r.configSummary.passes) visits=\(r.configSummary.visitCount) experts=\(r.configSummary.numExperts) topK=\(r.configSummary.topK) contiguous=\(r.configSummary.contiguityVerified)")
        for o in r.overhead {
            print("INKLING2_CHECK_A overhead[\(o.workload)] prompt=\(o.promptTokens) nil=\(String(format: "%.2f", o.nilDecodeMsPerToken))ms (\(String(format: "%.2f", o.nilTokPerSec)) tok/s) obs=\(String(format: "%.2f", o.obsDecodeMsPerToken))ms (\(String(format: "%.2f", o.obsTokPerSec)) tok/s) delta=\(String(format: "%+.2f", o.deltaMsPerToken))ms identical=\(o.tokensIdentical)")
        }
        print("INKLING2_CHECK_A locality crossPass01=\(String(format: "%.3f", r.locality.meanCrossPassOverlap01)) crossPass12=\(String(format: "%.3f", r.locality.meanCrossPassOverlap12)) crossPass02=\(String(format: "%.3f", r.locality.meanCrossPassOverlap02)) meanUnion=\(String(format: "%.2f", r.locality.meanCrossPassUnion)) tokToTok=\(String(format: "%.3f", r.locality.meanTokenToTokenOverlap))")
        for row in r.simulation {
            if row.infeasible {
                print("INKLING2_CHECK_A sim budget=\(row.budget) policy=\(row.policy) INFEASIBLE")
            } else {
                print("INKLING2_CHECK_A sim budget=\(row.budget) policy=\(row.policy) hit=\(String(format: "%.1f", row.hitRate * 100))% steady=\(String(format: "%.1f", row.steadyHitRate * 100))% miss/tok=\(String(format: "%.1f", row.missesPerToken)) missMB/tok=\(String(format: "%.1f", row.alignedMissedMBPerToken)) residentMB=\(String(format: "%.0f", row.residentMB)) proj2GBps=\(String(format: "%.1f", row.projMsAt2GBps))ms (\(String(format: "%.2f", row.projTokPerSec)) tok/s)")
            }
        }
        for row in r.ioReplay where !row.infeasible {
            print("INKLING2_CHECK_A replay budget=\(row.budget) policy=\(row.policy) replayed=\(row.tokensReplayed) timedOut=\(row.timedOut) mean=\(String(format: "%.1f", row.meanMsPerToken))ms p50=\(String(format: "%.1f", row.p50MsPerToken))ms p95=\(String(format: "%.1f", row.p95MsPerToken))ms reads/tok=\(String(format: "%.1f", row.readsPerToken)) MB/s=\(String(format: "%.0f", row.effectiveMBps)) suspectCached=\(row.suspectCached)")
        }
    }

    // MARK: - JSON Report Schemas

    private struct ConfigSummary: Codable {
        let hiddenLayers: Int
        let coreStart: Int
        let coreEnd: Int
        let passes: Int
        let visitCount: Int
        let numExperts: Int
        let topK: Int
        let contiguityVerified: Bool
        let perLayerExpertRawMB: [Double]
        let perLayerExpertAlignedMB: [Double]
        let totalExpertRawBytes: UInt64
    }

    private struct OverheadReport: Codable {
        let workload: String
        let promptTokens: Int
        let warmupTokens: Int
        let generatedTokens: Int
        let decodeTraceTokens: Int
        let nilPrefillMs: Double
        let nilDecodeMsPerToken: Double
        let nilTokPerSec: Double
        let obsPrefillMs: Double
        let obsDecodeMsPerToken: Double
        let obsTokPerSec: Double
        let deltaMsPerToken: Double
        let tokensIdentical: Bool
    }

    private struct ChunkUniqueSummary: Codable {
        let chunkSize: Int
        let meanUnique: Double
        let maxUnique: Int
    }

    private struct PrefillUniqueReport: Codable {
        let workload: String
        let promptTokens: Int
        let meanFullUnique: Double
        let maxFullUnique: Int
        let fullPerVisit: [Int]
        let chunks: [ChunkUniqueSummary]
    }

    private struct CrossPassLayerReport: Codable {
        let layer: Int
        let overlap01: Double
        let overlap12: Double
        let overlap02: Double
        let meanUnionPerToken: Double
    }

    private struct LayerWindowUniqueReport: Codable {
        let layer: Int
        let isLooped: Bool
        let w8: Double
        let w16: Double
        let w32: Double
        let w64: Double
    }

    private struct LocalityReport: Codable {
        let crossPassPerLoopedLayer: [CrossPassLayerReport]
        let meanCrossPassOverlap01: Double
        let meanCrossPassOverlap12: Double
        let meanCrossPassOverlap02: Double
        let meanCrossPassUnion: Double
        let tokenToTokenOverlapPerVisit: [Double]
        let meanTokenToTokenOverlap: Double
        let windowUniquePerLayer: [LayerWindowUniqueReport]
    }

    private struct SimulationRow: Codable {
        let budget: String
        let loopedSlots: Int
        let outerSlots: Int
        let policy: String
        let infeasible: Bool
        let hitRate: Double
        let steadyHitRate: Double
        let missesPerToken: Double
        let alignedMissedMBPerToken: Double
        let residentMB: Double
        let projMsAt2GBps: Double
        let projTokPerSec: Double
    }

    private struct IOReplayRow: Codable {
        let budget: String
        let policy: String
        let infeasible: Bool
        let tokensReplayed: Int
        let timedOut: Bool
        let meanMsPerToken: Double
        let p50MsPerToken: Double
        let p95MsPerToken: Double
        let readsPerToken: Double
        let alignedMBPerToken: Double
        let effectiveMBps: Double
        let suspectCached: Bool
        let projMsAt2GBps: Double
    }

    private struct CheckABenchmarkReport: Codable {
        let modelPath: String
        let hostModel: String
        let physicalRAMBytes: UInt64
        let macOSVersion: String
        let traceBackend: String
        let configSummary: ConfigSummary
        let overhead: [OverheadReport]
        let prefillUniqueExperts: [PrefillUniqueReport]
        let locality: LocalityReport
        let simulation: [SimulationRow]
        let ioReplay: [IOReplayRow]
    }

    // MARK: - Cycle 2 Offline Tests: Aligned F_NOCACHE Reader, Eviction, Synthetic Parity & Dual-Mode

    func testAlignedShardReaderUnalignedSlices() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-reader-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        let fileA = tempDir.appendingPathComponent("shard-a.bin")
        let fileB = tempDir.appendingPathComponent("shard-b.bin")
        let byteCount = 600_000
        var payloadA = [UInt8](repeating: 0, count: byteCount)
        var payloadB = [UInt8](repeating: 0, count: byteCount)
        for i in 0 ..< byteCount {
            payloadA[i] = UInt8((i * 37 &+ 13) & 0xFF)
            payloadB[i] = UInt8((i * 73 &+ 91) & 0xFF)
        }
        try Data(payloadA).write(to: fileA)
        try Data(payloadB).write(to: fileB)

        for bypassPageCache in [false, true] {
            let reader = try Inkling2AlignedShardReader(
                shardURLs: [fileA, fileB],
                maxAlignedSliceBytes: 262_144,
                workerCount: 4,
                bypassPageCache: bypassPageCache)
            defer { reader.close() }

            // Test unaligned offsets and non-4K-multiple lengths matching real Inkling 2 scale slices.
            let slices: [(shard: Int, offset: Int, len: Int)] = [
                (0, 441, 165_888),
                (0, 2873, 82_944),
                (1, 3001, 17),
                (1, 19_655, 165_888),
                (0, 3193, 331_776),
                (1, 511_111, 40_961),
            ]

            var destinations: [UnsafeMutableRawPointer] = []
            defer {
                for ptr in destinations { ptr.deallocate() }
            }
            var tasks: [Inkling2AlignedShardReader.ReadTask] = []
            for s in slices {
                let dst = UnsafeMutableRawPointer.allocate(byteCount: s.len, alignment: 16)
                destinations.append(dst)
                tasks.append(
                    Inkling2AlignedShardReader.ReadTask(
                        shardIndex: s.shard,
                        fileOffset: UInt64(s.offset),
                        byteCount: s.len,
                        destination: dst))
            }

            let io = try reader.readSlicesConcurrently(tasks)
            XCTAssertEqual(io.rawBytes, UInt64(slices.map(\.len).reduce(0, +)))
            XCTAssertGreaterThanOrEqual(io.alignedBytes, io.rawBytes)
            XCTAssertEqual(io.alignedBytes % 16384, 0)

            for (i, s) in slices.enumerated() {
                let expected = s.shard == 0
                    ? Array(payloadA[s.offset ..< (s.offset + s.len)])
                    : Array(payloadB[s.offset ..< (s.offset + s.len)])
                let actual = Array(UnsafeBufferPointer(
                    start: destinations[i].assumingMemoryBound(to: UInt8.self),
                    count: s.len))
                XCTAssertEqual(
                    actual,
                    expected,
                    "Mismatch (bypassPageCache=\(bypassPageCache)) at slice \(i) offset=\(s.offset) len=\(s.len)")
            }

            // Test contiguous fileOffset + destination coalescing into one contiguous buffer.
            let coalescedLengths = [1024, 2048, 4096]
            let coalescedTotal = coalescedLengths.reduce(0, +)
            let coalescedDst = UnsafeMutableRawPointer.allocate(byteCount: coalescedTotal, alignment: 16)
            defer { coalescedDst.deallocate() }
            var fileCursor = 441
            var dstCursor = 0
            var coalescedTasks: [Inkling2AlignedShardReader.ReadTask] = []
            for len in coalescedLengths {
                coalescedTasks.append(
                    Inkling2AlignedShardReader.ReadTask(
                        shardIndex: 0,
                        fileOffset: UInt64(fileCursor),
                        byteCount: len,
                        destination: coalescedDst.advanced(by: dstCursor)))
                fileCursor += len
                dstCursor += len
            }
            let coalescedIO = try reader.readSlicesConcurrently(coalescedTasks)
            XCTAssertEqual(coalescedIO.rawBytes, UInt64(coalescedTotal))
            let actualCoalesced = Array(UnsafeBufferPointer(
                start: coalescedDst.assumingMemoryBound(to: UInt8.self),
                count: coalescedTotal))
            XCTAssertEqual(actualCoalesced, Array(payloadA[441 ..< (441 + coalescedTotal)]))

            // Second warm pass exercises OS page-cache pread hits when bypassPageCache == false.
            for (i, dst) in destinations.enumerated() {
                memset(dst, 0, slices[i].len)
            }
            memset(coalescedDst, 0, coalescedTotal)
            let warmIO = try reader.readSlicesConcurrently(tasks)
            XCTAssertEqual(warmIO.rawBytes, io.rawBytes)
            XCTAssertEqual(warmIO.alignedBytes, io.alignedBytes)
            for (i, s) in slices.enumerated() {
                let expected = s.shard == 0
                    ? Array(payloadA[s.offset ..< (s.offset + s.len)])
                    : Array(payloadB[s.offset ..< (s.offset + s.len)])
                let actual = Array(UnsafeBufferPointer(
                    start: destinations[i].assumingMemoryBound(to: UInt8.self),
                    count: s.len))
                XCTAssertEqual(
                    actual,
                    expected,
                    "Warm pass mismatch (bypassPageCache=\(bypassPageCache)) at slice \(i)")
            }
            let warmCoalescedIO = try reader.readSlicesConcurrently(coalescedTasks)
            XCTAssertEqual(warmCoalescedIO.rawBytes, coalescedIO.rawBytes)
            let actualWarmCoalesced = Array(UnsafeBufferPointer(
                start: coalescedDst.assumingMemoryBound(to: UInt8.self),
                count: coalescedTotal))
            XCTAssertEqual(actualWarmCoalesced, Array(payloadA[441 ..< (441 + coalescedTotal)]))

            // Verify unaligned slices ending near or at EOF on non-16KB-aligned files do not throw shortRead.
            let eofSlices: [(shard: Int, offset: Int, len: Int)] = [
                (0, byteCount - 10_000, 9_993), // ends at EOF - 7
                (1, byteCount - 4_097, 4_097),  // ends right at EOF
            ]
            var eofDestinations: [UnsafeMutableRawPointer] = []
            defer {
                for ptr in eofDestinations { ptr.deallocate() }
            }
            var eofTasks: [Inkling2AlignedShardReader.ReadTask] = []
            for s in eofSlices {
                let dst = UnsafeMutableRawPointer.allocate(byteCount: s.len, alignment: 16)
                eofDestinations.append(dst)
                eofTasks.append(
                    Inkling2AlignedShardReader.ReadTask(
                        shardIndex: s.shard,
                        fileOffset: UInt64(s.offset),
                        byteCount: s.len,
                        destination: dst))
            }
            let eofIO = try reader.readSlicesConcurrently(eofTasks)
            XCTAssertEqual(eofIO.rawBytes, UInt64(eofSlices.map(\.len).reduce(0, +)))
            XCTAssertGreaterThanOrEqual(eofIO.alignedBytes, eofIO.rawBytes)
            for (i, s) in eofSlices.enumerated() {
                let expected = s.shard == 0
                    ? Array(payloadA[s.offset ..< (s.offset + s.len)])
                    : Array(payloadB[s.offset ..< (s.offset + s.len)])
                let actual = Array(UnsafeBufferPointer(
                    start: eofDestinations[i].assumingMemoryBound(to: UInt8.self),
                    count: s.len))
                XCTAssertEqual(
                    actual,
                    expected,
                    "Mismatch (bypassPageCache=\(bypassPageCache)) at EOF slice \(i) offset=\(s.offset) len=\(s.len)")
            }
        }
    }

    func testEvictionPoliciesPinningAndIOErrorInvalidation() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-evict-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        let (config, resident) = try Self.buildSyntheticLoopedCheckpoint(in: tempDir)
        let text = config.textConfiguration

        for (policy, slots) in [
            (Inkling2EvictionPolicy.sharedLRU, 2),
            (Inkling2EvictionPolicy.sharedSLRU, 3),
            (Inkling2EvictionPolicy.perPassLRU, 4),
        ] {
            let streamingCfg = Inkling2StreamingConfig(
                modelDirectory: tempDir,
                expertCacheSlots: slots,
                evictionPolicy: policy,
                prefillChunkSize: 4,
                maxConcurrentReads: 2,
                loopWeightedSlots: false)
            let streamer = try Inkling2ExpertStreamer(
                textConfig: text,
                streamingConfig: streamingCfg,
                activation: softcappedSwiGLU(gateSoftcap: text.gateSoftcap, upSoftcap: text.upSoftcap))
            defer { streamer.close() }

            let cache = streamer.layerCache(2) // core layer
            let x = MLXRandom.normal([1, text.hiddenSize]).asType(.bfloat16)
            let w = MLXArray([Float(0.5), 0.5]).reshaped(1, 2)
            let res = MLXArray.zeros([1, text.hiddenSize], dtype: .float32)
            let convW = MLXArray.ones([text.hiddenSize, 4, 1], dtype: .bfloat16)
            let st = MLXArray.zeros([1, 3, text.hiddenSize], dtype: .bfloat16)

            // Access experts [0, 1], then [0, 2], then [0, 1] again -> tests eviction + pinning + hit tracking.
            for (idxVals, pass) in [([UInt32(0), 1], 0), ([UInt32(0), 2], 1), ([UInt32(0), 1], 0)] {
                let idx = MLXArray(idxVals).reshaped(1, 2)
                let (out, _) = try cache.decode(
                    x, indices: idx, weights: w, residual: res, state: st,
                    convolutionWeight: convW, pass: pass)
                eval(out)
            }
            let statsAfterWarm = streamer.stats
            XCTAssertGreaterThan(statsAfterWarm.cacheHits, 0, "Expected cache hits for \(policy)")
            XCTAssertGreaterThan(statsAfterWarm.cacheMisses, 0, "Expected cache misses for \(policy)")
        }

        // Verify I/O error fails closed and invalidates victim slot rather than serving stale weights.
        let streamingCfg = Inkling2StreamingConfig(
            modelDirectory: tempDir,
            expertCacheSlots: 2,
            evictionPolicy: .sharedLRU,
            prefillChunkSize: 4,
            maxConcurrentReads: 2,
            loopWeightedSlots: false)
        let streamed = try Inkling2(config, streaming: streamingCfg)
        try loadWeights(modelDirectory: tempDir, model: streamed)
        defer { streamed.releaseStreamingResources() }

        let tokens = MLXArray([Int32(3), Int32(7), Int32(11)]).reshaped(1, 3)
        let cache = try streamed.newCache(parameters: nil)
        let okLogits = streamed(tokens[0..., ..<2], cache: cache)
        eval(okLogits)
        XCTAssertNil(streamed.consumeStreamingError())

        // Inject shard failure before decode step -> decode records I/O error.
        let streamedRef = try XCTUnwrap(streamed.debugExpertStreamer)
        streamedRef.debugInjectIOFailure(onShardSubstring: "model.safetensors")
        let errStep = streamed(MLXArray([Int32(15)]).reshaped(1, 1), cache: cache)
        eval(errStep)
        XCTAssertNotNil(streamed.consumeStreamingError(), "Injected SSD I/O failure must be surfaced")

        // Clear failure -> next forward must recover cleanly and produce valid logits.
        streamedRef.debugInjectIOFailure(onShardSubstring: nil)
        let recoveryCache = try streamed.newCache(parameters: nil)
        let recovered = streamed(tokens, cache: recoveryCache)
        eval(recovered)
        XCTAssertNil(streamed.consumeStreamingError())

        let resCache = try resident.newCache(parameters: nil)
        let expected = resident(tokens, cache: resCache)
        eval(expected)
        let maxDiff = abs(recovered.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(maxDiff, 1e-2, "Recovered logits diverged from resident: \(maxDiff)")
    }

    func testSyntheticStreamedMatchesResidentPrefillAndDecode() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-parity-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        let (config, resident) = try Self.buildSyntheticLoopedCheckpoint(in: tempDir)

        for (policy, slots) in [
            (Inkling2EvictionPolicy.sharedSLRU, 2),
            (Inkling2EvictionPolicy.sharedLRU, 2),
            (Inkling2EvictionPolicy.perPassLRU, 4),
        ] {
            let streamingCfg = Inkling2StreamingConfig(
                modelDirectory: tempDir,
                expertCacheSlots: slots,
                evictionPolicy: policy,
                prefillChunkSize: 4,
                maxConcurrentReads: 2,
                loopWeightedSlots: false)
            let streamed = try Inkling2(config, streaming: streamingCfg)
            defer { streamed.releaseStreamingResources() }

            XCTAssertTrue(streamed.isStreamingExpertsActive)
            XCTAssertEqual(streamed.excludedCheckpointNamespaces.count, config.textConfiguration.hiddenLayers * 3)
            XCTAssertFalse(
                streamed.parameters().flattened().contains { $0.0.contains(".mlp.experts.") },
                "Streamed Inkling2 must not register expert weights as resident parameters")

            // Exercise production loadWeights path with strict verify: [.all].
            try loadWeights(modelDirectory: tempDir, model: streamed)
            eval(streamed)

            // 10-token prompt forces multi-chunk prefill (4 + 4 + 2) with S=2 < numExperts=4 batching.
            let prompt = MLXArray((0 ..< 10).map { Int32(($0 * 7 + 3) % 48) }).reshaped(1, 10)
            let resCache = try resident.newCache(parameters: nil)
            let strCache = try streamed.newCache(parameters: nil)

            let resPrep = try resident.prepare(
                LMInput(tokens: prompt), cache: resCache, state: nil, prefill: .init(stepSize: 4))
            let strPrep = try streamed.prepare(
                LMInput(tokens: prompt), cache: strCache, state: nil, prefill: .init(stepSize: 4))

            guard case .logits(let resOut) = resPrep, case .logits(let strOut) = strPrep else {
                XCTFail("prepare did not return logits")
                return
            }
            let prefillDiff = abs(resOut.logits.asType(.float32) - strOut.logits.asType(.float32)).max().item(Float.self)
            XCTAssertLessThan(prefillDiff, 1e-2, "Prefill diff too large for \(policy): \(prefillDiff)")
            XCTAssertEqual(
                argMax(resOut.logits[0, -1], axis: -1).item(Int.self),
                argMax(strOut.logits[0, -1], axis: -1).item(Int.self),
                "Prefill argmax mismatch for \(policy)")

            // 6 decode steps across loop passes with S=2 forcing slot evictions on every layer.
            var nextTok = argMax(resOut.logits[0, -1], axis: -1).item(Int32.self)
            for step in 0 ..< 6 {
                let xTok = MLXArray([nextTok]).reshaped(1, 1)
                let resStep = resident(xTok, cache: resCache)
                let strStep = streamed(xTok, cache: strCache)
                eval(resStep, strStep)
                XCTAssertNil(streamed.consumeStreamingError())

                let stepDiff = abs(resStep.asType(.float32) - strStep.asType(.float32)).max().item(Float.self)
                XCTAssertLessThan(stepDiff, 2e-2, "Decode step \(step) diff too large for \(policy): \(stepDiff)")
                let rArg = argMax(resStep[0, -1], axis: -1).item(Int32.self)
                let sArg = argMax(strStep[0, -1], axis: -1).item(Int32.self)
                XCTAssertEqual(rArg, sArg, "Decode step \(step) argmax mismatch for \(policy)")
                nextTok = rArg
            }

            let stats = try XCTUnwrap(streamed.streamingStats)
            XCTAssertGreaterThan(stats.bytesRead, 0)
            XCTAssertGreaterThan(stats.totalRequests, 0)
        }
    }

    func testStreamedNewCachePassesKVCacheValidationAcrossContextSizes() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-kvcache-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        let (config, _) = try Self.buildSyntheticLoopedCheckpoint(in: tempDir)
        let streamingCfg = Inkling2StreamingConfig(
            modelDirectory: tempDir,
            expertCacheSlots: 2,
            evictionPolicy: .sharedSLRU,
            prefillChunkSize: 4,
            maxConcurrentReads: 2,
            loopWeightedSlots: false)
        let streamed = try Inkling2(config, streaming: streamingCfg)
        try loadWeights(modelDirectory: tempDir, model: streamed)
        defer { streamed.releaseStreamingResources() }

        for maxKVSize in [2, 6, 4_096, 8_192, 16_384, 32_768] {
            let params = GenerateParameters(maxKVSize: maxKVSize)
            let legacyConfig = KVCacheConfiguration(
                capacity: try .init(
                    maxTokens: maxKVSize,
                    preservedPrefixTokens: min(4, max(0, maxKVSize - 1))))
            let cache = try streamed.newCache(parameters: params)
            XCTAssertNoThrow(
                try validateKVCacheCompatibility(cache, configuration: legacyConfig),
                "Legacy maxKVSize validation failed for maxKVSize=\(maxKVSize)")
        }

        for maxKVSize in [6, 4_096, 8_192, 16_384, 32_768] {
            let kvConfig = KVCacheConfiguration(
                capacity: try .init(maxTokens: maxKVSize, preservedPrefixTokens: 0))
            let params = GenerateParameters(kvCache: kvConfig)
            let cache = try streamed.newCache(parameters: params)
            XCTAssertNoThrow(
                try validateKVCacheCompatibility(cache, configuration: kvConfig),
                "Typed kvCache validation failed for maxKVSize=\(maxKVSize)")
        }

        // Exercise TokenIterator's parameters initializer (same kvCachePlan.validated path ChatSession uses).
        let prompt = MLXArray((0 ..< 6).map { Int32(($0 * 5 + 1) % 48) }).reshaped(1, 6)
        let genParams = GenerateParameters(
            maxTokens: 4,
            maxKVSize: 16_384,
            temperature: 0.0,
            prefill: PrefillParameters(stepSize: 4))
        var iter = try TokenIterator(
            input: LMInput(tokens: prompt),
            model: streamed,
            cache: nil,
            parameters: genParams)
        var generated: [Int] = []
        while let tok = iter.next() {
            generated.append(tok)
        }
        Stream.gpu.synchronize()
        XCTAssertEqual(generated.count, 4)
        XCTAssertNil(streamed.consumeStreamingError())
    }

    @MainActor
    func testDualModeCatalogStoreAndRuntime() {
        let defaults = UserDefaults.standard
        let savedEnabled = defaults.object(forKey: Keys.enabledMLXStreamedModelIDs)
        let savedCtx = defaults.object(forKey: Keys.streamedModelContextTokens)
        let savedSlots = defaults.object(forKey: Keys.streamedModelExpertCacheSlots)
        defer {
            if let savedEnabled { defaults.set(savedEnabled, forKey: Keys.enabledMLXStreamedModelIDs) }
            else { defaults.removeObject(forKey: Keys.enabledMLXStreamedModelIDs) }
            if let savedCtx { defaults.set(savedCtx, forKey: Keys.streamedModelContextTokens) }
            else { defaults.removeObject(forKey: Keys.streamedModelContextTokens) }
            if let savedSlots { defaults.set(savedSlots, forKey: Keys.streamedModelExpertCacheSlots) }
            else { defaults.removeObject(forKey: Keys.streamedModelExpertCacheSlots) }
        }

        // StreamedMoEInstall remains untouched (6 .gturbo variants).
        XCTAssertEqual(StreamedMoEInstall.variants.count, 6)

        let sourceID = MLXStreamedInstall.inkling2Mini4Bit.sourceModelID
        let ssdID = MLXStreamedInstall.inkling2Mini4Bit.modelID
        XCTAssertEqual(sourceID, "mlx-community/Inkling-2-Mini-4bit")
        XCTAssertEqual(ssdID, "mlx-streamed/inkling-2-mini-4bit")

        let sourceModel = ModelCatalog.model(for: sourceID)
        let ssdModel = ModelCatalog.model(for: ssdID)
        XCTAssertNotNil(sourceModel)
        XCTAssertNotNil(ssdModel)
        XCTAssertEqual(sourceModel?.backend, .mlx)
        XCTAssertEqual(ssdModel?.backend, .mlxStreamed)
        XCTAssertEqual(ssdModel?.supportsVision, true)
        XCTAssertEqual(ssdModel?.supportsThinking, true)
        XCTAssertEqual(ssdModel?.supportsThinkingToggle, false)
        XCTAssertEqual(sourceModel?.supportsThinkingToggle, false)
        XCTAssertEqual(ssdModel?.pickerDisplayName, "SSD · Inkling 2 Mini 4-bit")
        XCTAssertEqual(sourceModel?.pickerDisplayName, "MLX · Inkling 2 Mini 4-bit")
        XCTAssertEqual(sourceModel?.runtimeContextTokens, 1_048_576)
        XCTAssertTrue(StreamedMoERuntimeSettings.allowedContextTokens.contains(ssdModel?.runtimeContextTokens ?? 0))
        XCTAssertEqual(ssdModel?.hugginFaceURL.absoluteString, "https://huggingface.co/mlx-community/Inkling-2-Mini-4bit")
        XCTAssertEqual(ModelDownloadMode.availableModes(for: sourceID), [.mlx, .ssdStreamed])
        XCTAssertTrue(ModelRow.showsAddStreamedVersionButton(
            sourceModelID: sourceID, isStreamedAvailable: false, isStreamedDownloading: false))

        // RAM estimate matches actual expert bytes across 8..32 slots.
        for slots in StreamedMoERuntimeSettings.allowedExpertCacheSlots {
            let bytes = MLXStreamedInstall.inkling2Mini4Bit.expertCacheBytes(forSlots: slots)
            XCTAssertGreaterThan(bytes, 500_000_000)
        }

        let store = ModelStore()
        store.enableMLXStreamedModel(ssdID)
        XCTAssertTrue(store.isMLXStreamedEnabled(ssdID))
        // Re-reading persisted UserDefaults works across store restart.
        let reloadedStore = ModelStore()
        XCTAssertTrue(reloadedStore.isMLXStreamedEnabled(ssdID))
        // Deleting SSD entry disables it without deleting anything on disk.
        XCTAssertNil(reloadedStore.delete(ssdID))
        XCTAssertFalse(reloadedStore.isMLXStreamedEnabled(ssdID))

        // EngineRuntime settings reload applies to both .streamedMoE and .mlxStreamed, never .mlx.
        let runtime = EngineRuntime(modelStore: store)
        let fakeSSD = DummyLoadedEngine(id: ssdID)
        runtime.installSlot(fakeSSD)
        try? runtime.loadDidSucceed(fakeSSD, id: ssdID)
        runtime.streamedModelSettingsDidChange()
        XCTAssertFalse(fakeSSD.isLoaded)
        XCTAssertNil(runtime.loadedModelID)

        let fakeMLX = DummyLoadedEngine(id: sourceID)
        runtime.installSlot(fakeMLX)
        try? runtime.loadDidSucceed(fakeMLX, id: sourceID)
        runtime.streamedModelSettingsDidChange()
        XCTAssertTrue(fakeMLX.isLoaded)
        runtime.unloadCurrent()
    }

    func testRealInkling2StreamedParityAndRAM() async throws {
        guard let benchDir = Self.envValue("INKLING2_BENCH_DIR"), !benchDir.isEmpty else {
            return
        }
        let modelURL = URL(fileURLWithPath: benchDir, isDirectory: true)
        XCTAssertTrue(FileManager.default.fileExists(atPath: modelURL.appendingPathComponent("config.json").path))
        await Inkling2.register()

        let configuration = ModelConfiguration(
            directory: modelURL,
            extraEOSTokens: ["<|content_model_end_sampling|>"])
        let promptText = "Explain how NVMe direct I/O bypasses the OS page cache in 3 concise sentences."

        // 1. Measure 8-slot (8 GB Mac budget) load footprint first.
        do {
            Memory.cacheLimit = 64 * 1024 * 1024
            let cfg8 = Inkling2StreamingConfig(
                modelDirectory: modelURL,
                expertCacheSlots: 8)
            let c8 = try await Inkling2.withStreamingLoad(cfg8) {
                try await VLMModelFactory.shared.loadContainer(
                    from: HubDownloader(ModelStore.hubClient),
                    using: HuggingFaceTokenizerLoader(),
                    configuration: configuration)
            }
            let stats8: Inkling2StreamingStats? = await c8.perform { ctx in
                (ctx.model as? Inkling2)?.streamingStats
            }
            let active8 = Memory.activeMemory
            let foot8 = Self.processPhysFootprintBytes()
            print("INKLING2_REAL_SSD[slots=8] postLoadActiveMB=\(active8 / 1_048_576) footprintMB=\(foot8 / 1_048_576) slotCacheMB=\((stats8?.residentSlotBytes ?? 0) / 1_048_576)")
            await c8.perform { ctx in (ctx.model as? Inkling2)?.releaseStreamingResources() }
        }
        Memory.clearCache()

        // 2. Measure 16-slot SSD-streamed load + 32-token greedy generation.
        Memory.cacheLimit = 64 * 1024 * 1024
        let cfg16 = Inkling2StreamingConfig(
            modelDirectory: modelURL,
            expertCacheSlots: 16)
        let streamedContainer = try await Inkling2.withStreamingLoad(cfg16) {
            try await VLMModelFactory.shared.loadContainer(
                from: HubDownloader(ModelStore.hubClient),
                using: HuggingFaceTokenizerLoader(),
                configuration: configuration)
        }
        let postLoadActive16 = Memory.activeMemory
        let postLoadFootprint16 = Self.processPhysFootprintBytes()

        let streamedResult = try await streamedContainer.perform { context -> (tokens: [Int], decodedText: String, singleTokenLogits: [Float], prefillLogits: [Float], prefillMs: Double, tokPerSec: Double, prefillIoMs: Double, decodeIoMsPerTok: Double, decodeHitRate: Double, warmPrefillMs: Double, warmTokPerSec: Double, warmPrefillIoMs: Double, warmDecodeIoMsPerTok: Double, warmDecodeHitRate: Double, stats: Inkling2StreamingStats, activeAfter: Int, footprintAfter: UInt64) in
            guard let inkling = context.model as? Inkling2 else {
                throw BenchmarkError.unexpectedModelType
            }
            XCTAssertTrue(inkling.isStreamingExpertsActive)
            let promptIDs = context.tokenizer.encode(text: promptText, addSpecialTokens: true)
            let promptArray = MLXArray(promptIDs.map(Int32.init)).reshaped(1, promptIDs.count)

            // A. Single-token (L=1) decode-path logit check against resident weights.
            let singleCache = try inkling.newCache(parameters: GenerateParameters(maxKVSize: 4_096))
            let singleTokenArr = MLXArray([Int32(promptIDs[0])]).reshaped(1, 1)
            let singleLogitsArr = inkling(singleTokenArr, cache: singleCache).reshaped(-1).asType(.float32)
            eval(singleLogitsArr)
            if let streamErr = inkling.consumeStreamingError() {
                throw streamErr
            }
            let singleLogits = singleLogitsArr.asArray(Float.self)

            // B. Step-0 full-prompt prefill logit check.
            let prefillCheckCache = try inkling.newCache(parameters: GenerateParameters(maxKVSize: 4_096))
            let prefillPrepared = try inkling.prepare(
                LMInput(tokens: promptArray),
                cache: prefillCheckCache,
                state: nil,
                prefill: PrefillParameters(stepSize: cfg16.prefillChunkSize))
            guard case .logits(let prefillOut) = prefillPrepared else {
                throw BenchmarkError.unexpectedModelType
            }
            let prefillLogitsArr = prefillOut.logits.reshaped(-1).asType(.float32)
            eval(prefillLogitsArr)
            if let streamErr = inkling.consumeStreamingError() {
                throw streamErr
            }
            let prefillLogits = prefillLogitsArr.asArray(Float.self)

            // C. First 32-token greedy generation telemetry (prefill I/O split from decode I/O).
            let statsBeforeC = try XCTUnwrap(inkling.streamingStats)
            let cache = try inkling.newCache(parameters: GenerateParameters(maxKVSize: 4_096))
            var iter = try TokenIterator(
                input: LMInput(tokens: promptArray),
                model: context.model,
                cache: cache,
                processor: nil,
                sampler: ArgMaxSampler(),
                prefill: PrefillParameters(stepSize: cfg16.prefillChunkSize),
                maxTokens: 32)
            let prefillMs = iter.promptPrefillTime * 1000.0
            let statsAfterPrefill = try XCTUnwrap(inkling.streamingStats)
            var ids: [Int] = []
            var t0: UInt64 = 0
            var tLast: UInt64 = 0
            while let tok = iter.next() {
                let now = DispatchTime.now().uptimeNanoseconds
                if t0 == 0 { t0 = now }
                tLast = now
                ids.append(tok)
            }
            Stream.gpu.synchronize()
            if let streamErr = inkling.consumeStreamingError() {
                throw streamErr
            }
            let stats = try XCTUnwrap(inkling.streamingStats)
            let prefillIoMs = Double(statsAfterPrefill.ioNanoseconds &- statsBeforeC.ioNanoseconds) / 1_000_000.0
            let decodeIoMs = Double(stats.ioNanoseconds &- statsAfterPrefill.ioNanoseconds) / 1_000_000.0
            let decodeIoMsPerTok = ids.isEmpty ? 0 : decodeIoMs / Double(ids.count)
            let decodeReqs = stats.totalRequests - statsAfterPrefill.totalRequests
            let decodeHits = stats.cacheHits - statsAfterPrefill.cacheHits
            let decodeHitRate = decodeReqs > 0 ? Double(decodeHits) / Double(decodeReqs) : 0
            let decodedText = context.tokenizer.decode(tokenIds: ids)
            let decodeMs = Double(tLast &- t0) / 1_000_000.0
            let tps = (ids.count > 1 && decodeMs > 0) ? Double(ids.count - 1) / (decodeMs / 1000.0) : 0

            // D. Warm second 32-token pass on the same container (OS page cache + warm slots).
            let warmCache = try inkling.newCache(parameters: GenerateParameters(maxKVSize: 4_096))
            var warmIter = try TokenIterator(
                input: LMInput(tokens: promptArray),
                model: context.model,
                cache: warmCache,
                processor: nil,
                sampler: ArgMaxSampler(),
                prefill: PrefillParameters(stepSize: cfg16.prefillChunkSize),
                maxTokens: 32)
            let warmPrefillMs = warmIter.promptPrefillTime * 1000.0
            let statsAfterWarmPrefill = try XCTUnwrap(inkling.streamingStats)
            var warmCount = 0
            var wt0: UInt64 = 0
            var wtLast: UInt64 = 0
            while warmIter.next() != nil {
                let now = DispatchTime.now().uptimeNanoseconds
                if wt0 == 0 { wt0 = now }
                wtLast = now
                warmCount += 1
            }
            Stream.gpu.synchronize()
            if let streamErr = inkling.consumeStreamingError() {
                throw streamErr
            }
            let statsAfterWarm = try XCTUnwrap(inkling.streamingStats)
            let warmPrefillIoMs = Double(statsAfterWarmPrefill.ioNanoseconds &- stats.ioNanoseconds) / 1_000_000.0
            let warmDecodeIoMs = Double(statsAfterWarm.ioNanoseconds &- statsAfterWarmPrefill.ioNanoseconds) / 1_000_000.0
            let warmDecodeIoMsPerTok = warmCount > 0 ? warmDecodeIoMs / Double(warmCount) : 0
            let warmDecodeReqs = statsAfterWarm.totalRequests - statsAfterWarmPrefill.totalRequests
            let warmDecodeHits = statsAfterWarm.cacheHits - statsAfterWarmPrefill.cacheHits
            let warmDecodeHitRate = warmDecodeReqs > 0 ? Double(warmDecodeHits) / Double(warmDecodeReqs) : 0
            let warmDecodeMs = Double(wtLast &- wt0) / 1_000_000.0
            let warmTps = (warmCount > 1 && warmDecodeMs > 0) ? Double(warmCount - 1) / (warmDecodeMs / 1000.0) : 0

            let activeAfter = Memory.activeMemory
            let footAfter = Self.processPhysFootprintBytes()
            return (
                ids, decodedText, singleLogits, prefillLogits, prefillMs, tps,
                prefillIoMs, decodeIoMsPerTok, decodeHitRate,
                warmPrefillMs, warmTps, warmPrefillIoMs, warmDecodeIoMsPerTok, warmDecodeHitRate,
                stats, activeAfter, footAfter
            )
        }

        // 2b. Exercise full ChatSession path with maxKVSize=16_384 (matching SSD Settings default)
        // before releasing streamedContainer to catch KV-cache validation bugs end-to-end.
        let sessionParams = GenerateParameters(
            maxTokens: 16,
            maxKVSize: 16_384,
            temperature: 0.0,
            prefill: PrefillParameters(stepSize: Inkling2StreamingConfig.defaultPrefillChunkSize))
        let chatSession = ChatSession(
            streamedContainer,
            generateParameters: sessionParams,
            additionalContext: Inkling2.additionalContext(thinkingEnabled: true))
        var sessionReasoning = ""
        var sessionChunks = ""
        for try await event in chatSession.streamDetails(to: "explain gradient descent") {
            switch event {
            case .reasoning(let text): sessionReasoning += text
            case .chunk(let text): sessionChunks += text
            case .info, .toolCall, .rejectedToolCall, .steering: break
            }
        }
        let sessionStreamErr = await streamedContainer.perform { ctx in
            (ctx.model as? Inkling2)?.consumeStreamingError()
        }
        XCTAssertNil(sessionStreamErr)
        XCTAssertFalse(
            (sessionReasoning + sessionChunks).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "ChatSession with maxKVSize=16384 produced empty output")
        print("INKLING2_REAL_SSD_CHATSESSION reasoningChars=\(sessionReasoning.count) chunkChars=\(sessionChunks.count) preview=\((sessionReasoning + sessionChunks).prefix(80))")
        fflush(stdout)

        await streamedContainer.perform { ctx in (ctx.model as? Inkling2)?.releaseStreamingResources() }
        Memory.clearCache()

        XCTAssertLessThan(postLoadActive16, 4_000_000_000, "Streamed post-load active memory too high: \(postLoadActive16)")

        // 3. Resident Standard MLX greedy baseline on the exact same prompt.
        Memory.cacheLimit = MLXEngine.activeBufferCacheLimit()
        let residentContainer = try await Inkling2.withStreamingLoad(nil) {
            try await VLMModelFactory.shared.loadContainer(
                from: HubDownloader(ModelStore.hubClient),
                using: HuggingFaceTokenizerLoader(),
                configuration: configuration)
        }
        let residentResult = try await residentContainer.perform { context -> (tokens: [Int], decodedText: String, singleTokenLogits: [Float], prefillLogits: [Float], prefillMs: Double, tokPerSec: Double) in
            guard let inkling = context.model as? Inkling2 else {
                throw BenchmarkError.unexpectedModelType
            }
            XCTAssertFalse(inkling.isStreamingExpertsActive)
            let promptIDs = context.tokenizer.encode(text: promptText, addSpecialTokens: true)
            let promptArray = MLXArray(promptIDs.map(Int32.init)).reshaped(1, promptIDs.count)

            let singleCache = try inkling.newCache(parameters: GenerateParameters(maxKVSize: 4_096))
            let singleTokenArr = MLXArray([Int32(promptIDs[0])]).reshaped(1, 1)
            let singleLogitsArr = inkling(singleTokenArr, cache: singleCache).reshaped(-1).asType(.float32)
            eval(singleLogitsArr)
            let singleLogits = singleLogitsArr.asArray(Float.self)

            let prefillCheckCache = try inkling.newCache(parameters: GenerateParameters(maxKVSize: 4_096))
            let prefillPrepared = try inkling.prepare(
                LMInput(tokens: promptArray),
                cache: prefillCheckCache,
                state: nil,
                prefill: PrefillParameters(stepSize: cfg16.prefillChunkSize))
            guard case .logits(let prefillOut) = prefillPrepared else {
                throw BenchmarkError.unexpectedModelType
            }
            let prefillLogitsArr = prefillOut.logits.reshaped(-1).asType(.float32)
            eval(prefillLogitsArr)
            let prefillLogits = prefillLogitsArr.asArray(Float.self)

            let cache = try inkling.newCache(parameters: GenerateParameters(maxKVSize: 4_096))
            var iter = try TokenIterator(
                input: LMInput(tokens: promptArray),
                model: context.model,
                cache: cache,
                processor: nil,
                sampler: ArgMaxSampler(),
                prefill: PrefillParameters(stepSize: cfg16.prefillChunkSize),
                maxTokens: 32)
            let resPrefillMs = iter.promptPrefillTime * 1000.0
            var ids: [Int] = []
            var rt0: UInt64 = 0
            var rtLast: UInt64 = 0
            while let tok = iter.next() {
                let now = DispatchTime.now().uptimeNanoseconds
                if rt0 == 0 { rt0 = now }
                rtLast = now
                ids.append(tok)
            }
            Stream.gpu.synchronize()
            let resDecodeMs = Double(rtLast &- rt0) / 1_000_000.0
            let resTps = (ids.count > 1 && resDecodeMs > 0) ? Double(ids.count - 1) / (resDecodeMs / 1000.0) : 0
            let decodedText = context.tokenizer.decode(tokenIds: ids)
            return (ids, decodedText, singleLogits, prefillLogits, resPrefillMs, resTps)
        }

        let singleTokenMaxLogitDiff = zip(streamedResult.singleTokenLogits, residentResult.singleTokenLogits)
            .map { abs($0 - $1) }
            .max() ?? 0
        let prefillDiffs = zip(streamedResult.prefillLogits, residentResult.prefillLogits).map { abs($0 - $1) }
        let prefillMaxLogitDiff = prefillDiffs.max() ?? 0
        let prefillMeanAbsLogitDiff = prefillDiffs.isEmpty ? 0 : (prefillDiffs.reduce(0, +) / Float(prefillDiffs.count))
        let sortedResPrefill = residentResult.prefillLogits.sorted(by: >)
        let prefillTopMargin = sortedResPrefill.count >= 2 ? (sortedResPrefill[0] - sortedResPrefill[1]) : 0

        let residentIDs = residentResult.tokens
        let exactMatchCount = zip(streamedResult.tokens, residentIDs).filter { $0 == $1 }.count
        let prefixMatch = zip(streamedResult.tokens, residentIDs).prefix { $0 == $1 }.count
        print("""
            INKLING2_REAL_SSD[slots=16] postLoadActiveMB=\(postLoadActive16 / 1_048_576) postLoadFootprintMB=\(postLoadFootprint16 / 1_048_576) \
            steadyActiveMB=\(streamedResult.activeAfter / 1_048_576) steadyFootprintMB=\(streamedResult.footprintAfter / 1_048_576) \
            singleTokenMaxLogitDiff=\(singleTokenMaxLogitDiff) prefillMaxLogitDiff=\(prefillMaxLogitDiff) \
            prefillMeanAbsLogitDiff=\(String(format: "%.4f", prefillMeanAbsLogitDiff)) prefillTopMargin=\(prefillTopMargin) \
            prefillMs=\(String(format: "%.1f", streamedResult.prefillMs)) prefillIoMs=\(String(format: "%.1f", streamedResult.prefillIoMs)) \
            decodeTokPerSec=\(String(format: "%.2f", streamedResult.tokPerSec)) decodeIoMsPerTok=\(String(format: "%.2f", streamedResult.decodeIoMsPerTok)) \
            decodeHitRate=\(String(format: "%.1f", streamedResult.decodeHitRate * 100))% \
            warmPrefillMs=\(String(format: "%.1f", streamedResult.warmPrefillMs)) warmPrefillIoMs=\(String(format: "%.1f", streamedResult.warmPrefillIoMs)) \
            warmDecodeTokPerSec=\(String(format: "%.2f", streamedResult.warmTokPerSec)) warmDecodeIoMsPerTok=\(String(format: "%.2f", streamedResult.warmDecodeIoMsPerTok)) \
            warmDecodeHitRate=\(String(format: "%.1f", streamedResult.warmDecodeHitRate * 100))% \
            hitRate=\(String(format: "%.1f", streamedResult.stats.hitRate * 100))% alignedReadMB=\(String(format: "%.1f", Double(streamedResult.stats.alignedBytesRead) / 1e6)) \
            greedyPrefix=\(prefixMatch)/\(residentIDs.count) exactMatches=\(exactMatchCount)/\(residentIDs.count)
            """)
        print("INKLING2_REAL_RESIDENT prefillMs=\(String(format: "%.1f", residentResult.prefillMs)) decodeTokPerSec=\(String(format: "%.2f", residentResult.tokPerSec))")
        print("INKLING2_REAL_SSD_TEXT streamed: \(streamedResult.decodedText)")
        print("INKLING2_REAL_SSD_TEXT resident: \(residentResult.decodedText)")
        fflush(stdout)
        XCTAssertLessThan(singleTokenMaxLogitDiff, 1e-4, "Single-token (decode-path) real-weight logit diff too high: \(singleTokenMaxLogitDiff)")
        // On ~20-token prefill, MLX selects `gather_qmm_rhs` for streamed (`B / E >= 4` at S=16) vs `gather_qmv` for resident (`B / E < 4` at E=128) across 48 looped BF16 visits.
        XCTAssertLessThan(prefillMeanAbsLogitDiff, 0.35, "Post-prefill mean abs logit diff too high: \(prefillMeanAbsLogitDiff)")
        XCTAssertLessThan(prefillMaxLogitDiff, 2.0, "Post-prefill max logit diff too high: \(prefillMaxLogitDiff)")
        XCTAssertEqual(streamedResult.tokens.first, residentIDs.first, "First greedy token diverged from resident")
        XCTAssertGreaterThanOrEqual(prefixMatch, 16, "Greedy prefix diverged early: \(prefixMatch)/\(residentIDs.count)")
    }

    // MARK: - Helpers for Synthetic Safetensors & Runtime Tests

    private final class DummyLoadedEngine: ChatEngine, @unchecked Sendable {
        let id: String
        var isLoaded: Bool = true
        init(id: String) { self.id = id }
        func load(progress: @escaping @Sendable (Double) -> Void) async throws { isLoaded = true }
        func unload() { isLoaded = false }
        func stream(
            history: [EngineMessage], systemPrompt: String, temperature: Double, conversationID: UUID
        ) -> AsyncThrowingStream<EngineEvent, Error> {
            AsyncThrowingStream { $0.finish() }
        }
    }

    private static func processPhysFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<integer_t>.stride)
        let kr = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        return kr == KERN_SUCCESS ? info.phys_footprint : 0
    }

    private static func buildSyntheticLoopedCheckpoint(
        in directory: URL
    ) throws -> (Inkling2Configuration, Inkling2) {
        try Data(Inkling2ModelTests.tinyJSON.utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(
            Inkling2Configuration.self, from: Data(Inkling2ModelTests.tinyJSON.utf8))
        let text = config.textConfiguration
        let resident = Inkling2(config)

        MLXRandom.seed(42)
        var parameters: [String: MLXArray] = [:]
        for (key, value) in resident.parameters().flattened() where value.dtype == .float32 {
            parameters[key] = MLXRandom.normal(value.shape) * 0.25
        }
        let float32Suffixes = [
            ".query", "mlp.gate.weight", "e_score_correction_bias",
            "forget_gate_proj.weight", "message_gate_proj.weight", ".global_scale",
        ]
        for (key, value) in parameters
        where value.dtype == .float32 && !float32Suffixes.contains(where: key.hasSuffix) {
            parameters[key] = value.asType(.bfloat16)
        }

        // Layers 0..<4 use NVFP4 (bits=4, groupSize=16) with global_scale; layers 4..<6 use MXFP8 (bits=8, groupSize=32).
        for layer in 0 ..< text.hiddenLayers {
            let prefix = "language_model.layers.\(layer).mlp.experts"
            let useMXFP8 = layer >= 4
            let mode: QuantizationMode = useMXFP8 ? .mxfp8 : .nvfp4
            let bits = useMXFP8 ? 8 : 4
            let groupSize = useMXFP8 ? 32 : 16

            for (proj, inDims, outDims) in [
                ("gate_up_proj", text.hiddenSize, 2 * text.intermediateSize),
                ("down_proj", text.intermediateSize, text.hiddenSize),
            ] {
                let dense = (MLXRandom.normal([text.experts, outDims, inDims]) * 0.2).asType(.bfloat16)
                let (w, s, _) = MLX.quantized(dense, groupSize: groupSize, bits: bits, mode: mode)
                parameters["\(prefix).\(proj).weight"] = w
                parameters["\(prefix).\(proj).scales"] = s
                if !useMXFP8 {
                    parameters["\(prefix).\(proj).global_scale"] = MLXArray.ones([text.experts], dtype: .float32)
                } else {
                    parameters.removeValue(forKey: "\(prefix).\(proj).global_scale")
                }
            }
        }

        // Persist in real Inkling-2-Mini-4bit layout ("model.<key>") to verify checkpoint mapping too.
        var diskArrays: [String: MLXArray] = [:]
        for (k, v) in parameters {
            eval(v)
            diskArrays["model.\(k)"] = v
        }
        try MLX.save(arrays: diskArrays, url: directory.appendingPathComponent("model.safetensors"))

        try loadWeights(modelDirectory: directory, model: resident)
        eval(resident)
        return (config, resident)
    }
}
