import Foundation
import Metal
import Testing
import TurboFieldfare
@testable import MLXChat

/// Live decode benchmark for the Flint SSD runner. Opt-in
/// (`MLXCHAT_RUN_FLINT_SSD_BENCH=1`) because it needs the 22.6 GB install.
/// Prints greedy tokens so runs before and after a change can be compared.
@Suite(.serialized) struct FlintSSDBenchmarkTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXCHAT_RUN_FLINT_SSD_BENCH"] == "1"
    }

    private static let directory = StreamedMoEInstall.flint.managedDirectory

    private static func argmax(_ buffer: MTLBuffer, vocab: Int) -> Int32 {
        let p = buffer.contents().assumingMemoryBound(to: Float16.self)
        var best = 0
        for i in 1..<vocab where p[i] > p[best] { best = i }
        return Int32(best)
    }

    /// GPU time of one Flint decode attention call (32 query heads, 4 KV
    /// heads, dim 128) at several cache lengths.
    static func attentionMicros(_ context: MetalContext) throws -> String {
        let attention = try FlintDecodeAttention(context: context, numQueryHeads: 32)
        let device = context.device
        let maxLength = 16_384, kvStride = 4 * 128
        func zeroed(_ bytes: Int) throws -> MTLBuffer {
            let buffer = try #require(device.makeBuffer(length: bytes, options: .storageModeShared))
            memset(buffer.contents(), 0, bytes)
            return buffer
        }
        let q = try zeroed(32 * 128 * 2), out = try zeroed(32 * 128 * 2), sinks = try zeroed(64)
        let k = try zeroed(maxLength * kvStride * 2), v = try zeroed(maxLength * kvStride * 2)
        var results: [String] = []
        for length in [128, 513, 4_096, 16_384] {
            let command = try #require(context.queue.makeCommandBuffer())
            for _ in 0..<20 {
                attention.encode(commandBuffer: command, q: q, k: k, v: v,
                                 sinks: sinks, sinksOffset: 0, out: out,
                                 numKVHeads: 4, seqLen: length, kvStart: 0, scale: 0.088)
            }
            command.commit()
            command.waitUntilCompleted()
            let micros = (command.gpuEndTime - command.gpuStartTime) / 20 * 1e6
            results.append("\(length): \(String(format: "%.0f", micros))")
        }
        return results.joined(separator: ", ")
    }

    @Test(.timeLimit(.minutes(30)))
    func greedyDecodeSpeedAndTokens() async throws {
        guard Self.enabled,
              FileManager.default.fileExists(
                atPath: Self.directory.appendingPathComponent("manifest.json").path)
        else { return }

        let tokenizer = try await GFTokenizer.load(forModelDirectory: Self.directory)
        let system = "You are a helpful assistant."
        let question = "How many r's are in the word Strawberry?"
        let rendered = GFTokenizer.flintPrompt(turns: [("user", question)], system: system)
        // Rendered with Python jinja2 (trim_blocks, lstrip_blocks) from the
        // checkpoint's chat_template.jinja.
        #expect(rendered == "<|system|>\(system)<|user|>\(question)<|assistant|>")
        let ids = tokenizer.encode(rendered, addBOS: false)

        let steps = Int(ProcessInfo.processInfo.environment["MLXCHAT_FLINT_BENCH_STEPS"] ?? "") ?? 96
        let environment = ProcessInfo.processInfo.environment
        let slots = Int(environment["MLXCHAT_FLINT_BENCH_SLOTS"] ?? "") ?? 16
        let context = try MetalContext()
        let model = try Model.load(directoryURL: Self.directory, device: context.device,
                                   streamingMode: .pread(slotCount: slots),
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let runner = try FlintForwardRunner(model: model, context: context, maxContext: 4_096)
        let vocab = model.config.vocabSize
        let logits = try #require(context.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.size, options: .storageModeShared))

        let prefillStart = Date()
        for (position, token) in ids.enumerated() {
            try await runner.produce(token: token, position: position, into: logits)
        }
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        let first = Self.argmax(logits, vocab: vocab)

        // Rewind and replay the last prompt token: the greedy pick must not move.
        try runner.prepareForContinuation(expectedPosition: ids.count - 1)
        try await runner.produce(token: ids.last!, position: ids.count - 1, into: logits)
        #expect(Self.argmax(logits, vocab: vocab) == first)

        var tokens: [Int32] = [first]
        runner.prefetchesNextLayer = environment["MLXCHAT_FLINT_BENCH_PREFETCH"] != "0"
        if let width = Int(environment["MLXCHAT_FLINT_BENCH_PREFETCH_WIDTH"] ?? "") {
            runner.prefetchWidth = width
        }
        runner.resetProfile()
        let decodeStart = Date()
        for step in 1..<steps {
            try await runner.produce(token: tokens[step - 1],
                                     position: ids.count + step - 1, into: logits)
            tokens.append(Self.argmax(logits, vocab: vocab))
        }
        let seconds = Date().timeIntervalSince(decodeStart)
        let profile = runner.profile
        func ms(_ nanos: UInt64) -> String {
            String(format: "%.2f", Double(nanos) / 1e6 / Double(max(profile.tokens, 1)))
        }
        print("""
            FLINT_BENCH prompt \(ids.count) tokens in \(String(format: "%.1f", prefillSeconds)) s \
            (\(String(format: "%.2f", Double(ids.count) / prefillSeconds)) tok/s), \
            decode \(String(format: "%.2f", Double(steps - 1) / seconds)) tok/s
            FLINT_BENCH per token ms: total \(ms(profile.totalNanos)) gpuWait \(ms(profile.gpuWaitNanos)) \
            expertIO \(ms(profile.expertIONanos)) router \(ms(profile.routerNanos)) \
            gpuBusy \(ms(profile.gpuBusyNanos))
            FLINT_BENCH slots \(slots) hits \(profile.expertHits) misses \(profile.expertMisses) \
            speculation \(profile.speculatedCorrect)/\(profile.speculatedTotal) \
            prefetchWait \(ms(profile.prefetchWaitNanos)) missFetches \(profile.missFetches)
            FLINT_BENCH gpu by stage ms: \(profile.gpuByStage.sorted { $0.key < $1.key }.map { "\($0.key) \(ms($0.value))" }.joined(separator: ", "))
            FLINT_BENCH tokens \(tokens.map(String.init).joined(separator: ","))
            FLINT_BENCH text \(tokenizer.decode(tokens).debugDescription)
            """)
        #expect(!tokens.isEmpty)

        // I/O floor: an all-hit fetch measures the queue hop alone; a
        // repeated read of one expert measures a page-cache copy.
        let hits = try #require(try model.planRoutedExperts(layer: 10, experts: [0]))
        _ = try await model.fetchRoutedExperts(plan: hits)
        let hopStart = Date()
        for _ in 0..<200 {
            let plan = try #require(try model.planRoutedExperts(layer: 10, experts: [0]))
            _ = try await model.fetchRoutedExperts(plan: plan)
        }
        let hopMicros = Date().timeIntervalSince(hopStart) / 200 * 1e6
        let file = Self.directory.appendingPathComponent("packed_experts/layer_12.bin").path
        let fd = open(file, O_RDONLY)
        defer { close(fd) }
        let stride = 1_769_472
        let scratch = UnsafeMutableRawPointer.allocate(byteCount: stride, alignment: 16_384)
        defer { scratch.deallocate() }
        _ = pread(fd, scratch, stride, 0)
        let copyStart = Date()
        for _ in 0..<200 { _ = pread(fd, scratch, stride, 0) }
        let copyMicros = Date().timeIntervalSince(copyStart) / 200 * 1e6
        print("FLINT_BENCH io floor: hop \(String(format: "%.0f", hopMicros)) us, cached expert read \(String(format: "%.0f", copyMicros)) us")

        print("FLINT_BENCH attention us: " + (try Self.attentionMicros(context)))
    }

    /// Deterministic, varied English-like text of roughly `words` words.
    static func longText(words: Int) -> String {
        let subjects = ["The engineer", "A small team", "The lighthouse keeper", "Our compiler",
                        "The river", "A tired student", "The orchestra", "Every merchant"]
        let verbs = ["measured", "rebuilt", "questioned", "carried", "painted", "translated",
                     "repaired", "remembered", "described", "tested"]
        let objects = ["the northern bridge", "a broken clock", "seven old letters",
                       "the harbour lights", "an unfinished map", "the winter harvest",
                       "a quiet melody", "the ledger of debts", "the copper pipes"]
        let tails = ["before the storm arrived.", "while the city slept.",
                     "with surprising patience.", "for the third time that year.",
                     "although nobody asked.", "under a pale green sky.",
                     "and wrote down every number.", "until the candles burned out."]
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next(_ count: Int) -> Int {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Int((state >> 33) % UInt64(count))
        }
        var text = ""
        var written = 0
        while written < words {
            let sentence = "\(subjects[next(subjects.count)]) \(verbs[next(verbs.count)]) "
                + "\(objects[next(objects.count)]) \(tails[next(tails.count)]) "
            text += sentence
            written += sentence.split(separator: " ").count
            if next(6) == 0 { text += "\n\n" }
        }
        return text
    }

    /// Prefill speed on a long prompt through `runRawPrefill`, the path the
    /// app's prompt warm-up and chunked prefill use.
    @Test(.timeLimit(.minutes(30)))
    func longPromptPrefillSpeed() async throws {
        guard Self.enabled,
              FileManager.default.fileExists(
                atPath: Self.directory.appendingPathComponent("manifest.json").path)
        else { return }
        let tokenizer = try await GFTokenizer.load(forModelDirectory: Self.directory)
        let words = Int(ProcessInfo.processInfo.environment["MLXCHAT_FLINT_PREFILL_WORDS"] ?? "") ?? 1_200
        let rendered = GFTokenizer.flintPrompt(
            turns: [("user", "Summarise this text.\n\n" + Self.longText(words: words))],
            system: "You are a helpful assistant.")
        let ids = tokenizer.encode(rendered, addBOS: false)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: Self.directory, device: context.device,
                                   streamingMode: .pread(slotCount: 16),
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let runner = try FlintForwardRunner(model: model, context: context, maxContext: 4_096)
        if let chunk = Int(ProcessInfo.processInfo.environment["MLXCHAT_FLINT_PREFILL_CHUNK"] ?? "") {
            runner.prefillChunkTokens = chunk
        }
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))
        runner.resetProfile()
        let started = Date()
        try await runRawPrefill(producer: runner, promptIds: ids, scratch: scratch)
        let seconds = Date().timeIntervalSince(started)
        let profile = runner.profile
        func ms(_ nanos: UInt64) -> String { String(format: "%.0f", Double(nanos) / 1e6) }
        print("""
            FLINT_PREFILL \(ids.count) tokens in \(String(format: "%.2f", seconds)) s \
            (\(String(format: "%.1f", Double(ids.count) / seconds)) tok/s); totals ms: \
            gpuWait \(ms(profile.gpuWaitNanos)) gpuBusy \(ms(profile.gpuBusyNanos)) \
            expertIO \(ms(profile.expertIONanos)) router \(ms(profile.routerNanos)) \
            hits \(profile.expertHits) misses \(profile.expertMisses) chunk \(runner.prefillChunkTokens)
            FLINT_PREFILL gpu by stage ms: \(profile.gpuByStage.sorted { $0.key < $1.key }.map { "\($0.key) \(ms($0.value))" }.joined(separator: ", "))
            """)
        #expect(runner.continuationPosition == ids.count)
    }

    /// Batched prefill against token-by-token `produce` on a long prompt with
    /// a ring small enough to wrap: per-layer key-cache error, last-token
    /// logits and greedy continuation. Different chunk sizes must agree exactly.
    @Test(.timeLimit(.minutes(30)))
    func batchedPrefillMatchesTokenByToken() async throws {
        guard Self.enabled,
              FileManager.default.fileExists(
                atPath: Self.directory.appendingPathComponent("manifest.json").path)
        else { return }
        let tokenizer = try await GFTokenizer.load(forModelDirectory: Self.directory)
        let rendered = GFTokenizer.flintPrompt(
            turns: [("user", "Summarise this text.\n\n" + Self.longText(words: 1_100))],
            system: "You are a helpful assistant.")
        let ids = tokenizer.encode(rendered, addBOS: false)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: Self.directory, device: context.device,
                                   streamingMode: .pread(slotCount: 16),
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let vocab = model.config.vocabSize
        // A 64-slot margin makes the rings wrap many times over the prompt.
        let margin = Int(ProcessInfo.processInfo.environment["MLXCHAT_FLINT_PREFILL_MARGIN"] ?? "") ?? 64
        func makeRunner() throws -> FlintForwardRunner {
            try FlintForwardRunner(model: model, context: context, maxContext: 4_096,
                                   slidingRewindMargin: margin)
        }
        func logitsBuffer() throws -> MTLBuffer {
            try #require(context.device.makeBuffer(length: vocab * 2, options: .storageModeShared))
        }
        func read(_ buffer: MTLBuffer) -> [Float] {
            let p = buffer.contents().assumingMemoryBound(to: Float16.self)
            return (0..<vocab).map { Float(p[$0]) }
        }
        func greedy(_ runner: FlintForwardRunner, from logits: MTLBuffer, steps: Int) async throws -> [Int32] {
            var tokens: [Int32] = []
            var current = Self.argmax(logits, vocab: vocab)
            for _ in 0..<steps {
                tokens.append(current)
                try await runner.produce(token: current, position: runner.continuationPosition,
                                         into: logits)
                current = Self.argmax(logits, vocab: vocab)
            }
            return tokens
        }

        // Reference: every prompt token through `produce`.
        let reference = try makeRunner()
        let referenceLogits = try logitsBuffer()
        let referenceStart = Date()
        for (position, token) in ids.enumerated() {
            try await reference.produce(token: token, position: position, into: referenceLogits)
        }
        let referenceSeconds = Date().timeIntervalSince(referenceStart)

        // Batched, at three chunk sizes.
        var batched: [Int: (runner: FlintForwardRunner, logits: MTLBuffer, seconds: Double)] = [:]
        let chunks = (ProcessInfo.processInfo.environment["MLXCHAT_FLINT_PREFILL_CHUNKS"] ?? "64,33,17")
            .split(separator: ",").compactMap { Int($0) }
        for chunk in chunks {
            let runner = try makeRunner()
            runner.prefillChunkTokens = chunk
            let chunkScratch = try RawCompletionScratch(
                context: context, vocab: vocab,
                logitSoftcap: Float(model.config.finalLogitSoftcap))
            let started = Date()
            try await runRawPrefill(producer: runner, promptIds: ids, scratch: chunkScratch)
            let seconds = Date().timeIntervalSince(started)
            let logits = try logitsBuffer()
            memcpy(logits.contents(), chunkScratch.logitsBuffer.contents(), vocab * 2)
            batched[chunk] = (runner, logits, seconds)
        }
        let main = try #require(batched[chunks[0]])

        // Key cache, layer by layer, over the positions every ring still holds.
        var worstLayer = (layer: -1, maxError: Float(0))
        var report: [String] = []
        for layer in 0..<model.config.numLayers {
            var maxError: Float = 0
            var sum: Float = 0
            var n = 0
            for position in max(0, ids.count - 500)..<ids.count {
                guard let a = reference.keyCacheRow(layer: layer, position: position),
                      let b = main.runner.keyCacheRow(layer: layer, position: position) else { continue }
                for i in a.indices {
                    let e = abs(Float(a[i]) - Float(b[i]))
                    maxError = max(maxError, e)
                    sum += e
                    n += 1
                }
            }
            if maxError > worstLayer.maxError { worstLayer = (layer, maxError) }
            if layer < 8 || layer % 8 == 0 || layer == model.config.numLayers - 1 {
                report.append("L\(layer) max \(String(format: "%.3f", maxError)) mean \(String(format: "%.4f", sum / Float(max(n, 1))))")
            }
        }

        // First layer and position whose cached keys hold inf or NaN.
        var firstBad = "none"
        scan: for layer in 0..<model.config.numLayers {
            for position in 0..<ids.count {
                if let row = main.runner.keyCacheRow(layer: layer, position: position),
                   row.contains(where: { !$0.isFinite }) {
                    firstBad = "L\(layer) p\(position)"
                    break scan
                }
            }
        }
        print("PREFILL_CHECK first non-finite key: \(firstBad)")
        let a = read(referenceLogits), b = read(main.logits)
        func top(_ x: [Float], _ k: Int) -> [Int] { Array(x.indices.sorted { x[$0] > x[$1] }.prefix(k)) }
        let overlap = Set(top(a, 5)).intersection(top(b, 5)).count
        let meanA = a.reduce(0, +) / Float(vocab), meanB = b.reduce(0, +) / Float(vocab)
        var cov: Float = 0, varA: Float = 0, varB: Float = 0
        for i in a.indices {
            cov += (a[i] - meanA) * (b[i] - meanB)
            varA += (a[i] - meanA) * (a[i] - meanA)
            varB += (b[i] - meanB) * (b[i] - meanB)
        }
        let correlation = cov / (varA.squareRoot() * varB.squareRoot())
        let referenceGreedy = try await greedy(reference, from: referenceLogits, steps: 32)
        let batchedGreedy = try await greedy(main.runner, from: main.logits, steps: 32)
        let matching = zip(referenceGreedy, batchedGreedy).prefix { $0 == $1 }.count

        // Chunk sizes 33 and 17 against 64, exactly.
        var chunkDiffs: [String] = []
        for chunk in chunks.dropFirst() {
            let other = try #require(batched[chunk])
            let x = read(other.logits)
            chunkDiffs.append("\(chunk): logits equal \(x == b)")
            #expect(x == b, "chunk \(chunk) logits differ")
        }
        print("""
            PREFILL_CHECK \(ids.count) tokens; token-by-token \(String(format: "%.1f", Double(ids.count) / referenceSeconds)) tok/s, \
            batched \(chunks[0]) \(String(format: "%.1f", Double(ids.count) / main.seconds)) tok/s, \
            others \(batched.filter { $0.key != chunks[0] }.map { "\($0.key): \(String(format: "%.1f", Double(ids.count) / $0.value.seconds))" }.joined(separator: " "))
            PREFILL_CHECK keys: \(report.joined(separator: ", ")); worst L\(worstLayer.layer) \(worstLayer.maxError)
            PREFILL_CHECK logits: top1 \(top(a, 1)) vs \(top(b, 1)), top5 overlap \(overlap)/5, correlation \(correlation), greedy match \(matching)/32
            PREFILL_CHECK chunks: \(chunkDiffs.joined(separator: ", "))
            PREFILL_CHECK reference text: \(tokenizer.decode(referenceGreedy).debugDescription)
            PREFILL_CHECK batched text:   \(tokenizer.decode(batchedGreedy).debugDescription)
            """)
        #expect(top(a, 1) == top(b, 1))
        #expect(overlap >= 4)
        #expect(correlation > 0.99)
    }

    /// Short-prompt crossover: after a warm-up turn, the same follow-up
    /// lengths through token-by-token and batched prefill, alternating.
    @Test(.timeLimit(.minutes(30)))
    func shortPromptPrefillCrossover() async throws {
        guard Self.enabled,
              FileManager.default.fileExists(
                atPath: Self.directory.appendingPathComponent("manifest.json").path)
        else { return }
        let tokenizer = try await GFTokenizer.load(forModelDirectory: Self.directory)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: Self.directory, device: context.device,
                                   streamingMode: .pread(slotCount: 16),
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))
        let words = Self.longText(words: 400).split(separator: " ").map(String.init)
        let warm = tokenizer.encode(GFTokenizer.flintPrompt(
            turns: [("user", "Hello there, how are you today?")],
            system: "You are a helpful assistant."), addBOS: false)
        var lines: [String] = []
        for length in [8, 16, 32, 64, 128, 256] {
            var seconds: [Bool: [Double]] = [:]
            for round in 0..<4 {
                let batched = round.isMultiple(of: 2) != (length % 16 == 0)
                let runner = try FlintForwardRunner(model: model, context: context, maxContext: 4_096)
                runner.batchedPrefillMinimumTokens = batched ? 0 : Int.max
                try await runRawPrefill(producer: runner, promptIds: warm, scratch: scratch)
                // A follow-up of about `length` tokens after the warm turn.
                var follow = tokenizer.encode(words.prefix(length).joined(separator: " "), addBOS: false)
                follow = Array(follow.prefix(length))
                let ids = warm + follow
                let started = Date()
                try await runRawPrefill(producer: runner, promptIds: ids, scratch: scratch,
                                        start: .resume(cachedPromptTokens: warm.count))
                seconds[batched, default: []].append(Date().timeIntervalSince(started))
            }
            func best(_ values: [Double]) -> String { String(format: "%.2f", values.min() ?? 0) }
            lines.append("\(length): token \(best(seconds[false] ?? [])) s, batched \(best(seconds[true] ?? [])) s")
        }
        print("PREFILL_CROSSOVER " + lines.joined(separator: " | "))
    }
}
