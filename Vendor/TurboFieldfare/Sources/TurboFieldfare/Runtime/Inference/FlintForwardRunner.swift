import Accelerate
import Foundation
import Metal

/// Top-6 sigmoid router. `expert_bias` changes which experts are chosen and
/// is not part of the weights. The weights are the chosen sigmoid scores
/// divided by their sum.
public enum FlintRouter {
    public static func select(hidden: [Float],
                              weight: [Float],
                              bias: [Float],
                              experts: Int,
                              hiddenSize: Int,
                              topK: Int) -> (indices: [Int], weights: [Float]) {
        let expertCount = max(experts, 0)
        let chosenCount = min(max(topK, 0), expertCount)
        guard expertCount > 0 else { return ([], []) }
        precondition(hiddenSize >= 0 && bias.count >= expertCount
                     && hidden.count >= hiddenSize
                     && weight.count >= expertCount * hiddenSize)

        let scores: [Float] = [Float](unsafeUninitializedCapacity: expertCount) { buffer, initializedCount in
            guard let base = buffer.baseAddress else { initializedCount = 0; return }
            base.initialize(repeating: 0, count: expertCount)
            if hiddenSize > 0 {
                weight.withUnsafeBufferPointer { w in
                    hidden.withUnsafeBufferPointer { h in
                        guard let wb = w.baseAddress, let hb = h.baseAddress else { return }
                        vDSP_mmul(wb, 1, hb, 1, base, 1,
                                  vDSP_Length(expertCount), 1, vDSP_Length(hiddenSize))
                    }
                }
            }
            var i = 0
            while i < expertCount { base[i] = 1 / (1 + exp(-base[i])); i &+= 1 }
            initializedCount = expertCount
        }
        let biased: [Float] = [Float](unsafeUninitializedCapacity: expertCount) { buffer, initializedCount in
            guard let base = buffer.baseAddress else { initializedCount = 0; return }
            base.initialize(repeating: 0, count: expertCount)
            scores.withUnsafeBufferPointer { s in
                bias.withUnsafeBufferPointer { b in
                    guard let sb = s.baseAddress, let bb = b.baseAddress else { return }
                    var i = 0
                    while i < expertCount { base[i] = sb[i] + bb[i]; i &+= 1 }
                }
            }
            initializedCount = expertCount
        }
        var ranked: [Int] = [Int](unsafeUninitializedCapacity: expertCount) { buffer, initializedCount in
            guard let base = buffer.baseAddress else { initializedCount = 0; return }
            base.initialize(repeating: 0, count: expertCount)
            var i = 0
            while i < expertCount { base[i] = i; i &+= 1 }
            initializedCount = expertCount
        }
        biased.withUnsafeBufferPointer { b in
            guard let bb = b.baseAddress else { return }
            ranked.withUnsafeMutableBufferPointer { r in
                r.sort { lhs, rhs in
                    bb[lhs] == bb[rhs] ? lhs < rhs : bb[lhs] > bb[rhs]
                }
            }
        }
        let chosen: [Int] = [Int](unsafeUninitializedCapacity: chosenCount) { buffer, initializedCount in
            guard chosenCount > 0, let base = buffer.baseAddress else { initializedCount = 0; return }
            base.initialize(repeating: 0, count: chosenCount)
            ranked.withUnsafeBufferPointer { r in
                guard let rb = r.baseAddress else { return }
                var i = 0
                while i < chosenCount { base[i] = rb[i]; i &+= 1 }
            }
            initializedCount = chosenCount
        }
        var scoreSum: Float = 0
        if chosenCount > 0 {
            scores.withUnsafeBufferPointer { s in
                chosen.withUnsafeBufferPointer { c in
                    guard let sb = s.baseAddress, let cb = c.baseAddress else { return }
                    var i = 0
                    while i < chosenCount { scoreSum += sb[cb[i]]; i &+= 1 }
                }
            }
        }
        let total = scoreSum + 1e-20
        let weights: [Float] = [Float](unsafeUninitializedCapacity: chosenCount) { buffer, initializedCount in
            guard chosenCount > 0, let base = buffer.baseAddress else { initializedCount = 0; return }
            base.initialize(repeating: 0, count: chosenCount)
            scores.withUnsafeBufferPointer { s in
                chosen.withUnsafeBufferPointer { c in
                    guard let sb = s.baseAddress, let cb = c.baseAddress else { return }
                    var i = 0
                    while i < chosenCount { base[i] = sb[cb[i]] / total; i &+= 1 }
                }
            }
            initializedCount = chosenCount
        }
        return (chosen, weights)
    }

    /// Allocation-free `select` for decode: the same arithmetic and the same
    /// output order, with a top-k scan in place of the full sort. `scores`
    /// holds `experts` floats; `indices` and `weights` hold `topK`.
    public static func select(hidden: UnsafePointer<Float>,
                              weight: UnsafePointer<Float>,
                              bias: UnsafePointer<Float>,
                              experts: Int,
                              hiddenSize: Int,
                              topK: Int,
                              scores: UnsafeMutablePointer<Float>,
                              indices: UnsafeMutablePointer<Int>,
                              weights: UnsafeMutablePointer<Float>) {
        let chosenCount = min(max(topK, 0), max(experts, 0))
        guard chosenCount > 0 else { return }
        scores.initialize(repeating: 0, count: experts)
        if hiddenSize > 0 {
            vDSP_mmul(weight, 1, hidden, 1, scores, 1,
                      vDSP_Length(experts), 1, vDSP_Length(hiddenSize))
        }
        var i = 0
        while i < experts { scores[i] = 1 / (1 + exp(-scores[i])); i &+= 1 }
        select(scores: scores, bias: bias, experts: experts, topK: topK,
               indices: indices, weights: weights)
    }

    /// The selection half of `select`, from sigmoid `scores` already computed
    /// (on the GPU in decode). Ranked by biased score, then by lower index.
    /// Weights divide by the chosen scores' sum plus `epsilon` (LFM2 uses 1e-6).
    public static func select(scores: UnsafePointer<Float>,
                              bias: UnsafePointer<Float>,
                              experts: Int,
                              topK: Int,
                              indices: UnsafeMutablePointer<Int>,
                              weights: UnsafeMutablePointer<Float>,
                              epsilon: Float = 1e-20) {
        let chosenCount = min(max(topK, 0), max(experts, 0))
        guard chosenCount > 0 else { return }
        var count = 0
        var i = 0
        while i < experts {
            let value = scores[i] + bias[i]
            if count < chosenCount || value > scores[indices[count - 1]] + bias[indices[count - 1]] {
                var slot = min(count, chosenCount - 1)
                while slot > 0 {
                    let above = indices[slot - 1]
                    if value > scores[above] + bias[above] {
                        indices[slot] = above
                        slot &-= 1
                    } else {
                        break
                    }
                }
                indices[slot] = i
                count = min(count + 1, chosenCount)
            }
            i &+= 1
        }
        var scoreSum: Float = 0
        i = 0
        while i < chosenCount { scoreSum += scores[indices[i]]; i &+= 1 }
        let total = scoreSum + epsilon
        i = 0
        while i < chosenCount { weights[i] = scores[indices[i]] / total; i &+= 1 }
    }
}

/// Decode runner for Flint (`ModelFamily.flint`).
///
/// Every layer is attention, then a residual, then either a dense SwiGLU
/// (the first `numDenseLayers`) or a shared expert plus the top-6 routed
/// experts. Full-attention layers have no rotary encoding and see the whole
/// cache. Sliding layers use NeoX rotary encoding and a window of 513
/// positions, including the current token. Each query head has a learned
/// attention sink.
///
/// `produce` runs one token. Prompts go through `prefillChunked`, which runs
/// chunks of tokens layer by layer (see `prefillChunk`). Rewinding overwrites
/// the key-value cache from the requested position, so no snapshot is stored.
///
/// Sliding layers keep their keys and values in a ring of
/// `slidingWindow + slidingRewindMargin` slots instead of one slot per context
/// position. A rewind may then go back at most `maxRewindDistance` positions,
/// so the window before the new position is still intact.
public final class FlintForwardRunner: ContinuableLogitProducer, ContextWindowReporting,
                                      @unchecked Sendable {
    public let maxContext: Int
    public private(set) var position: Int = 0
    public let supportsRewind = true

    private let model: Model
    private let ctx: MetalContext
    private let cfg: ArchConfig

    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let moe: MoE
    private let dense: SharedExpertInt4
    private let embedQ8: MTLComputePipelineState
    private let attention: FlintDecodeAttention
    private let fusedQKV: FusedQKVGEMV
    private let qkNormRope: MTLComputePipelineState
    private let residualNorm: MTLComputePipelineState
    private let routerGemv: MTLComputePipelineState
    private let clampInf: MTLComputePipelineState

    // Prompt prefill, layer by layer over a chunk of tokens.
    private let prefillQMM: PrefillInt4QMM
    /// Tensor-op int4 matrix multiply where the device compiles it; nil
    /// falls back to `prefillQMM`. Each output element accumulates in the
    /// same order whatever the token count, so chunking stays invisible.
    private let prefillTensorQMM: MPPPrefillInt4QMM?
    private let prefillRMS: PrefillRMSNorm
    private let groupedMoE: PrefillGroupedRoutedMoE
    private let moeReduce: PrefillMoE
    private let embedBlock: MTLComputePipelineState
    private let qkNormRopeBlock: MTLComputePipelineState
    private let routerGemvBlock: MTLComputePipelineState
    private let siluMul: MTLComputePipelineState
    private let residualNormBlock: MTLComputePipelineState
    private let attentionPrefill: MTLComputePipelineState
    private var prefillBuffers: PrefillBuffers?
    /// Tokens per prefill chunk. A chunk waits on the GPU once per routed layer
    /// and reads each expert its tokens need once, so larger chunks read less
    /// per token: a 2,672-token prompt ran at 234 tok/s with 512, 353 with
    /// 2,048. Scratch takes about 100 KB per chunk token and is freed after
    /// each prefill.
    public var prefillChunkTokens = 2_048 {
        didSet { prefillChunkTokens = min(max(prefillChunkTokens, 1), 2_048) }
    }
    /// Prompts shorter than this run token by token through the decode path.
    /// 0 batches everything: with a warm page cache, batched prefill was
    /// faster even for an 8-token follow-up (0.18 s against 0.32 s).
    public var batchedPrefillMinimumTokens = 0

    private let hidden: MTLBuffer
    private let normed: MTLBuffer
    private let query: MTLBuffer
    private let attnOut: MTLBuffer
    private let operatorOut: MTLBuffer
    private let ffnOut: MTLBuffer
    private let sharedOut: MTLBuffer
    private let moeActs: MTLBuffer
    private let routingWeights: MTLBuffer
    private let routerScoresBuffer: MTLBuffer
    private let prefetchScoresBuffer: MTLBuffer
    /// `expert_bias` per MoE layer as floats for the CPU top-k (zeros if absent).
    private var routerBias: [Int: [Float]] = [:]
    private var routerIndices: [Int]
    private var routerWeights: [Float]
    private let activeSlotsBuffer: MTLBuffer
    private let activeSlotIndices: [UInt32]
    private let scratchGate: MTLBuffer
    private let scratchUp: MTLBuffer
    private let scratchAct: MTLBuffer
    private let kCache: [Int: MTLBuffer]
    private let vCache: [Int: MTLBuffer]
    /// Slots in each sliding layer's ring; 0 when the ring would not be
    /// smaller than the context, so every layer holds every position.
    private let slidingRingCapacity: Int
    /// The farthest `prepareForContinuation` may move back from `position`.
    public var maxRewindDistance: Int {
        slidingRingCapacity == 0 ? Int.max : slidingRingCapacity - cfg.slidingWindow
    }

    /// Command buffers committed without a wait. The next attention wait
    /// orders them, so only their errors are checked later.
    private var pendingCommands: [MTLCommandBuffer] = []

    /// Wall time per decode phase, summed since the last `resetProfile()`.
    public struct Profile: Sendable {
        public var tokens = 0
        public var gpuWaitNanos: UInt64 = 0
        public var expertIONanos: UInt64 = 0
        public var routerNanos: UInt64 = 0
        public var totalNanos: UInt64 = 0
        public var gpuBusyNanos: UInt64 = 0
        public var expertHits = 0
        public var expertMisses = 0
        /// Prefetched experts the next layer chose, out of `speculatedTotal`.
        public var speculatedCorrect = 0
        public var speculatedTotal = 0
        public var prefetchWaitNanos: UInt64 = 0
        public var missFetches = 0
        /// GPU time by command buffer label prefix: attention, shared, routed.
        public var gpuByStage: [String: UInt64] = [:]
    }
    /// Reads layer L+1's likely experts while layer L runs. The guess applies
    /// layer L+1's router to layer L's input; a wrong guess costs only I/O.
    public var prefetchesNextLayer = true
    /// How many of layer L+1's top-ranked experts to prefetch.
    public var prefetchWidth = 8 {
        didSet { prefetchWidth = min(max(prefetchWidth, 1), cfg.numExperts) }
    }
    private var speculated: [Int] = []
    private var prefetches: [Int: Task<Void, Never>] = [:]
    public private(set) var profile = Profile()
    public func resetProfile() { profile = Profile() }

    private let eps: Float = 1e-6
    private let queriesPerKV: Int
    private static let streamedSlots = MoE.maxStreamedExperts

    public init(model: Model, context: MetalContext, maxContext: Int,
                slidingRewindMargin: Int = 4_096) throws {
        let cfg = model.config
        guard cfg.family == .flint else {
            throw ModelError.indexCorrupt(detail: "FlintForwardRunner needs a flint model")
        }
        guard cfg.headDim == 128, cfg.numHeads % cfg.numKVHeads == 0 else {
            throw ModelError.indexCorrupt(detail: "Flint attention kernel expects head dim 128")
        }
        guard maxContext > 0, maxContext <= cfg.maxPositionEmbeddings else {
            throw ModelError.contextExceedsModel(requested: maxContext,
                                                 maximum: cfg.maxPositionEmbeddings)
        }
        precondition(cfg.topKExperts <= Self.streamedSlots)
        self.model = model
        self.ctx = context
        self.cfg = cfg
        self.maxContext = maxContext
        self.queriesPerKV = cfg.numHeads / cfg.numKVHeads

        let D = cfg.hiddenSize
        let qDim = cfg.numHeads * cfg.headDim
        let kvDim = cfg.numKVHeads * cfg.headDim
        let wide = max(cfg.intermediateSize, cfg.sharedExpertIntermediateSize)
        self.rms = try RMSNorm(context: context)
        self.int4 = try DequantInt4GEMV(context: context, additionalShapes: [
            (m: qDim, n: D), (m: kvDim, n: D), (m: D, n: qDim),
            (m: cfg.intermediateSize, n: D), (m: D, n: cfg.intermediateSize),
            (m: cfg.sharedExpertIntermediateSize, n: D),
            (m: D, n: cfg.sharedExpertIntermediateSize),
        ])
        self.moe = try MoE(context: context,
                           siluActivation: true,
                           specializedD: UInt32(D),
                           specializedF: UInt32(cfg.moeIntermediateSize),
                           specializedNumExperts: UInt32(cfg.numExperts))
        self.dense = try SharedExpertInt4(context: context, siluActivation: true)
        self.embedQ8 = try context.pipeline("flint_embed_lookup_q8", privateModule: "flint")
        self.attention = try FlintDecodeAttention(context: context, numQueryHeads: cfg.numHeads)
        self.fusedQKV = try FusedQKVGEMV(context: context)
        self.qkNormRope = try context.pipeline("flint_qk_norm_rope", privateModule: "flint")
        self.residualNorm = try context.pipeline("flint_residual_rmsnorm", privateModule: "flint")
        self.routerGemv = try context.pipeline("flint_router_gemv_bf16", privateModule: "flint")
        self.clampInf = try context.pipeline("flint_clamp_inf_f16", privateModule: "flint")
        self.prefillQMM = try PrefillInt4QMM(context: context)
        let tensorQMM = MPPPrefillInt4QMM(context: context)
        self.prefillTensorQMM = tensorQMM.isAvailable ? tensorQMM : nil
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.groupedMoE = try PrefillGroupedRoutedMoE(context: context, siluActivation: true,
                                                      saturateFP16: true)
        self.moeReduce = try PrefillMoE(context: context)
        self.embedBlock = try context.pipeline("flint_embed_lookup_q8_block", privateModule: "flint")
        self.qkNormRopeBlock = try context.pipeline("flint_qk_norm_rope_block", privateModule: "flint")
        self.routerGemvBlock = try context.pipeline("flint_router_gemv_bf16_block",
                                                    privateModule: "flint")
        self.siluMul = try context.pipeline("flint_silu_mul", privateModule: "flint")
        self.residualNormBlock = try context.pipeline("flint_residual_rmsnorm_block",
                                                      privateModule: "flint")
        self.attentionPrefill = try context.pipeline("flint_attention_prefill", privateModule: "flint")

        let device = context.device
        func buf(_ count: Int, _ label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(count, 1) * MemoryLayout<Float16>.size,
                                                 options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = "flint.\(label)"
            return buffer
        }
        self.hidden = try buf(D, "hidden")
        self.normed = try buf(D, "normed")
        self.query = try buf(qDim, "query")
        self.attnOut = try buf(qDim, "attnOut")
        self.operatorOut = try buf(D, "operatorOut")
        self.ffnOut = try buf(D, "ffnOut")
        self.sharedOut = try buf(D, "sharedOut")
        self.moeActs = try buf(Self.streamedSlots * cfg.moeIntermediateSize, "moeActs")
        memset(moeActs.contents(), 0, moeActs.length)
        self.routingWeights = try buf(Self.streamedSlots, "routingWeights")
        memset(routingWeights.contents(), 0, routingWeights.length)
        guard let scoresBuf = device.makeBuffer(
            length: max(cfg.numExperts, 1) * MemoryLayout<Float>.size,
            options: .storageModeShared),
              let prefScoresBuf = device.makeBuffer(
                length: max(cfg.numExperts, 1) * MemoryLayout<Float>.size,
                options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        scoresBuf.label = "flint.routerScores"
        prefScoresBuf.label = "flint.prefetchScores"
        self.routerScoresBuffer = scoresBuf
        self.prefetchScoresBuffer = prefScoresBuf
        self.routerIndices = [Int](repeating: 0, count: cfg.numExperts)
        self.routerWeights = [Float](repeating: 0, count: cfg.numExperts)
        let slotIndices = (0..<UInt32(cfg.topKExperts)).map { $0 }
        self.activeSlotIndices = slotIndices
        guard let slotsBuf = slotIndices.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!,
                              length: max($0.count, MemoryLayout<UInt32>.size),
                              options: .storageModeShared)
        }) else {
            throw ModelError.residentBufferWrapFailed
        }
        slotsBuf.label = "flint.activeSlots"
        self.activeSlotsBuffer = slotsBuf
        self.scratchGate = try buf(wide, "scratchGate")
        self.scratchUp = try buf(wide, "scratchUp")
        self.scratchAct = try buf(wide, "scratchAct")

        let ring = cfg.slidingWindow + max(slidingRewindMargin, 0)
        self.slidingRingCapacity = cfg.slidingWindow > 0 && ring < maxContext ? ring : 0
        var keys: [Int: MTLBuffer] = [:]
        var values: [Int: MTLBuffer] = [:]
        for layer in 0..<cfg.numLayers {
            let slots = cfg.layerIsFull(layer) || slidingRingCapacity == 0
                ? maxContext : slidingRingCapacity
            let bytes = slots * kvDim * MemoryLayout<Float16>.size
            guard let key = device.makeBuffer(length: bytes, options: .storageModeShared),
                  let value = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw ModelError.kvAllocationFailed(layer: layer, bytes: bytes)
            }
            key.label = "flint.kv.K.layer\(layer)"
            value.label = "flint.kv.V.layer\(layer)"
            keys[layer] = key
            values[layer] = value
        }
        self.kCache = keys
        self.vCache = values
        for layer in cfg.numDenseLayers..<cfg.numLayers {
            // The router GEMV reads every BF16 row of the gate on the GPU.
            _ = try Self.checkedTensor(model: model, name: "model.layers.\(layer).mlp.gate.weight",
                                       bytes: cfg.numExperts * cfg.hiddenSize * 2)
            routerBias[layer] = try Self.loadRouterBias(model: model, layer: layer,
                                                        experts: cfg.numExperts)
        }
    }

    /// A resident tensor holding at least `bytes`, so raw and GPU reads of
    /// that many bytes stay inside it whatever the model file declares. The
    /// declared length and offset are checked against the buffer itself too.
    static func checkedTensor(model: Model, name: String, bytes: Int) throws -> TensorView {
        let view = try model.resident(name: name)
        let (end, overflow) = view.offset.addingReportingOverflow(UInt64(bytes))
        guard view.length >= UInt64(bytes), !overflow, end <= UInt64(view.buffer.length) else {
            throw ModelError.indexCorrupt(
                detail: "\(name) holds \(view.length) bytes at offset \(view.offset) of a \(view.buffer.length)-byte buffer; the model needs \(bytes)")
        }
        return view
    }

    private static func loadRouterBias(model: Model, layer: Int, experts: Int) throws -> [Float] {
        let name = "model.layers.\(layer).mlp.gate.expert_bias"
        guard model.residentIndex.entries[name] != nil else {
            return [Float](repeating: 0, count: experts)
        }
        let view = try checkedTensor(model: model, name: name, bytes: experts * 2)
        let bits = view.buffer.contents().advanced(by: Int(view.offset))
            .assumingMemoryBound(to: UInt16.self)
        return (0..<experts).map { Quantization.bf16ToFloat(bits[$0]) }
    }

    public func reset() {
        position = 0
    }

    /// The cached key row for `position` in `layer`, for verification. Nil
    /// when a sliding layer's ring no longer holds that position.
    public func keyCacheRow(layer: Int, position target: Int) -> [Float16]? {
        guard let buffer = kCache[layer], target >= 0, target < position else { return nil }
        let ring = cfg.layerIsFull(layer) ? 0 : slidingRingCapacity
        if ring > 0, target < position - ring { return nil }
        let kvDim = cfg.numKVHeads * cfg.headDim
        let slot = ring > 0 ? target % ring : target
        let row = buffer.contents().assumingMemoryBound(to: Float16.self) + slot * kvDim
        return Array(UnsafeBufferPointer(start: row, count: kvDim))
    }

    public var continuationPosition: Int { position }

    /// Continues at `expectedPosition`. Later cache slots are overwritten by
    /// the next tokens, and attention never reads past the current position.
    public func prepareForContinuation(expectedPosition: Int) throws {
        guard expectedPosition >= 0, expectedPosition <= position,
              position - expectedPosition <= maxRewindDistance else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected position \(expectedPosition), current \(position)")
        }
        position = expectedPosition
    }

    public func produce(token: Int32, position requested: Int, into logits: MTLBuffer) async throws {
        guard requested == position else {
            throw PrefillError.prefillCursorMismatch(
                "Flint decode expected position \(position), got \(requested)")
        }
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try await step(token: token, into: logits, emitsLogits: true)
        profile.tokens += 1
        profile.totalNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
    }

    /// `emitsLogits` false skips the output head: prompt tokens before the
    /// last need only their KV entries.
    private func step(token: Int32, into logits: MTLBuffer, emitsLogits: Bool) async throws {
        guard position < maxContext else {
            throw GeneratorError.contextOverflow(prompt: position, maxNew: 1,
                                                 maxContext: maxContext)
        }
        guard token >= 0, Int(token) < cfg.vocabSize else {
            throw ModelError.indexCorrupt(detail: "token \(token) outside the vocabulary")
        }
        // A step that threw may have left committed work behind; its error
        // was already reported.
        try? drainPendingCommands()
        speculated.removeAll(keepingCapacity: true)
        let D = UInt32(cfg.hiddenSize)
        let embedding = try resident("model.embed_tokens.weight")
        var command = try makeCommandBuffer("embed")
        encodeEmbedding(command, table: embedding, token: UInt32(token))

        let firstNorm = try resident("model.layers.0.input_layernorm.weight")
        rms.encodeBF16W(commandBuffer: command, x: hidden,
                        weight: firstNorm.buffer, weightOffset: Int(firstNorm.offset),
                        out: normed, d: D, eps: eps)

        // Each residual add also writes the next norm: post-attention after
        // attention, the next layer's input norm (or the final norm) after the FFN.
        for layer in 0..<cfg.numLayers {
            try encodeAttention(command, layer: layer)
            try encodeResidualNorm(command, delta: operatorOut,
                                   weight: "model.layers.\(layer).post_attention_layernorm.weight")
            let nextNorm = layer + 1 < cfg.numLayers
                ? "model.layers.\(layer + 1).input_layernorm.weight" : "model.norm.weight"
            if cfg.layerIsDenseFFN(layer) {
                try encodeDenseFFN(command, layer: layer, width: cfg.intermediateSize,
                                   namePrefix: "model.layers.\(layer).mlp")
                try encodeResidualNorm(command, delta: ffnOut, weight: nextNorm)
                continue
            }
            // Router scores come back with the layer's only wait.
            let didSpeculate = try encodeRouter(command, layer: layer)
            try commitAndWait(command)
            try drainPendingCommands()
            let routeStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let experts = try route(layer: layer, didSpeculate: didSpeculate)
            profile.routerNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - routeStart
            try startPrefetch(layer: layer + 1, experts: speculated)

            // The shared expert needs only `normed`, so it runs while the
            // routed experts are read from disk.
            let shared = try makeCommandBuffer("shared.layer\(layer)")
            try encodeDenseFFN(shared, layer: layer,
                               width: cfg.sharedExpertIntermediateSize,
                               namePrefix: "model.layers.\(layer).mlp.shared_expert",
                               destination: sharedOut)
            encodeClamp(shared, [(sharedOut, cfg.hiddenSize)])
            shared.commit()
            pendingCommands.append(shared)

            let routed = try makeCommandBuffer("moe.layer\(layer)")
            try await encodeRoutedExperts(routed, layer: layer, experts: experts)
            try encodeResidualNorm(routed, delta: ffnOut, weight: nextNorm)
            routed.commit()
            pendingCommands.append(routed)
            command = try makeCommandBuffer("layer\(layer + 1)")
        }

        if emitsLogits {
            let head = try resident("lm_head.weight")
            gemv(command, head, x: normed, y: logits, m: cfg.vocabSize, n: cfg.hiddenSize)
        }
        try commitAndWait(command)
        try drainPendingCommands()
        position += 1
    }

    /// `hidden += delta`, then `normed = RMSNorm(hidden) * weight`, saturating inf.
    private func encodeResidualNorm(_ command: MTLCommandBuffer, delta: MTLBuffer,
                                    weight name: String) throws {
        let weight = try resident(name)
        guard let encoder = command.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(residualNorm)
        encoder.setBuffer(hidden, offset: 0, index: 0)
        encoder.setBuffer(delta, offset: 0, index: 1)
        encoder.setBuffer(weight.buffer, offset: Int(weight.offset), index: 2)
        encoder.setBuffer(normed, offset: 0, index: 3)
        var count = UInt32(cfg.hiddenSize)
        var epsilon = eps
        encoder.setBytes(&count, length: 4, index: 4)
        encoder.setBytes(&epsilon, length: 4, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeEmbedding(_ command: MTLCommandBuffer, table: TensorView, token: UInt32) {
        guard let encoder = command.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(embedQ8)
        encoder.setBuffer(table.buffer, offset: Int(table.offset), index: 0)
        encoder.setBuffer(table.buffer, offset: Int(table.scaleOffset), index: 1)
        encoder.setBuffer(table.buffer, offset: Int(table.biasOffset), index: 2)
        encoder.setBuffer(hidden, offset: 0, index: 3)
        var tokenValue = token
        var dim = UInt32(cfg.hiddenSize)
        encoder.setBytes(&tokenValue, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 5)
        let width = min(embedQ8.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: cfg.hiddenSize, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeAttention(_ command: MTLCommandBuffer, layer: Int) throws {
        guard let key = kCache[layer], let value = vCache[layer] else {
            throw ModelError.indexCorrupt(detail: "layer \(layer) has no KV cache")
        }
        let D = cfg.hiddenSize
        let qDim = cfg.numHeads * cfg.headDim
        let kvDim = cfg.numKVHeads * cfg.headDim
        let ring = cfg.layerIsFull(layer) ? 0 : slidingRingCapacity
        let slot = (ring > 0 ? position % ring : position) * kvDim * MemoryLayout<Float16>.size
        let prefix = "model.layers.\(layer).self_attn."
        let q = try resident(prefix + "q_proj.weight")
        let k = try resident(prefix + "k_proj.weight")
        let v = try resident(prefix + "v_proj.weight")
        fusedQKV.encode(commandBuffer: command,
                        qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                        qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                        qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                        kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                        kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                        kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                        vWeights: v.buffer, vWeightsOffset: Int(v.offset),
                        vScales: v.buffer, vScalesOffset: Int(v.scaleOffset),
                        vBiases: v.buffer, vBiasesOffset: Int(v.biasOffset),
                        x: normed, qOut: query,
                        kOut: key, kOutOffset: slot, vOut: value, vOutOffset: slot,
                        qRows: UInt32(qDim), kvRows: UInt32(kvDim), n: UInt32(D))
        // Q/K head norms, plus rotary encoding on sliding layers, in one dispatch.
        let qNorm = try resident(prefix + "q_norm.weight")
        let kNorm = try resident(prefix + "k_norm.weight")
        if let encoder = command.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(qkNormRope)
            encoder.setBuffer(query, offset: 0, index: 0)
            encoder.setBuffer(key, offset: slot, index: 1)
            encoder.setBuffer(qNorm.buffer, offset: Int(qNorm.offset), index: 2)
            encoder.setBuffer(kNorm.buffer, offset: Int(kNorm.offset), index: 3)
            var queryHeads = UInt32(cfg.numHeads)
            var at = UInt32(position)
            var theta = Float(cfg.ropeTheta)
            var rotate: UInt32 = cfg.layerIsFull(layer) ? 0 : 1
            var epsilon = eps
            encoder.setBytes(&queryHeads, length: 4, index: 4)
            encoder.setBytes(&at, length: 4, index: 5)
            encoder.setBytes(&theta, length: 4, index: 6)
            encoder.setBytes(&rotate, length: 4, index: 7)
            encoder.setBytes(&epsilon, length: 4, index: 8)
            encoder.dispatchThreadgroups(
                MTLSize(width: cfg.numHeads + cfg.numKVHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: cfg.headDim / 2, height: 1, depth: 1))
            encoder.endEncoding()
        }
        let sinks = try resident(prefix + "softmax_offset")
        let length = position + 1
        let window = cfg.layerIsFull(layer) ? length : cfg.slidingWindow
        attention.encode(commandBuffer: command, q: query, k: key, v: value,
                         sinks: sinks.buffer, sinksOffset: Int(sinks.offset), out: attnOut,
                         numKVHeads: cfg.numKVHeads, seqLen: length,
                         kvStart: max(length - window, 0), scale: Float(cfg.attentionScale),
                         ringCapacity: ring)
        gemv(command, try resident(prefix + "o_proj.weight"),
             x: attnOut, y: operatorOut, m: D, n: qDim)
    }

    private func encodeDenseFFN(_ command: MTLCommandBuffer, layer: Int, width: Int,
                                namePrefix: String, destination: MTLBuffer? = nil) throws {
        let rows = UInt32(width)
        let columns = UInt32(cfg.hiddenSize)
        func projection(_ name: String, rows: UInt32, cols: UInt32) throws -> SharedExpertProjection {
            let view = try resident("\(namePrefix).\(name).weight")
            return SharedExpertProjection(weights: view.buffer, scales: view.buffer,
                                          biases: view.buffer,
                                          weightsOffset: Int(view.offset),
                                          scalesOffset: Int(view.scaleOffset),
                                          biasesOffset: Int(view.biasOffset),
                                          rows: rows, cols: cols)
        }
        try dense.encode(commandBuffer: command, x: normed,
                         gate: projection("gate_proj", rows: rows, cols: columns),
                         up: projection("up_proj", rows: rows, cols: columns),
                         down: projection("down_proj", rows: columns, cols: rows),
                         y: destination ?? ffnOut,
                         scratchGate: scratchGate, scratchUp: scratchUp, scratchAct: scratchAct)
    }

    /// Encodes router scores for `layer`, and for `layer + 1` from the same
    /// input when prefetching. Returns whether the second set was encoded.
    private func encodeRouter(_ command: MTLCommandBuffer, layer: Int) throws -> Bool {
        let nextLayer = layer + 1
        let canSpeculate = prefetchesNextLayer && nextLayer < cfg.numLayers
            && !cfg.layerIsDenseFFN(nextLayer)
        guard let encoder = command.makeComputeCommandEncoder() else { return false }
        encoder.setComputePipelineState(routerGemv)
        encoder.setBuffer(normed, offset: 0, index: 1)
        var numExperts = UInt32(cfg.numExperts)
        var hiddenDim = UInt32(cfg.hiddenSize)
        encoder.setBytes(&numExperts, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.size, index: 4)
        for (target, scores) in [(layer, routerScoresBuffer), (nextLayer, prefetchScoresBuffer)]
            where target == layer || canSpeculate {
            let gate = try resident("model.layers.\(target).mlp.gate.weight")
            encoder.setBuffer(gate.buffer, offset: Int(gate.offset), index: 0)
            encoder.setBuffer(scores, offset: 0, index: 2)
            encoder.dispatchThreadgroups(
                MTLSize(width: (cfg.numExperts + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        encoder.endEncoding()
        return canSpeculate
    }

    /// Top-k on the CPU from the GPU scores. Writes the routing weights and
    /// the next layer's guess into `speculated`.
    private func route(layer: Int, didSpeculate: Bool) throws -> [Int] {
        let experts = cfg.numExperts, topK = cfg.topKExperts
        func select(_ scores: MTLBuffer, layer: Int, count: Int) throws {
            guard let bias = routerBias[layer] else {
                throw ModelError.indexCorrupt(detail: "layer \(layer) has no router")
            }
            bias.withUnsafeBufferPointer { bias in
                routerIndices.withUnsafeMutableBufferPointer { indices in
                    routerWeights.withUnsafeMutableBufferPointer { weights in
                        FlintRouter.select(
                            scores: scores.contents().assumingMemoryBound(to: Float.self),
                            bias: bias.baseAddress!, experts: experts, topK: count,
                            indices: indices.baseAddress!, weights: weights.baseAddress!)
                    }
                }
            }
        }
        var next: [Int] = []
        if didSpeculate {
            try select(prefetchScoresBuffer, layer: layer + 1, count: prefetchWidth)
            next = Array(routerIndices.prefix(prefetchWidth))
        }
        try select(routerScoresBuffer, layer: layer, count: topK)
        let chosen = Array(routerIndices.prefix(topK))
        if !speculated.isEmpty {
            profile.speculatedCorrect += Set(speculated).intersection(chosen).count
            profile.speculatedTotal += topK
        }
        speculated = next
        let destination = routingWeights.contents().assumingMemoryBound(to: Float16.self)
        var slot = 0
        while slot < Self.streamedSlots {
            destination[slot] = slot < topK ? Float16(routerWeights[slot]) : 0
            slot &+= 1
        }
        return chosen
    }

    private func encodeRoutedExperts(_ command: MTLCommandBuffer, layer: Int,
                                     experts: [Int]) async throws {
        let packed = layer - cfg.numDenseLayers
        let ioStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        if let pending = prefetches.removeValue(forKey: layer) {
            let waitStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            await pending.value
            profile.prefetchWaitNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - waitStart
        }
        let plan = try model.planRoutedExperts(layer: packed, experts: experts)!
        profile.expertHits += plan.hits
        profile.expertMisses += plan.misses.count
        if !plan.misses.isEmpty { profile.missFetches += 1 }
        let blobs = plan.misses.isEmpty
            ? try model.routedExpertBuffers(for: plan)
            : try await model.fetchRoutedExperts(plan: plan)
        profile.expertIONanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - ioStart
        let unique = blobs.map(\.buffer)
        let routed = (0..<Self.streamedSlots).map { unique[$0 % unique.count] }
        let offsets = model.routedExpertOffsets(layer: packed)
        let topK = UInt32(Self.streamedSlots)
        let D = UInt32(cfg.hiddenSize)
        let width = UInt32(cfg.moeIntermediateSize)
        let arguments = moe.makeReusedRoutedArgumentBuffer(routedBlobs: routed, topK: topK)
        moe.encodeRoutedPersistentPhase1SubsetU16Load(
            commandBuffer: command,
            routedArgBuffer: arguments,
            routedBlobs: routed,
            routedOffsets: offsets,
            x: normed,
            acts: moeActs,
            activeSlots: activeSlotsBuffer,
            activeSlotIndices: activeSlotIndices,
            activeCount: UInt32(cfg.topKExperts),
            d: D, f: width, topK: topK)
        encodeClamp(command, [(moeActs, cfg.topKExperts * cfg.moeIntermediateSize)])
        moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: command,
                                               routedArgBuffer: arguments,
                                               routedBlobs: routed,
                                               routedOffsets: offsets,
                                               acts: moeActs,
                                               routingWeights: routingWeights,
                                               residual: sharedOut,
                                               y: ffnOut,
                                               d: D, f: width, topK: topK)
    }

    private func gemv(_ command: MTLCommandBuffer, _ view: TensorView,
                      x: MTLBuffer, y: MTLBuffer, yOffset: Int = 0,
                      m: Int, n: Int) {
        int4.encode(commandBuffer: command,
                    weights: view.buffer, weightsOffset: Int(view.offset),
                    scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                    biases: view.buffer, biasesOffset: Int(view.biasOffset),
                    x: x, y: y, yOffset: yOffset,
                    m: UInt32(m), n: UInt32(n))
    }

    private func resident(_ name: String) throws -> TensorView {
        try model.resident(name: name)
    }

    private func makeCommandBuffer(_ label: String) throws -> MTLCommandBuffer {
        guard let command = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        command.label = "flint pos=\(position) \(label)"
        return command
    }

    private func commitAndWait(_ command: MTLCommandBuffer) throws {
        command.commit()
        let waitStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        command.waitUntilCompleted()
        profile.gpuWaitNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - waitStart
        recordGPU(command)
        try checkCommandBufferError(command)
    }

    /// Starts reading `experts` into `layer`'s cache. Each layer has its own
    /// slots, and the GPU is done with this layer's slots from the last token.
    private func startPrefetch(layer: Int, experts: [Int]) throws {
        guard !experts.isEmpty, layer < cfg.numLayers, !cfg.layerIsDenseFFN(layer),
              prefetches[layer] == nil,
              let plan = try model.planRoutedExpertsIfPossible(
                layer: layer - cfg.numDenseLayers, experts: experts),
              !plan.misses.isEmpty
        else { return }
        let model = self.model
        prefetches[layer] = Task.detached(priority: .userInitiated) {
            _ = try? await model.fetchRoutedExperts(plan: plan)
        }
    }

    private func recordGPU(_ command: MTLCommandBuffer) {
        let nanos = UInt64(max(0, command.gpuEndTime - command.gpuStartTime) * 1e9)
        profile.gpuBusyNanos &+= nanos
        let stage = (command.label ?? "").split(separator: " ").last.map {
            String($0.prefix { $0 != "." && !$0.isNumber })
        } ?? ""
        profile.gpuByStage[stage, default: 0] &+= nanos
    }

    private func drainPendingCommands() throws {
        for pending in pendingCommands {
            pending.waitUntilCompleted()
            recordGPU(pending)
            try checkCommandBufferError(pending)
        }
        pendingCommands.removeAll(keepingCapacity: true)
    }

    /// Replaces +/-inf with the largest finite FP16 value in each buffer.
    private func encodeClamp(_ command: MTLCommandBuffer, _ buffers: [(MTLBuffer, Int)]) {
        guard let encoder = command.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(clampInf)
        let width = min(clampInf.maxTotalThreadsPerThreadgroup, 256)
        for (buffer, count) in buffers {
            var total = UInt32(count)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&total, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        }
        encoder.endEncoding()
    }
}

extension FlintForwardRunner: ChunkedPrefillRunner {
    /// Runs the prompt in chunks of `prefillChunkTokens`; only the last token
    /// of the last chunk runs the output head.
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        guard startPosition == position, !tokens.isEmpty else {
            throw PrefillError.prefillCursorMismatch(
                "Flint prefill expected position \(position), got \(startPosition)")
        }
        if tokens.count < batchedPrefillMinimumTokens {
            var done = 0
            for token in tokens {
                try Task.checkCancellation()
                done += 1
                try await step(token: token, into: logits, emitsLogits: done == tokens.count)
                onProgress(done)
            }
            return PrefillResult(newPosition: position, seed: .logitsWritten)
        }
        defer { prefillBuffers = nil }
        // A sliding layer's ring must still hold the window of every token in
        // a chunk after the whole chunk's keys are written.
        let chunk = min(prefillChunkTokens, maxRewindDistance)
        var start = tokens.startIndex
        while start < tokens.endIndex {
            try Task.checkCancellation()
            let end = min(start + chunk, tokens.endIndex)
            let before = start - tokens.startIndex
            try await prefillChunk(Array(tokens[start..<end]), into: logits,
                                   emitsLogits: end == tokens.endIndex) { onProgress(before + $0) }
            onProgress(end - tokens.startIndex)
            start = end
        }
        return PrefillResult(newPosition: position, seed: .logitsWritten)
    }
}

/// Prompt prefill: one chunk of tokens at a time, layer by layer. Projections
/// are matrix multiplies over the chunk; attention, norms and router scores
/// run per token with the decode arithmetic. Routed experts run as grouped
/// tiles through the expert cache, so each expert the chunk needs is read once
/// and its weights serve every token routed to it. Every per-token result
/// depends only on that token's inputs, never on how the prompt was chunked.
extension FlintForwardRunner {
    struct PrefillBuffers {
        let capacity: Int
        let tokens: MTLBuffer
        let hidden: MTLBuffer
        let normed: MTLBuffer
        let query: MTLBuffer
        let key: MTLBuffer
        let value: MTLBuffer
        let attn: MTLBuffer
        let projected: MTLBuffer
        let shared: MTLBuffer
        let routed: MTLBuffer
        let gate: MTLBuffer
        let up: MTLBuffer
        let act: MTLBuffer
        let scores: MTLBuffer
        let weights: MTLBuffer
        let partials: MTLBuffer
        let gateUpAct: MTLBuffer
        let down: MTLBuffer
    }

    /// The grouped expert kernel runs pairs 32 at a time over shared scratch.
    private static let pairMicrobatchRows = 32

    private func prefillBuffers(for count: Int) throws -> PrefillBuffers {
        if let existing = prefillBuffers, existing.capacity >= count { return existing }
        let n = count
        let D = cfg.hiddenSize
        let qDim = cfg.numHeads * cfg.headDim
        let kvDim = cfg.numKVHeads * cfg.headDim
        let wide = max(cfg.intermediateSize, cfg.sharedExpertIntermediateSize)
        let topK = cfg.topKExperts
        let device = ctx.device
        func buf(_ elements: Int, _ stride: Int, _ label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(elements, 1) * stride,
                                                 options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = "flint.prefill.\(label)"
            return buffer
        }
        let h = MemoryLayout<Float16>.size
        let buffers = PrefillBuffers(
            capacity: n,
            tokens: try buf(n, MemoryLayout<UInt32>.size, "tokens"),
            hidden: try buf(n * D, h, "hidden"),
            normed: try buf(n * D, h, "normed"),
            query: try buf(n * qDim, h, "query"),
            key: try buf(n * kvDim, h, "key"),
            value: try buf(n * kvDim, h, "value"),
            attn: try buf(n * qDim, h, "attn"),
            projected: try buf(n * D, h, "projected"),
            shared: try buf(n * D, h, "shared"),
            routed: try buf(n * D, h, "routed"),
            gate: try buf(n * wide, h, "gate"),
            up: try buf(n * wide, h, "up"),
            act: try buf(n * wide, h, "act"),
            scores: try buf(n * cfg.numExperts, MemoryLayout<Float>.size, "scores"),
            weights: try buf(n * topK, h, "weights"),
            partials: try buf(n * topK * D, h, "partials"),
            gateUpAct: try buf(3 * Self.pairMicrobatchRows * cfg.moeIntermediateSize, h, "gateUpAct"),
            down: try buf(Self.pairMicrobatchRows * D, h, "down"))
        prefillBuffers = buffers
        return buffers
    }

    /// Feeds `tokens` at `position`. `emitsLogits` runs the output head for
    /// the last token. `progress` hears an estimate of the tokens done after
    /// each routed layer, for a prompt-reading indicator.
    func prefillChunk(_ tokens: [Int32], into logits: MTLBuffer, emitsLogits: Bool,
                      progress: (Int) -> Void = { _ in }) async throws {
        let count = tokens.count
        guard count > 0, position + count <= maxContext else {
            throw GeneratorError.contextOverflow(prompt: position, maxNew: count,
                                                 maxContext: maxContext)
        }
        for token in tokens where token < 0 || Int(token) >= cfg.vocabSize {
            throw ModelError.indexCorrupt(detail: "token \(token) outside the vocabulary")
        }
        try? drainPendingCommands()
        speculated.removeAll(keepingCapacity: true)
        let buffers = try prefillBuffers(for: count)
        let D = cfg.hiddenSize
        let tokenSlots = buffers.tokens.contents().assumingMemoryBound(to: UInt32.self)
        for (index, token) in tokens.enumerated() { tokenSlots[index] = UInt32(token) }

        var command = try makeCommandBuffer("prefill-embed")
        try encodeEmbeddingBlock(command, buffers: buffers, count: count)
        let firstNorm = try resident("model.layers.0.input_layernorm.weight")
        prefillRMS.encodeBF16W(commandBuffer: command, x: buffers.hidden,
                               weight: firstNorm.buffer, weightOffset: Int(firstNorm.offset),
                               out: buffers.normed, t: UInt32(count), d: UInt32(D), eps: eps)

        for layer in 0..<cfg.numLayers {
            try encodeAttentionBlock(command, layer: layer, buffers: buffers, count: count)
            try encodeResidualNormBlock(command, buffers: buffers, delta: buffers.projected,
                                        delta2: nil,
                                        weight: "model.layers.\(layer).post_attention_layernorm.weight",
                                        count: count)
            let nextNorm = layer + 1 < cfg.numLayers
                ? "model.layers.\(layer + 1).input_layernorm.weight" : "model.norm.weight"
            if cfg.layerIsDenseFFN(layer) {
                try encodeDenseFFNBlock(command, prefix: "model.layers.\(layer).mlp",
                                        width: cfg.intermediateSize, buffers: buffers,
                                        destination: buffers.projected, count: count)
                try encodeResidualNormBlock(command, buffers: buffers, delta: buffers.projected,
                                            delta2: nil, weight: nextNorm, count: count)
                continue
            }
            try encodeDenseFFNBlock(command, prefix: "model.layers.\(layer).mlp.shared_expert",
                                    width: cfg.sharedExpertIntermediateSize, buffers: buffers,
                                    destination: buffers.shared, count: count)
            encodeClamp(command, [(buffers.shared, count * D)])
            try encodeRouterBlock(command, layer: layer, buffers: buffers, count: count)
            // Router scores for the whole chunk: the layer's only wait.
            try commitAndWait(command)
            try drainPendingCommands()
            if let pending = prefetches.removeValue(forKey: layer) {
                await pending.value
            }
            progress(count * (layer + 1) / cfg.numLayers)
            let routeStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let routes = try prefillRoutes(layer: layer, buffers: buffers, count: count)
            profile.routerNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - routeStart
            try await encodeRoutedTiles(layer: layer, routes: routes, buffers: buffers)

            command = try makeCommandBuffer("prefill-layer.\(layer + 1)")
            moeReduce.encodeReduceTokenMajor(commandBuffer: command,
                                             routePartials: buffers.partials,
                                             routeWeights: buffers.weights,
                                             h2: buffers.routed,
                                             queryCount: UInt32(count),
                                             topK: UInt32(cfg.topKExperts),
                                             d: UInt32(D))
            try encodeResidualNormBlock(command, buffers: buffers, delta: buffers.shared,
                                        delta2: buffers.routed, weight: nextNorm, count: count)
        }

        if emitsLogits, let blit = command.makeBlitCommandEncoder() {
            let row = D * MemoryLayout<Float16>.size
            blit.copy(from: buffers.normed, sourceOffset: (count - 1) * row,
                      to: normed, destinationOffset: 0, size: row)
            blit.endEncoding()
            let head = try resident("lm_head.weight")
            gemv(command, head, x: normed, y: logits, m: cfg.vocabSize, n: D)
        }
        try commitAndWait(command)
        try drainPendingCommands()
        position += count
    }

    private func encodeEmbeddingBlock(_ command: MTLCommandBuffer, buffers: PrefillBuffers,
                                      count: Int) throws {
        let table = try resident("model.embed_tokens.weight")
        guard let encoder = command.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(embedBlock)
        encoder.setBuffer(table.buffer, offset: Int(table.offset), index: 0)
        encoder.setBuffer(table.buffer, offset: Int(table.scaleOffset), index: 1)
        encoder.setBuffer(table.buffer, offset: Int(table.biasOffset), index: 2)
        encoder.setBuffer(buffers.hidden, offset: 0, index: 3)
        encoder.setBuffer(buffers.tokens, offset: 0, index: 4)
        var dim = UInt32(cfg.hiddenSize)
        encoder.setBytes(&dim, length: 4, index: 5)
        encoder.dispatchThreads(MTLSize(width: cfg.hiddenSize, height: count, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeAttentionBlock(_ command: MTLCommandBuffer, layer: Int,
                                      buffers: PrefillBuffers, count: Int) throws {
        guard let key = kCache[layer], let value = vCache[layer] else {
            throw ModelError.indexCorrupt(detail: "layer \(layer) has no KV cache")
        }
        let D = cfg.hiddenSize
        let qDim = cfg.numHeads * cfg.headDim
        let kvDim = cfg.numKVHeads * cfg.headDim
        let prefix = "model.layers.\(layer).self_attn."
        qmm(command, try resident(prefix + "q_proj.weight"), x: buffers.normed, y: buffers.query,
            rows: qDim, cols: D, count: count)
        qmm(command, try resident(prefix + "k_proj.weight"), x: buffers.normed, y: buffers.key,
            rows: kvDim, cols: D, count: count)
        qmm(command, try resident(prefix + "v_proj.weight"), x: buffers.normed, y: buffers.value,
            rows: kvDim, cols: D, count: count)
        let qNorm = try resident(prefix + "q_norm.weight")
        let kNorm = try resident(prefix + "k_norm.weight")
        if let encoder = command.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(qkNormRopeBlock)
            encoder.setBuffer(buffers.query, offset: 0, index: 0)
            encoder.setBuffer(buffers.key, offset: 0, index: 1)
            encoder.setBuffer(qNorm.buffer, offset: Int(qNorm.offset), index: 2)
            encoder.setBuffer(kNorm.buffer, offset: Int(kNorm.offset), index: 3)
            var queryHeads = UInt32(cfg.numHeads)
            var start = UInt32(position)
            var theta = Float(cfg.ropeTheta)
            var rotate: UInt32 = cfg.layerIsFull(layer) ? 0 : 1
            var epsilon = eps
            var qStride = UInt32(qDim)
            var kStride = UInt32(kvDim)
            encoder.setBytes(&queryHeads, length: 4, index: 4)
            encoder.setBytes(&start, length: 4, index: 5)
            encoder.setBytes(&theta, length: 4, index: 6)
            encoder.setBytes(&rotate, length: 4, index: 7)
            encoder.setBytes(&epsilon, length: 4, index: 8)
            encoder.setBytes(&qStride, length: 4, index: 9)
            encoder.setBytes(&kStride, length: 4, index: 10)
            encoder.dispatchThreadgroups(
                MTLSize(width: cfg.numHeads + cfg.numKVHeads, height: count, depth: 1),
                threadsPerThreadgroup: MTLSize(width: cfg.headDim / 2, height: 1, depth: 1))
            encoder.endEncoding()
        }
        let ring = cfg.layerIsFull(layer) ? 0 : slidingRingCapacity
        encodeKVCopy(command, from: buffers.key, to: key, count: count, ring: ring)
        encodeKVCopy(command, from: buffers.value, to: value, count: count, ring: ring)
        let sinks = try resident(prefix + "softmax_offset")
        if let encoder = command.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(attentionPrefill)
            encoder.setBuffer(buffers.query, offset: 0, index: 0)
            encoder.setBuffer(key, offset: 0, index: 1)
            encoder.setBuffer(value, offset: 0, index: 2)
            encoder.setBuffer(sinks.buffer, offset: Int(sinks.offset), index: 3)
            encoder.setBuffer(buffers.attn, offset: 0, index: 4)
            var kvHeads = UInt32(cfg.numKVHeads)
            var group = UInt32(queriesPerKV)
            var start = UInt32(position)
            var window = UInt32(cfg.layerIsFull(layer) ? 0 : cfg.slidingWindow)
            var scale = Float(cfg.attentionScale)
            var ringSlots = UInt32(ring)
            encoder.setBytes(&kvHeads, length: 4, index: 5)
            encoder.setBytes(&group, length: 4, index: 6)
            encoder.setBytes(&start, length: 4, index: 7)
            encoder.setBytes(&window, length: 4, index: 8)
            encoder.setBytes(&scale, length: 4, index: 9)
            encoder.setBytes(&ringSlots, length: 4, index: 10)
            encoder.dispatchThreadgroups(
                MTLSize(width: cfg.numKVHeads, height: count, depth: 1),
                threadsPerThreadgroup: MTLSize(width: queriesPerKV * 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
        qmm(command, try resident(prefix + "o_proj.weight"), x: buffers.attn, y: buffers.projected,
            rows: D, cols: qDim, count: count)
    }

    /// Copies `count` K or V rows for positions `position...` into the cache,
    /// in up to two runs when a ring wraps.
    private func encodeKVCopy(_ command: MTLCommandBuffer, from source: MTLBuffer,
                              to cache: MTLBuffer, count: Int, ring: Int) {
        let row = cfg.numKVHeads * cfg.headDim * MemoryLayout<Float16>.size
        guard let blit = command.makeBlitCommandEncoder() else { return }
        var copied = 0
        while copied < count {
            let slot = ring > 0 ? (position + copied) % ring : position + copied
            let run = ring > 0 ? min(count - copied, ring - slot) : count - copied
            blit.copy(from: source, sourceOffset: copied * row,
                      to: cache, destinationOffset: slot * row, size: run * row)
            copied += run
        }
        blit.endEncoding()
    }

    private func encodeDenseFFNBlock(_ command: MTLCommandBuffer, prefix: String, width: Int,
                                     buffers: PrefillBuffers, destination: MTLBuffer,
                                     count: Int) throws {
        let D = cfg.hiddenSize
        qmm(command, try resident("\(prefix).gate_proj.weight"), x: buffers.normed,
            y: buffers.gate, rows: width, cols: D, count: count)
        qmm(command, try resident("\(prefix).up_proj.weight"), x: buffers.normed,
            y: buffers.up, rows: width, cols: D, count: count)
        if let encoder = command.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(siluMul)
            encoder.setBuffer(buffers.gate, offset: 0, index: 0)
            encoder.setBuffer(buffers.up, offset: 0, index: 1)
            encoder.setBuffer(buffers.act, offset: 0, index: 2)
            var total = UInt32(count * width)
            encoder.setBytes(&total, length: 4, index: 3)
            encoder.dispatchThreads(MTLSize(width: count * width, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
        }
        qmm(command, try resident("\(prefix).down_proj.weight"), x: buffers.act,
            y: destination, rows: D, cols: width, count: count)
    }

    private func encodeRouterBlock(_ command: MTLCommandBuffer, layer: Int,
                                   buffers: PrefillBuffers, count: Int) throws {
        let gate = try resident("model.layers.\(layer).mlp.gate.weight")
        guard let encoder = command.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(routerGemvBlock)
        encoder.setBuffer(gate.buffer, offset: Int(gate.offset), index: 0)
        encoder.setBuffer(buffers.normed, offset: 0, index: 1)
        encoder.setBuffer(buffers.scores, offset: 0, index: 2)
        var experts = UInt32(cfg.numExperts)
        var hiddenDim = UInt32(cfg.hiddenSize)
        encoder.setBytes(&experts, length: 4, index: 3)
        encoder.setBytes(&hiddenDim, length: 4, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: (cfg.numExperts + 7) / 8, height: count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeResidualNormBlock(_ command: MTLCommandBuffer, buffers: PrefillBuffers,
                                         delta: MTLBuffer, delta2: MTLBuffer?,
                                         weight name: String, count: Int) throws {
        let weight = try resident(name)
        guard let encoder = command.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(residualNormBlock)
        encoder.setBuffer(buffers.hidden, offset: 0, index: 0)
        encoder.setBuffer(delta, offset: 0, index: 1)
        encoder.setBuffer(weight.buffer, offset: Int(weight.offset), index: 2)
        encoder.setBuffer(buffers.normed, offset: 0, index: 3)
        var dim = UInt32(cfg.hiddenSize)
        var epsilon = eps
        var hasDelta2: UInt32 = delta2 == nil ? 0 : 1
        encoder.setBytes(&dim, length: 4, index: 4)
        encoder.setBytes(&epsilon, length: 4, index: 5)
        encoder.setBuffer(delta2 ?? delta, offset: 0, index: 6)
        encoder.setBytes(&hasDelta2, length: 4, index: 7)
        encoder.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Top-k per token with the decode router, writing each token's weights,
    /// then (token, expert) pairs grouped into tiles. Experts already cached
    /// sort first, so their tiles need no reads.
    private func prefillRoutes(layer: Int, buffers: PrefillBuffers,
                               count: Int) throws -> PrefillMoEGroupedRoutes {
        let experts = cfg.numExperts, topK = cfg.topKExperts
        guard let bias = routerBias[layer] else {
            throw ModelError.indexCorrupt(detail: "layer \(layer) has no router")
        }
        let scores = buffers.scores.contents().assumingMemoryBound(to: Float.self)
        let weightsOut = buffers.weights.contents().assumingMemoryBound(to: Float16.self)
        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(count * topK)
        bias.withUnsafeBufferPointer { bias in
            routerIndices.withUnsafeMutableBufferPointer { indices in
                routerWeights.withUnsafeMutableBufferPointer { weights in
                    for t in 0..<count {
                        FlintRouter.select(scores: scores + t * experts, bias: bias.baseAddress!,
                                           experts: experts, topK: topK,
                                           indices: indices.baseAddress!,
                                           weights: weights.baseAddress!)
                        for rank in 0..<topK {
                            let weight = Float16(weights[rank])
                            weightsOut[t * topK + rank] = weight
                            pairs.append(PrefillTokenExpertPair(
                                token: UInt32(t), expert: UInt32(indices[rank]),
                                rank: UInt32(rank), weight: weight))
                        }
                    }
                }
            }
        }
        let packed = layer - cfg.numDenseLayers
        let resident = try model.routedExpertStreamer(layer: packed).residentExperts()
        let sortKeys = (0..<experts).map { UInt64(resident.contains($0) ? 0 : 1) }
        let slots = model.routedExpertCacheSlotCount(layer: packed) ?? 16
        return try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: count, topK: topK, numExperts: experts,
            tileExpertCount: min(16, max(1, slots / 2)), expertSortKeys: sortKeys)
    }

    /// Reads each tile's experts into the cache and runs its pairs. A tile's
    /// slots stay out of later plans until its command buffer finishes, so
    /// the next tile reads while this one computes.
    private func encodeRoutedTiles(layer: Int, routes: PrefillMoEGroupedRoutes,
                                   buffers: PrefillBuffers) async throws {
        let packed = layer - cfg.numDenseLayers
        let metadata = try groupedMoE.makeStreamedMetadataBuffers(device: ctx.device, routes: routes)
        let offsets = model.routedExpertOffsets(layer: packed)
        var inFlight: [(command: MTLCommandBuffer, slots: Set<Int>)] = []
        func busySlots() -> Set<Int> { inFlight.reduce(into: Set<Int>()) { $0.formUnion($1.slots) } }
        for tileIndex in routes.tiles.indices {
            let tile = routes.tiles[tileIndex]
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: tileIndex, routes: routes)
            let ioStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            var plan = try model.planRoutedExpertsIfPossible(layer: packed, experts: expertIDs,
                                                              avoidingSlots: busySlots())
            while plan == nil, !inFlight.isEmpty {
                try waitForTile(inFlight.removeFirst().command)
                plan = try model.planRoutedExpertsIfPossible(layer: packed, experts: expertIDs,
                                                              avoidingSlots: busySlots())
            }
            guard let plan else {
                throw ModelError.indexCorrupt(detail: "prefill tile of \(expertIDs.count) experts does not fit the cache")
            }
            profile.expertHits += plan.hits
            profile.expertMisses += plan.misses.count
            let views = plan.misses.isEmpty
                ? try model.routedExpertBuffers(for: plan)
                : try await model.fetchRoutedExperts(plan: plan)
            profile.expertIONanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - ioStart
            let binding = try PrefillStreamedTileBinding(expertIDs: expertIDs, views: views)
            let arguments = try groupedMoE.makeStreamedArgumentBuffer(device: ctx.device,
                                                                      binding: binding)
            let tileCommand = try makeCommandBuffer("prefill-moe.layer\(layer)")
            groupedMoE.encodeStreamedBatched(
                commandBuffer: tileCommand,
                hidden: buffers.normed,
                sortedPairs: metadata.sortedPairs,
                routePartials: buffers.partials,
                gateUpActScratch: buffers.gateUpAct,
                downScratch: buffers.down,
                argumentBuffer: arguments,
                binding: binding,
                params: PrefillGroupedRoutedMoEStreamedParams(
                    pairStart: tile.pairStart, pairCount: tile.pairCount,
                    d: UInt32(cfg.hiddenSize),
                    routedIntermediate: UInt32(cfg.moeIntermediateSize),
                    topK: UInt32(cfg.topKExperts),
                    hiddenStrideElements: UInt32(cfg.hiddenSize),
                    binding: binding, offsets: offsets),
                pairMicrobatchRows: Self.pairMicrobatchRows)
            tileCommand.commit()
            pendingCommands.append(tileCommand)
            inFlight.append((tileCommand, Set(plan.assignedSlots)))
        }
    }

    private func waitForTile(_ command: MTLCommandBuffer) throws {
        let waitStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        command.waitUntilCompleted()
        profile.gpuWaitNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - waitStart
        try checkCommandBufferError(command)
    }

    private func qmm(_ command: MTLCommandBuffer, _ view: TensorView, x: MTLBuffer, y: MTLBuffer,
                     rows: Int, cols: Int, count: Int) {
        if let prefillTensorQMM,
           prefillTensorQMM.encode(commandBuffer: command,
                                   weights: view.buffer, weightsOffset: Int(view.offset),
                                   scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                                   biases: view.buffer, biasesOffset: Int(view.biasOffset),
                                   x: x, y: y, m: count, n: rows, k: cols).path
            == .affineThreadgroupF16 {
            return
        }
        prefillQMM.encode(commandBuffer: command,
                          weights: view.buffer, weightsOffset: Int(view.offset),
                          scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                          biases: view.buffer, biasesOffset: Int(view.biasOffset),
                          x: x, y: y, t: count, n: rows, k: cols)
    }
}
