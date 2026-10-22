import Accelerate
import Foundation
import Metal

/// Decode runner for LFM2 MoE (`ModelFamily.lfm2moe`).
///
/// Per layer: `operator_norm` → gated short convolution or full attention →
/// residual, then `ffn_norm` → dense SwiGLU (the first `numDenseLayers`) or
/// routed experts → residual. `embedding_norm` precedes a 6-bit lm_head tied
/// to the embedding table.
///
/// The router is a float32 `[numExperts, hidden]` matrix applied on the CPU:
/// selection uses `sigmoid(logits) + expert_bias`, the weights are the
/// unbiased sigmoid scores renormalized over the top-k. The streamed-expert
/// kernels are fixed at eight slots, so the top-4 experts are dispatched
/// twice with the second copy weighted zero.
///
/// Prompts run token by token through `produce`; this runner does not
/// implement chunked prefill. It can rewind to any earlier position: the
/// attention KV past that point is simply overwritten, and the short-conv
/// state is restored from the nearest snapshot at or before the target, then
/// the tokens up to the target are replayed. Snapshots are dense near the
/// start, where conversations share the system prompt, and sparse after it.
public final class LFM2MoEForwardRunner: ContinuableLogitProducer, ContextWindowReporting,
                                         @unchecked Sendable {
    public let maxContext: Int
    public private(set) var position: Int = 0
    /// Rewinding to an earlier position is supported; see `prepareForContinuation`.
    public let supportsRewind = true
    static let snapshotInterval = 64
    /// Below this position snapshots come every `denseSnapshotInterval`.
    static let denseSnapshotLimit = 1_024
    static let denseSnapshotInterval = 8

    static func takesSnapshot(at position: Int) -> Bool {
        guard position > 0 else { return false }
        return position < denseSnapshotLimit
            ? position % denseSnapshotInterval == 0
            : position % snapshotInterval == 0
    }

    /// Token fed at each position so far.
    private var fedTokens: [Int32] = []
    /// Tokens to replay before the next `produce`, after a rewind.
    private var pendingReplay: ArraySlice<Int32> = []
    /// Conv state before each snapshot position, all layers packed.
    private var convSnapshots: [Int: Data] = [:]

    private let model: Model
    private let ctx: MetalContext
    private let cfg: ArchConfig

    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let rope: RoPE
    private let attention: Attention
    private let moe: MoE
    private let dense: SharedExpertInt4
    private let elementwise: Elementwise
    private let embedQ6: MTLComputePipelineState
    private let gemvQ6: MTLComputePipelineState
    private let shortConv: MTLComputePipelineState

    // Prompt prefill, layer by layer over a chunk of tokens.
    private let prefillQMM: PrefillInt4QMM
    private let prefillTensorQMM: MPPPrefillInt4QMM?
    private let prefillRMS: PrefillRMSNorm
    private let prefillHeadNorm: PrefillPerHeadNorm
    private let prefillRoPE: PrefillRoPE
    private let groupedMoE: PrefillGroupedRoutedMoE
    private let moeReduce: PrefillMoE
    private let embedBlock: MTLComputePipelineState
    private let convBlock: MTLComputePipelineState
    private let attentionPrefill: MTLComputePipelineState
    private let routerBlock: MTLComputePipelineState
    private let siluMul: MTLComputePipelineState
    /// Tokens per prefill chunk. A chunk waits on the GPU once per routed layer
    /// and reads each expert its tokens need once. Scratch takes about 60 KB
    /// per chunk token (plus a fixed 18 MB for the dense layers) and is freed
    /// after each prefill.
    public var prefillChunkTokens = 2_048 {
        didSet { prefillChunkTokens = min(max(prefillChunkTokens, 1), 2_048) }
    }
    /// False runs every projection on the FP32-weight matrix multiply.
    public var usesTensorQMM = true
    /// Projection kinds ("conv", "attention", "dense") kept on the FP32-weight
    /// multiply. The tensor-op multiply rounds dequantized weights to FP16,
    /// which LFM2.5's conv and dense layers do not tolerate: on a 1,283-token
    /// prompt the last-token logits correlated 0.96 with token-by-token
    /// prefill when every projection used it, 0.995 with these two exact.
    public var exactQMMKinds: Set<String> = ["conv", "dense"]
    /// Rows per pass through the two 11,776-wide dense layers.
    private static let denseRowBlock = 256

    private let hidden: MTLBuffer
    private let normed: MTLBuffer
    private let bcx: MTLBuffer
    private let convOut: MTLBuffer
    private let operatorOut: MTLBuffer
    private let query: MTLBuffer
    private let attnOut: MTLBuffer
    private let ffnOut: MTLBuffer
    private let zeroResidual: MTLBuffer
    private let moeActs: MTLBuffer
    /// Slots 0..<topK: phase 1 runs only the routed experts. The padding
    /// slots keep zero activations and zero weights.
    private let activeSlots: MTLBuffer
    private let activeSlotIndices: [UInt32]
    private let routingWeights: MTLBuffer
    private let denseGate: MTLBuffer
    private let denseUp: MTLBuffer
    private let denseAct: MTLBuffer
    private let kCache: [Int: MTLBuffer]
    private let vCache: [Int: MTLBuffer]
    private let convState: [Int: MTLBuffer]

    private let eps: Float = 1e-5
    private static let streamedSlots = MoE.maxStreamedExperts

    /// Reads layer L+1's likeliest experts while layer L runs, guessed by
    /// applying layer L+1's router to layer L's input. A wrong guess costs
    /// only I/O; the cache plan and the output are unchanged.
    public var prefetchesNextLayer = true
    /// How many of layer L+1's top-ranked experts to prefetch.
    public var prefetchWidth = 8 {
        didSet { prefetchWidth = min(max(prefetchWidth, 1), cfg.numExperts) }
    }
    private var prefetches: [Int: Task<Void, Never>] = [:]
    private var routerInput: [Float]
    private var routerScores: [Float]
    private var routerIndices: [Int]
    private var routerWeights: [Float]

    public init(model: Model, context: MetalContext, maxContext: Int) throws {
        let cfg = model.config
        guard cfg.family == .lfm2moe else {
            throw ModelError.indexCorrupt(detail: "LFM2MoEForwardRunner needs an lfm2moe model")
        }
        guard maxContext > 0, maxContext <= cfg.maxPositionEmbeddings else {
            throw ModelError.contextExceedsModel(requested: maxContext,
                                                 maximum: cfg.maxPositionEmbeddings)
        }
        precondition(cfg.topKExperts <= Self.streamedSlots)
        self.model = model
        self.ctx = context
        self.cfg = cfg
        self.routerInput = [Float](repeating: 0, count: cfg.hiddenSize)
        self.routerScores = [Float](repeating: 0, count: cfg.numExperts)
        self.routerIndices = [Int](repeating: 0, count: cfg.numExperts)
        self.routerWeights = [Float](repeating: 0, count: cfg.numExperts)
        self.maxContext = maxContext

        let D = cfg.hiddenSize
        let qDim = cfg.numHeads * cfg.headDim
        let kvDim = cfg.numKVHeads * cfg.headDim
        self.rms = try RMSNorm(context: context)
        self.int4 = try DequantInt4GEMV(context: context, additionalShapes: [
            (m: 3 * D, n: D), (m: D, n: D), (m: qDim, n: D), (m: kvDim, n: D),
            (m: D, n: qDim), (m: cfg.intermediateSize, n: D), (m: D, n: cfg.intermediateSize),
        ])
        self.rope = try RoPE(context: context)
        self.attention = try Attention(context: context)
        self.moe = try MoE(context: context,
                           siluActivation: true,
                           specializedD: UInt32(D),
                           specializedF: UInt32(cfg.moeIntermediateSize),
                           specializedNumExperts: UInt32(cfg.numExperts))
        self.dense = try SharedExpertInt4(context: context, siluActivation: true)
        self.elementwise = try Elementwise(context: context)
        self.embedQ6 = try context.pipeline("lfm2_embed_lookup_q6", privateModule: "lfm2")
        self.gemvQ6 = try context.pipeline("lfm2_q6_gemv", privateModule: "lfm2")
        self.shortConv = try context.pipeline("lfm2_short_conv_decode", privateModule: "lfm2")
        self.prefillQMM = try PrefillInt4QMM(context: context)
        let tensorQMM = MPPPrefillInt4QMM(context: context)
        self.prefillTensorQMM = tensorQMM.isAvailable ? tensorQMM : nil
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillHeadNorm = try PrefillPerHeadNorm(context: context)
        self.prefillRoPE = try PrefillRoPE(context: context)
        self.groupedMoE = try PrefillGroupedRoutedMoE(context: context, siluActivation: true)
        self.moeReduce = try PrefillMoE(context: context)
        self.embedBlock = try context.pipeline("lfm2_embed_lookup_q6_block", privateModule: "lfm2")
        self.convBlock = try context.pipeline("lfm2_short_conv_block", privateModule: "lfm2")
        self.attentionPrefill = try context.pipeline("lfm2_attention_prefill", privateModule: "lfm2")
        self.routerBlock = try context.pipeline("lfm2_router_f32_block", privateModule: "lfm2")
        self.siluMul = try context.pipeline("lfm2_silu_mul", privateModule: "lfm2")

        let device = context.device
        func buf(_ count: Int, _ label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * MemoryLayout<Float16>.size,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            b.label = "lfm2.\(label)"
            return b
        }
        self.hidden = try buf(D, "hidden")
        self.normed = try buf(D, "normed")
        self.bcx = try buf(3 * D, "bcx")
        self.convOut = try buf(D, "convOut")
        self.operatorOut = try buf(D, "operatorOut")
        self.query = try buf(qDim, "query")
        self.attnOut = try buf(qDim, "attnOut")
        self.ffnOut = try buf(D, "ffnOut")
        self.zeroResidual = try buf(D, "zeroResidual")
        memset(zeroResidual.contents(), 0, zeroResidual.length)
        self.moeActs = try buf(Self.streamedSlots * cfg.moeIntermediateSize, "moeActs")
        memset(moeActs.contents(), 0, moeActs.length)
        let slotIndices = (0..<UInt32(cfg.topKExperts)).map { $0 }
        self.activeSlotIndices = slotIndices
        guard let slots = slotIndices.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 4),
                              options: .storageModeShared)
        }) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.activeSlots = slots
        self.routingWeights = try buf(Self.streamedSlots, "routingWeights")
        self.denseGate = try buf(cfg.intermediateSize, "denseGate")
        self.denseUp = try buf(cfg.intermediateSize, "denseUp")
        self.denseAct = try buf(cfg.intermediateSize, "denseAct")

        var ks: [Int: MTLBuffer] = [:]
        var vs: [Int: MTLBuffer] = [:]
        var states: [Int: MTLBuffer] = [:]
        let taps = max(cfg.linearAttention.convKernelSize - 1, 1)
        for L in 0..<cfg.numLayers {
            if cfg.layerIsFull(L) {
                let bytes = maxContext * kvDim * MemoryLayout<Float16>.size
                guard let k = device.makeBuffer(length: bytes, options: .storageModeShared),
                      let v = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                    throw ModelError.kvAllocationFailed(layer: L, bytes: bytes)
                }
                k.label = "lfm2.kv.K.layer\(L)"
                v.label = "lfm2.kv.V.layer\(L)"
                ks[L] = k
                vs[L] = v
            } else if cfg.layerIsShortConv(L) {
                states[L] = try buf(taps * D, "conv.state.layer\(L)")
            }
        }
        self.kCache = ks
        self.vCache = vs
        self.convState = states
        resetConvState()
        // The router reads the FP32 gate and bias as raw memory on the CPU
        // and the GPU; refuse a model file whose tensors are shorter.
        for layer in 0..<cfg.numLayers where !cfg.layerIsDenseFFN(layer) {
            _ = try FlintForwardRunner.checkedTensor(
                model: model, name: "model.layers.\(layer).feed_forward.gate.weight",
                bytes: cfg.numExperts * cfg.hiddenSize * 4)
            _ = try FlintForwardRunner.checkedTensor(
                model: model, name: "model.layers.\(layer).feed_forward.expert_bias",
                bytes: cfg.numExperts * 4)
        }
    }

    // MARK: - LogitProducer

    public func reset() {
        position = 0
        fedTokens.removeAll(keepingCapacity: true)
        pendingReplay = []
        convSnapshots.removeAll()
        resetConvState()
    }

    public var continuationPosition: Int { position + pendingReplay.count }

    /// The cached key row for `position` in an attention layer, or the conv
    /// state of a short-conv layer when `position` is nil; for verification.
    public func debugCacheRow(layer: Int, position target: Int?) -> [Float16]? {
        if let target {
            guard let buffer = kCache[layer], target >= 0, target < position else { return nil }
            let kvDim = cfg.numKVHeads * cfg.headDim
            let row = buffer.contents().assumingMemoryBound(to: Float16.self) + target * kvDim
            return Array(UnsafeBufferPointer(start: row, count: kvDim))
        }
        guard let buffer = convState[layer] else { return nil }
        let p = buffer.contents().assumingMemoryBound(to: Float16.self)
        return Array(UnsafeBufferPointer(start: p, count: buffer.length / 2))
    }

    /// Continues at `expectedPosition`, which may be earlier than the current
    /// position. The rewind restores the nearest conv snapshot at or before
    /// the target; the next `produce` replays the tokens in between.
    public func prepareForContinuation(expectedPosition: Int) throws {
        let current = position + pendingReplay.count
        guard expectedPosition > 0, expectedPosition <= current else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected position \(expectedPosition), current \(current)")
        }
        guard expectedPosition < current else { return }
        let anchor = convSnapshots.keys.filter { $0 <= expectedPosition }.max() ?? 0
        if anchor == 0 {
            resetConvState()
        } else {
            guard let snapshot = convSnapshots[anchor] else {
                throw PrefillError.prefillCursorMismatch(
                    "no conv snapshot at position \(anchor)")
            }
            restoreConvState(snapshot)
        }
        let known = fedTokens + pendingReplay
        pendingReplay = known[anchor..<expectedPosition]
        fedTokens.removeSubrange(anchor...)
        convSnapshots = convSnapshots.filter { $0.key <= anchor }
        position = anchor
    }

    public func produce(token: Int32, position requested: Int, into logits: MTLBuffer) async throws {
        while let replay = pendingReplay.first {
            pendingReplay = pendingReplay.dropFirst()
            try await step(token: replay, into: logits, emitsLogits: false)
        }
        guard requested == position else {
            throw PrefillError.prefillCursorMismatch(
                "LFM2 decode expected position \(position), got \(requested)")
        }
        try await step(token: token, into: logits, emitsLogits: true)
    }

    /// `emitsLogits` false skips the final norm and output head: replayed and
    /// non-final prompt tokens need only their cache entries.
    private func step(token: Int32, into logits: MTLBuffer, emitsLogits: Bool) async throws {
        if Self.takesSnapshot(at: position) {
            convSnapshots[position] = snapshotConvState()
        }
        guard position < maxContext else {
            throw GeneratorError.contextOverflow(prompt: position, maxNew: 1,
                                                 maxContext: maxContext)
        }
        guard token >= 0, Int(token) < cfg.vocabSize else {
            throw ModelError.indexCorrupt(detail: "token \(token) outside the vocabulary")
        }
        let D = UInt32(cfg.hiddenSize)
        let embedding = try lfmResident("model.embed_tokens.weight")

        var cb = try makeCommandBuffer("embed")
        encodeEmbedding(cb, table: embedding, token: UInt32(token))

        for L in 0..<cfg.numLayers {
            let operatorNorm = try lfmResident("model.layers.\(L).operator_norm.weight")
            rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: operatorNorm.buffer, weightOffset: Int(operatorNorm.offset),
                            out: normed, d: D, eps: eps)
            if cfg.layerIsShortConv(L) {
                try encodeShortConv(cb, layer: L)
            } else {
                try encodeAttention(cb, layer: L)
            }
            elementwise.encodeResidualAdd(commandBuffer: cb, hidden: hidden,
                                          delta: operatorOut, count: cfg.hiddenSize)

            let ffnNorm = try lfmResident("model.layers.\(L).ffn_norm.weight")
            rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: ffnNorm.buffer, weightOffset: Int(ffnNorm.offset),
                            out: normed, d: D, eps: eps)
            if cfg.layerIsDenseFFN(L) {
                try encodeDenseFFN(cb, layer: L)
            } else {
                try commitAndWait(cb)
                let (experts, next) = try route(layer: L)
                try startPrefetch(layer: L + 1, experts: next)
                cb = try makeCommandBuffer("moe.layer\(L)")
                try await encodeRoutedExperts(cb, layer: L, experts: experts)
            }
            elementwise.encodeResidualAdd(commandBuffer: cb, hidden: hidden,
                                          delta: ffnOut, count: cfg.hiddenSize)
        }

        if emitsLogits {
            let finalNorm = try lfmResident("model.embedding_norm.weight")
            rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: finalNorm.buffer, weightOffset: Int(finalNorm.offset),
                            out: normed, d: D, eps: eps)
            encodeLMHead(cb, table: embedding, logits: logits)
        }
        try commitAndWait(cb)
        fedTokens.append(token)
        position += 1
    }

    // MARK: - Layer pieces

    private func encodeEmbedding(_ cb: MTLCommandBuffer, table: TensorView, token: UInt32) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(embedQ6)
        enc.setBuffer(table.buffer, offset: Int(table.offset), index: 0)
        enc.setBuffer(table.buffer, offset: Int(table.scaleOffset), index: 1)
        enc.setBuffer(table.buffer, offset: Int(table.biasOffset), index: 2)
        enc.setBuffer(hidden, offset: 0, index: 3)
        var tokenValue = token
        var dim = UInt32(cfg.hiddenSize)
        enc.setBytes(&tokenValue, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 5)
        let width = min(embedQ6.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: cfg.hiddenSize, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeLMHead(_ cb: MTLCommandBuffer, table: TensorView, logits: MTLBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(gemvQ6)
        enc.setBuffer(table.buffer, offset: Int(table.offset), index: 0)
        enc.setBuffer(table.buffer, offset: Int(table.scaleOffset), index: 1)
        enc.setBuffer(table.buffer, offset: Int(table.biasOffset), index: 2)
        enc.setBuffer(normed, offset: 0, index: 3)
        enc.setBuffer(logits, offset: 0, index: 4)
        var rows = UInt32(cfg.vocabSize)
        var dim = UInt32(cfg.hiddenSize)
        enc.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 6)
        let rowsPerGroup = 8
        enc.dispatchThreadgroups(
            MTLSize(width: (cfg.vocabSize + rowsPerGroup - 1) / rowsPerGroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * rowsPerGroup, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeShortConv(_ cb: MTLCommandBuffer, layer L: Int) throws {
        let D = cfg.hiddenSize
        let inProj = try lfmResident("model.layers.\(L).conv.in_proj.weight")
        let convWeight = try lfmResident("model.layers.\(L).conv.conv.weight")
        let outProj = try lfmResident("model.layers.\(L).conv.out_proj.weight")
        gemv(cb, inProj, x: normed, y: bcx, m: 3 * D, n: D)
        guard let state = convState[L], let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(shortConv)
        enc.setBuffer(bcx, offset: 0, index: 0)
        enc.setBuffer(convWeight.buffer, offset: Int(convWeight.offset), index: 1)
        enc.setBuffer(state, offset: 0, index: 2)
        enc.setBuffer(convOut, offset: 0, index: 3)
        var dim = UInt32(D)
        var kernelSize = UInt32(cfg.linearAttention.convKernelSize)
        enc.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&kernelSize, length: MemoryLayout<UInt32>.size, index: 5)
        let width = min(shortConv.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: D, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
        gemv(cb, outProj, x: convOut, y: operatorOut, m: D, n: D)
    }

    private func encodeAttention(_ cb: MTLCommandBuffer, layer L: Int) throws {
        guard let k = kCache[L], let v = vCache[L] else {
            throw ModelError.indexCorrupt(detail: "layer \(L) has no KV cache")
        }
        let D = cfg.hiddenSize
        let headDim = UInt32(cfg.headDim)
        let qDim = cfg.numHeads * cfg.headDim
        let kvDim = cfg.numKVHeads * cfg.headDim
        let slot = position * kvDim * MemoryLayout<Float16>.size
        let prefix = "model.layers.\(L).self_attn."
        let qProj = try lfmResident(prefix + "q_proj.weight")
        let kProj = try lfmResident(prefix + "k_proj.weight")
        let vProj = try lfmResident(prefix + "v_proj.weight")
        let outProj = try lfmResident(prefix + "out_proj.weight")
        let qNorm = try lfmResident(prefix + "q_layernorm.weight")
        let kNorm = try lfmResident(prefix + "k_layernorm.weight")

        gemv(cb, qProj, x: normed, y: query, m: qDim, n: D)
        gemv(cb, kProj, x: normed, y: k, yOffset: slot, m: kvDim, n: D)
        gemv(cb, vProj, x: normed, y: v, yOffset: slot, m: kvDim, n: D)
        rms.encodeBF16WPerHead(commandBuffer: cb, x: query,
                               weight: qNorm.buffer, weightOffset: Int(qNorm.offset),
                               out: query, headDim: headDim, numHeads: cfg.numHeads, eps: eps)
        rms.encodeBF16WPerHead(commandBuffer: cb, x: k, xOffset: slot,
                               weight: kNorm.buffer, weightOffset: Int(kNorm.offset),
                               out: k, outOffset: slot,
                               headDim: headDim, numHeads: cfg.numKVHeads, eps: eps)
        let theta = Float(cfg.fullRopeTheta)
        rope.encodeDefaultNeox(commandBuffer: cb, data: query, position: UInt32(position),
                               headDim: headDim, numHeads: UInt32(cfg.numHeads), theta: theta)
        rope.encodeDefaultNeox(commandBuffer: cb, data: k, dataOffset: slot,
                               position: UInt32(position),
                               headDim: headDim, numHeads: UInt32(cfg.numKVHeads), theta: theta)
        attention.encodeFull(commandBuffer: cb, q: query, k: k, v: v, out: attnOut,
                             headDim: headDim,
                             numQHeads: UInt32(cfg.numHeads),
                             numKVHeads: UInt32(cfg.numKVHeads),
                             seqLen: UInt32(position + 1),
                             scale: Float(cfg.attentionScale),
                             useGroupedVector: false)
        gemv(cb, outProj, x: attnOut, y: operatorOut, m: D, n: qDim)
    }

    private func encodeDenseFFN(_ cb: MTLCommandBuffer, layer L: Int) throws {
        let D = UInt32(cfg.hiddenSize)
        let F = UInt32(cfg.intermediateSize)
        func projection(_ name: String, rows: UInt32, cols: UInt32) throws -> SharedExpertProjection {
            let view = try lfmResident("model.layers.\(L).feed_forward.\(name).weight")
            return SharedExpertProjection(weights: view.buffer, scales: view.buffer,
                                          biases: view.buffer,
                                          weightsOffset: Int(view.offset),
                                          scalesOffset: Int(view.scaleOffset),
                                          biasesOffset: Int(view.biasOffset),
                                          rows: rows, cols: cols)
        }
        try dense.encode(commandBuffer: cb, x: normed,
                         gate: projection("gate_proj", rows: F, cols: D),
                         up: projection("up_proj", rows: F, cols: D),
                         down: projection("down_proj", rows: D, cols: F),
                         y: ffnOut,
                         scratchGate: denseGate, scratchUp: denseUp, scratchAct: denseAct)
    }

    /// CPU router over the committed `normed` activations. Writes the top-k
    /// weights into `routingWeights` and returns the experts, plus layer L+1's
    /// likeliest experts from the same input when prefetching.
    private func route(layer L: Int) throws -> (experts: [Int], next: [Int]) {
        let E = cfg.numExperts
        let D = cfg.hiddenSize
        let topK = cfg.topKExperts
        routerInput.withUnsafeMutableBufferPointer { input in
            var source = vImage_Buffer(data: normed.contents(), height: 1,
                                       width: vImagePixelCount(D), rowBytes: D * 2)
            var destination = vImage_Buffer(data: input.baseAddress, height: 1,
                                            width: vImagePixelCount(D), rowBytes: D * 4)
            vImageConvert_Planar16FtoPlanarF(&source, &destination, vImage_Flags(kvImageNoFlags))
        }
        func select(_ layer: Int, count: Int) throws {
            let gate = try lfmResident("model.layers.\(layer).feed_forward.gate.weight")
            let bias = try lfmResident("model.layers.\(layer).feed_forward.expert_bias")
            let w = gate.buffer.contents().advanced(by: Int(gate.offset))
                .assumingMemoryBound(to: Float.self)
            let b = bias.buffer.contents().advanced(by: Int(bias.offset))
                .assumingMemoryBound(to: Float.self)
            routerInput.withUnsafeBufferPointer { input in
                routerScores.withUnsafeMutableBufferPointer { scores in
                    routerIndices.withUnsafeMutableBufferPointer { indices in
                        routerWeights.withUnsafeMutableBufferPointer { weights in
                            let s = scores.baseAddress!
                            vDSP_mmul(w, 1, input.baseAddress!, 1, s, 1,
                                      vDSP_Length(E), 1, vDSP_Length(D))
                            var e = 0
                            while e < E { s[e] = 1 / (1 + exp(-s[e])); e &+= 1 }
                            FlintRouter.select(scores: s, bias: b, experts: E, topK: count,
                                               indices: indices.baseAddress!,
                                               weights: weights.baseAddress!, epsilon: 1e-6)
                        }
                    }
                }
            }
        }
        var next: [Int] = []
        let nextLayer = L + 1
        if prefetchesNextLayer, nextLayer < cfg.numLayers, !cfg.layerIsDenseFFN(nextLayer) {
            try select(nextLayer, count: prefetchWidth)
            next = Array(routerIndices.prefix(prefetchWidth))
        }
        try select(L, count: topK)
        let out = routingWeights.contents().assumingMemoryBound(to: Float16.self)
        for slot in 0..<Self.streamedSlots {
            out[slot] = slot < topK ? Float16(routerWeights[slot]) : 0
        }
        return (Array(routerIndices.prefix(topK)), next)
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

    private func encodeRoutedExperts(_ cb: MTLCommandBuffer, layer L: Int,
                                     experts: [Int]) async throws {
        let packed = L - cfg.numDenseLayers
        if let pending = prefetches.removeValue(forKey: L) {
            await pending.value
        }
        let plan = try model.planRoutedExperts(layer: packed, experts: experts)!
        let blobs = plan.misses.isEmpty
            ? try model.routedExpertBuffers(for: plan)
            : try await model.fetchRoutedExperts(plan: plan)
        let unique = blobs.map(\.buffer)
        let routed = (0..<Self.streamedSlots).map { unique[$0 % unique.count] }
        let offsets = model.routedExpertOffsets(layer: packed)
        let topK = UInt32(Self.streamedSlots)
        let D = UInt32(cfg.hiddenSize)
        let F = UInt32(cfg.moeIntermediateSize)
        let argBuffer = moe.makeReusedRoutedArgumentBuffer(routedBlobs: routed, topK: topK)
        moe.encodeRoutedPersistentPhase1SubsetU16Load(commandBuffer: cb,
                                                      routedArgBuffer: argBuffer,
                                                      routedBlobs: routed,
                                                      routedOffsets: offsets,
                                                      x: normed,
                                                      acts: moeActs,
                                                      activeSlots: activeSlots,
                                                      activeSlotIndices: activeSlotIndices,
                                                      activeCount: UInt32(cfg.topKExperts),
                                                      d: D, f: F, topK: topK)
        moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: cb,
                                               routedArgBuffer: argBuffer,
                                               routedBlobs: routed,
                                               routedOffsets: offsets,
                                               acts: moeActs,
                                               routingWeights: routingWeights,
                                               residual: zeroResidual,
                                               y: ffnOut,
                                               d: D, f: F, topK: topK)
    }

    // MARK: - Helpers

    private func gemv(_ cb: MTLCommandBuffer, _ view: TensorView,
                      x: MTLBuffer, y: MTLBuffer, yOffset: Int = 0,
                      m: Int, n: Int) {
        int4.encode(commandBuffer: cb,
                    weights: view.buffer, weightsOffset: Int(view.offset),
                    scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                    biases: view.buffer, biasesOffset: Int(view.biasOffset),
                    x: x, y: y, yOffset: yOffset,
                    m: UInt32(m), n: UInt32(n))
    }

    private func lfmResident(_ name: String) throws -> TensorView {
        try model.resident(name: name)
    }

    private func makeCommandBuffer(_ label: String) throws -> MTLCommandBuffer {
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        cb.label = "lfm2 pos=\(position) \(label)"
        return cb
    }

    private func commitAndWait(_ cb: MTLCommandBuffer) throws {
        cb.commit()
        cb.waitUntilCompleted()
        try checkCommandBufferError(cb)
    }

    private func resetConvState() {
        for buffer in convState.values {
            memset(buffer.contents(), 0, buffer.length)
        }
    }

    /// Called only between tokens, after the last command buffer completed.
    private func snapshotConvState() -> Data {
        var data = Data()
        for layer in convState.keys.sorted() {
            let buffer = convState[layer]!
            data.append(buffer.contents().assumingMemoryBound(to: UInt8.self),
                        count: buffer.length)
        }
        return data
    }

    private func restoreConvState(_ data: Data) {
        var offset = 0
        data.withUnsafeBytes { raw in
            for layer in convState.keys.sorted() {
                let buffer = convState[layer]!
                memcpy(buffer.contents(), raw.baseAddress!.advanced(by: offset), buffer.length)
                offset += buffer.length
            }
        }
    }
}

extension LFM2MoEForwardRunner: ChunkedPrefillRunner {
    /// Replays any rewound tokens with the prompt, in chunks of
    /// `prefillChunkTokens`; only the last token runs the output head.
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        guard startPosition == continuationPosition, !tokens.isEmpty else {
            throw PrefillError.prefillCursorMismatch(
                "LFM2 prefill expected position \(continuationPosition), got \(startPosition)")
        }
        let stream = Array(pendingReplay) + Array(tokens)
        let replayed = pendingReplay.count
        pendingReplay = []
        var start = 0
        while start < stream.count {
            try Task.checkCancellation()
            let end = min(start + prefillChunkTokens, stream.count)
            let before = start
            try await prefillChunk(Array(stream[start..<end]), into: logits,
                                   emitsLogits: end == stream.count) {
                if before + $0 > replayed { onProgress(before + $0 - replayed) }
            }
            if end > replayed { onProgress(end - replayed) }
            start = end
        }
        return PrefillResult(newPosition: position, seed: .logitsWritten)
    }
}

/// Prompt prefill: one chunk of tokens at a time, layer by layer. Projections
/// are matrix multiplies over the chunk, the short convolution runs the
/// chunk's tokens in order per channel (writing the same conv snapshots the
/// decode path takes), and routed experts run as grouped tiles through the
/// expert cache. A token's result does not depend on how the prompt was
/// chunked.
extension LFM2MoEForwardRunner {
    private struct PrefillBuffers {
        let tokens: MTLBuffer
        let hidden: MTLBuffer
        let normed: MTLBuffer
        let projected: MTLBuffer
        let bcx: MTLBuffer
        let conv: MTLBuffer
        let query: MTLBuffer
        let key: MTLBuffer
        let value: MTLBuffer
        let attn: MTLBuffer
        let gate: MTLBuffer
        let up: MTLBuffer
        let act: MTLBuffer
        let scores: MTLBuffer
        let weights: MTLBuffer
        let partials: MTLBuffer
        let gateUpAct: MTLBuffer
        let down: MTLBuffer
        let snapshotSlots: MTLBuffer
        let snapshots: MTLBuffer
    }

    private static let pairMicrobatchRows = 32

    private var convLayers: [Int] { convState.keys.sorted() }

    private func makePrefillBuffers(count: Int, snapshots: Int) throws -> PrefillBuffers {
        let D = cfg.hiddenSize
        let qDim = cfg.numHeads * cfg.headDim
        let kvDim = cfg.numKVHeads * cfg.headDim
        let taps = max(cfg.linearAttention.convKernelSize - 1, 1)
        let rowBlock = min(count, Self.denseRowBlock)
        let device = ctx.device
        func buf(_ elements: Int, _ stride: Int, _ label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(elements, 1) * stride,
                                                 options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = "lfm2.prefill.\(label)"
            return buffer
        }
        let h = MemoryLayout<Float16>.size
        return PrefillBuffers(
            tokens: try buf(count, MemoryLayout<UInt32>.size, "tokens"),
            hidden: try buf(count * D, h, "hidden"),
            normed: try buf(count * D, h, "normed"),
            projected: try buf(count * D, h, "projected"),
            bcx: try buf(count * 3 * D, h, "bcx"),
            conv: try buf(count * D, h, "conv"),
            query: try buf(count * qDim, h, "query"),
            key: try buf(count * kvDim, h, "key"),
            value: try buf(count * kvDim, h, "value"),
            attn: try buf(count * qDim, h, "attn"),
            gate: try buf(rowBlock * cfg.intermediateSize, h, "gate"),
            up: try buf(rowBlock * cfg.intermediateSize, h, "up"),
            act: try buf(rowBlock * cfg.intermediateSize, h, "act"),
            scores: try buf(count * cfg.numExperts, MemoryLayout<Float>.size, "scores"),
            weights: try buf(count * cfg.topKExperts, h, "weights"),
            partials: try buf(count * cfg.topKExperts * D, h, "partials"),
            gateUpAct: try buf(3 * Self.pairMicrobatchRows * cfg.moeIntermediateSize, h, "gateUpAct"),
            down: try buf(Self.pairMicrobatchRows * D, h, "down"),
            snapshotSlots: try buf(count, MemoryLayout<Int32>.size, "snapshotSlots"),
            snapshots: try buf(snapshots * convLayers.count * taps * D, h, "snapshots"))
    }

    /// Feeds `tokens` at `position`; `emitsLogits` runs the output head for
    /// the last one.
    private func prefillChunk(_ tokens: [Int32], into logits: MTLBuffer, emitsLogits: Bool,
                              progress: (Int) -> Void = { _ in }) async throws {
        let count = tokens.count
        guard count > 0, position + count <= maxContext else {
            throw GeneratorError.contextOverflow(prompt: position, maxNew: count,
                                                 maxContext: maxContext)
        }
        for token in tokens where token < 0 || Int(token) >= cfg.vocabSize {
            throw ModelError.indexCorrupt(detail: "token \(token) outside the vocabulary")
        }
        let D = cfg.hiddenSize
        let taps = max(cfg.linearAttention.convKernelSize - 1, 1)
        // Conv snapshots at the positions decode would take them.
        let snapshotPositions = (0..<count).map { position + $0 }.filter { Self.takesSnapshot(at: $0) }
        let buffers = try makePrefillBuffers(count: count, snapshots: snapshotPositions.count)
        let tokenSlots = buffers.tokens.contents().assumingMemoryBound(to: UInt32.self)
        for (index, token) in tokens.enumerated() { tokenSlots[index] = UInt32(token) }
        let slots = buffers.snapshotSlots.contents().assumingMemoryBound(to: Int32.self)
        for t in 0..<count { slots[t] = -1 }
        for (index, snapshotPosition) in snapshotPositions.enumerated() {
            slots[snapshotPosition - position] = Int32(index)
        }

        var tileCommands: [MTLCommandBuffer] = []
        var command = try makeCommandBuffer("prefill-embed")
        let embedding = try lfmResident("model.embed_tokens.weight")
        if let encoder = command.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(embedBlock)
            encoder.setBuffer(embedding.buffer, offset: Int(embedding.offset), index: 0)
            encoder.setBuffer(embedding.buffer, offset: Int(embedding.scaleOffset), index: 1)
            encoder.setBuffer(embedding.buffer, offset: Int(embedding.biasOffset), index: 2)
            encoder.setBuffer(buffers.hidden, offset: 0, index: 3)
            encoder.setBuffer(buffers.tokens, offset: 0, index: 4)
            var dim = UInt32(D)
            encoder.setBytes(&dim, length: 4, index: 5)
            encoder.dispatchThreads(MTLSize(width: D, height: count, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
        }

        for L in 0..<cfg.numLayers {
            let operatorNorm = try lfmResident("model.layers.\(L).operator_norm.weight")
            prefillRMS.encodeBF16W(commandBuffer: command, x: buffers.hidden,
                                   weight: operatorNorm.buffer, weightOffset: Int(operatorNorm.offset),
                                   out: buffers.normed, t: UInt32(count), d: UInt32(D), eps: eps)
            if cfg.layerIsShortConv(L) {
                try encodeShortConvBlock(command, layer: L, buffers: buffers, count: count)
            } else {
                try encodeAttentionBlock(command, layer: L, buffers: buffers, count: count)
            }
            elementwise.encodeResidualAdd(commandBuffer: command, hidden: buffers.hidden,
                                          delta: buffers.projected, count: count * D)
            let ffnNorm = try lfmResident("model.layers.\(L).ffn_norm.weight")
            prefillRMS.encodeBF16W(commandBuffer: command, x: buffers.hidden,
                                   weight: ffnNorm.buffer, weightOffset: Int(ffnNorm.offset),
                                   out: buffers.normed, t: UInt32(count), d: UInt32(D), eps: eps)
            if cfg.layerIsDenseFFN(L) {
                try encodeDenseFFNBlock(command, layer: L, buffers: buffers, count: count)
                elementwise.encodeResidualAdd(commandBuffer: command, hidden: buffers.hidden,
                                              delta: buffers.projected, count: count * D)
                continue
            }
            try encodeRouterBlock(command, layer: L, buffers: buffers, count: count)
            // Router scores for the whole chunk: the layer's only wait.
            try commitAndWait(command)
            progress(count * (L + 1) / cfg.numLayers)
            if let pending = prefetches.removeValue(forKey: L) {
                await pending.value
            }
            let routes = try prefillRoutes(layer: L, buffers: buffers, count: count)
            tileCommands += try await encodeRoutedTiles(layer: L, routes: routes, buffers: buffers)
            command = try makeCommandBuffer("prefill-layer.\(L + 1)")
            moeReduce.encodeReduceTokenMajor(commandBuffer: command,
                                             routePartials: buffers.partials,
                                             routeWeights: buffers.weights,
                                             h2: buffers.projected,
                                             queryCount: UInt32(count),
                                             topK: UInt32(cfg.topKExperts),
                                             d: UInt32(D))
            elementwise.encodeResidualAdd(commandBuffer: command, hidden: buffers.hidden,
                                          delta: buffers.projected, count: count * D)
        }

        if emitsLogits {
            if let blit = command.makeBlitCommandEncoder() {
                let row = D * MemoryLayout<Float16>.size
                blit.copy(from: buffers.hidden, sourceOffset: (count - 1) * row,
                          to: hidden, destinationOffset: 0, size: row)
                blit.endEncoding()
            }
            let finalNorm = try lfmResident("model.embedding_norm.weight")
            rms.encodeBF16W(commandBuffer: command, x: hidden,
                            weight: finalNorm.buffer, weightOffset: Int(finalNorm.offset),
                            out: normed, d: UInt32(D), eps: eps)
            encodeLMHead(command, table: embedding, logits: logits)
        }
        try commitAndWait(command)
        for tile in tileCommands { try checkCommandBufferError(tile) }

        let snapshotBytes = convLayers.count * taps * D * MemoryLayout<Float16>.size
        let base = buffers.snapshots.contents()
        for (index, snapshotPosition) in snapshotPositions.enumerated() {
            convSnapshots[snapshotPosition] = Data(bytes: base.advanced(by: index * snapshotBytes),
                                                   count: snapshotBytes)
        }
        fedTokens.append(contentsOf: tokens)
        position += count
    }

    private func encodeShortConvBlock(_ command: MTLCommandBuffer, layer L: Int,
                                      buffers: PrefillBuffers, count: Int) throws {
        let D = cfg.hiddenSize
        let taps = max(cfg.linearAttention.convKernelSize - 1, 1)
        qmm(command, try lfmResident("model.layers.\(L).conv.in_proj.weight"),
            x: buffers.normed, y: buffers.bcx, rows: 3 * D, cols: D, count: count, kind: "conv")
        let convWeight = try lfmResident("model.layers.\(L).conv.conv.weight")
        guard let state = convState[L], let order = convLayers.firstIndex(of: L),
              let encoder = command.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(convBlock)
        encoder.setBuffer(buffers.bcx, offset: 0, index: 0)
        encoder.setBuffer(convWeight.buffer, offset: Int(convWeight.offset), index: 1)
        encoder.setBuffer(state, offset: 0, index: 2)
        encoder.setBuffer(buffers.conv, offset: 0, index: 3)
        var dim = UInt32(D)
        var kernelSize = UInt32(cfg.linearAttention.convKernelSize)
        var rows = UInt32(count)
        var stride = UInt32(convLayers.count * taps * D)
        encoder.setBytes(&dim, length: 4, index: 4)
        encoder.setBytes(&kernelSize, length: 4, index: 5)
        encoder.setBytes(&rows, length: 4, index: 6)
        encoder.setBuffer(buffers.snapshotSlots, offset: 0, index: 7)
        encoder.setBuffer(buffers.snapshots, offset: order * taps * D * MemoryLayout<Float16>.size,
                          index: 8)
        encoder.setBytes(&stride, length: 4, index: 9)
        encoder.dispatchThreads(MTLSize(width: D, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        qmm(command, try lfmResident("model.layers.\(L).conv.out_proj.weight"),
            x: buffers.conv, y: buffers.projected, rows: D, cols: D, count: count, kind: "conv")
    }

    private func encodeAttentionBlock(_ command: MTLCommandBuffer, layer L: Int,
                                      buffers: PrefillBuffers, count: Int) throws {
        guard let key = kCache[L], let value = vCache[L] else {
            throw ModelError.indexCorrupt(detail: "layer \(L) has no KV cache")
        }
        let D = cfg.hiddenSize
        let qDim = cfg.numHeads * cfg.headDim
        let kvDim = cfg.numKVHeads * cfg.headDim
        let prefix = "model.layers.\(L).self_attn."
        qmm(command, try lfmResident(prefix + "q_proj.weight"), x: buffers.normed,
            y: buffers.query, rows: qDim, cols: D, count: count, kind: "attention")
        qmm(command, try lfmResident(prefix + "k_proj.weight"), x: buffers.normed,
            y: buffers.key, rows: kvDim, cols: D, count: count, kind: "attention")
        qmm(command, try lfmResident(prefix + "v_proj.weight"), x: buffers.normed,
            y: buffers.value, rows: kvDim, cols: D, count: count, kind: "attention")
        let qNorm = try lfmResident(prefix + "q_layernorm.weight")
        let kNorm = try lfmResident(prefix + "k_layernorm.weight")
        let headDim = UInt32(cfg.headDim)
        prefillHeadNorm.encodeBF16W(commandBuffer: command, x: buffers.query,
                                    weight: qNorm.buffer, weightOffset: Int(qNorm.offset),
                                    out: buffers.query, queryCount: UInt32(count),
                                    headDim: headDim, numHeads: UInt32(cfg.numHeads),
                                    tokenStrideElements: UInt32(qDim), eps: eps)
        prefillHeadNorm.encodeBF16W(commandBuffer: command, x: buffers.key,
                                    weight: kNorm.buffer, weightOffset: Int(kNorm.offset),
                                    out: buffers.key, queryCount: UInt32(count),
                                    headDim: headDim, numHeads: UInt32(cfg.numKVHeads),
                                    tokenStrideElements: UInt32(kvDim), eps: eps)
        let theta = Float(cfg.fullRopeTheta)
        prefillRoPE.encodeDefaultNeox(commandBuffer: command, data: buffers.query,
                                      startPosition: UInt32(position), queryCount: UInt32(count),
                                      headDim: headDim, numHeads: UInt32(cfg.numHeads),
                                      tokenStrideElements: UInt32(qDim), theta: theta)
        prefillRoPE.encodeDefaultNeox(commandBuffer: command, data: buffers.key,
                                      startPosition: UInt32(position), queryCount: UInt32(count),
                                      headDim: headDim, numHeads: UInt32(cfg.numKVHeads),
                                      tokenStrideElements: UInt32(kvDim), theta: theta)
        if let blit = command.makeBlitCommandEncoder() {
            let row = kvDim * MemoryLayout<Float16>.size
            blit.copy(from: buffers.key, sourceOffset: 0, to: key,
                      destinationOffset: position * row, size: count * row)
            blit.copy(from: buffers.value, sourceOffset: 0, to: value,
                      destinationOffset: position * row, size: count * row)
            blit.endEncoding()
        }
        if let encoder = command.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(attentionPrefill)
            encoder.setBuffer(buffers.query, offset: 0, index: 0)
            encoder.setBuffer(key, offset: 0, index: 1)
            encoder.setBuffer(value, offset: 0, index: 2)
            encoder.setBuffer(buffers.attn, offset: 0, index: 3)
            var kvHeads = UInt32(cfg.numKVHeads)
            var group = UInt32(cfg.numHeads / cfg.numKVHeads)
            var start = UInt32(position)
            var scale = Float(cfg.attentionScale)
            encoder.setBytes(&kvHeads, length: 4, index: 4)
            encoder.setBytes(&group, length: 4, index: 5)
            encoder.setBytes(&start, length: 4, index: 6)
            encoder.setBytes(&scale, length: 4, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: cfg.numKVHeads, height: count, depth: 1),
                threadsPerThreadgroup: MTLSize(width: Int(group) * 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
        qmm(command, try lfmResident(prefix + "out_proj.weight"), x: buffers.attn,
            y: buffers.projected, rows: D, cols: qDim, count: count, kind: "attention")
    }

    /// The 11,776-wide dense SwiGLU in row blocks, so its scratch stays fixed.
    private func encodeDenseFFNBlock(_ command: MTLCommandBuffer, layer L: Int,
                                     buffers: PrefillBuffers, count: Int) throws {
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let prefix = "model.layers.\(L).feed_forward."
        let gate = try lfmResident(prefix + "gate_proj.weight")
        let up = try lfmResident(prefix + "up_proj.weight")
        let down = try lfmResident(prefix + "down_proj.weight")
        let row = D * MemoryLayout<Float16>.size
        var start = 0
        while start < count {
            let rows = min(Self.denseRowBlock, count - start)
            qmm(command, gate, x: buffers.normed, xOffset: start * row, y: buffers.gate,
                rows: F, cols: D, count: rows, kind: "dense")
            qmm(command, up, x: buffers.normed, xOffset: start * row, y: buffers.up,
                rows: F, cols: D, count: rows, kind: "dense")
            if let encoder = command.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(siluMul)
                encoder.setBuffer(buffers.gate, offset: 0, index: 0)
                encoder.setBuffer(buffers.up, offset: 0, index: 1)
                encoder.setBuffer(buffers.act, offset: 0, index: 2)
                var total = UInt32(rows * F)
                encoder.setBytes(&total, length: 4, index: 3)
                encoder.dispatchThreads(MTLSize(width: rows * F, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
            }
            qmm(command, down, x: buffers.act, y: buffers.projected, yOffset: start * row,
                rows: D, cols: F, count: rows, kind: "dense")
            start += rows
        }
    }

    private func encodeRouterBlock(_ command: MTLCommandBuffer, layer L: Int,
                                   buffers: PrefillBuffers, count: Int) throws {
        let gate = try lfmResident("model.layers.\(L).feed_forward.gate.weight")
        guard let encoder = command.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(routerBlock)
        encoder.setBuffer(gate.buffer, offset: Int(gate.offset), index: 0)
        encoder.setBuffer(buffers.normed, offset: 0, index: 1)
        encoder.setBuffer(buffers.scores, offset: 0, index: 2)
        var experts = UInt32(cfg.numExperts)
        var dim = UInt32(cfg.hiddenSize)
        encoder.setBytes(&experts, length: 4, index: 3)
        encoder.setBytes(&dim, length: 4, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: (cfg.numExperts + 7) / 8, height: count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Top-k per token with the decode router's selection, then (token,
    /// expert) pairs grouped into tiles, cached experts first.
    private func prefillRoutes(layer L: Int, buffers: PrefillBuffers,
                               count: Int) throws -> PrefillMoEGroupedRoutes {
        let experts = cfg.numExperts, topK = cfg.topKExperts
        let bias = try lfmResident("model.layers.\(L).feed_forward.expert_bias")
        let b = bias.buffer.contents().advanced(by: Int(bias.offset))
            .assumingMemoryBound(to: Float.self)
        let scores = buffers.scores.contents().assumingMemoryBound(to: Float.self)
        let weightsOut = buffers.weights.contents().assumingMemoryBound(to: Float16.self)
        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(count * topK)
        routerIndices.withUnsafeMutableBufferPointer { indices in
            routerWeights.withUnsafeMutableBufferPointer { weights in
                for t in 0..<count {
                    FlintRouter.select(scores: scores + t * experts, bias: b, experts: experts,
                                       topK: topK, indices: indices.baseAddress!,
                                       weights: weights.baseAddress!, epsilon: 1e-6)
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
        let packed = L - cfg.numDenseLayers
        let resident = try model.routedExpertStreamer(layer: packed).residentExperts()
        let sortKeys = (0..<experts).map { UInt64(resident.contains($0) ? 0 : 1) }
        let slots = model.routedExpertCacheSlotCount(layer: packed) ?? 16
        return try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: count, topK: topK, numExperts: experts,
            tileExpertCount: min(16, max(1, slots / 2)), expertSortKeys: sortKeys)
    }

    /// Reads each tile's experts into the cache and runs its pairs; a tile's
    /// slots stay out of later plans until its command buffer finishes.
    private func encodeRoutedTiles(layer L: Int, routes: PrefillMoEGroupedRoutes,
                                   buffers: PrefillBuffers) async throws -> [MTLCommandBuffer] {
        let packed = L - cfg.numDenseLayers
        let metadata = try groupedMoE.makeStreamedMetadataBuffers(device: ctx.device, routes: routes)
        let offsets = model.routedExpertOffsets(layer: packed)
        var committed: [MTLCommandBuffer] = []
        var inFlight: [(command: MTLCommandBuffer, slots: Set<Int>)] = []
        func busySlots() -> Set<Int> { inFlight.reduce(into: Set<Int>()) { $0.formUnion($1.slots) } }
        for tileIndex in routes.tiles.indices {
            let tile = routes.tiles[tileIndex]
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: tileIndex, routes: routes)
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
            let views = plan.misses.isEmpty
                ? try model.routedExpertBuffers(for: plan)
                : try await model.fetchRoutedExperts(plan: plan)
            let binding = try PrefillStreamedTileBinding(expertIDs: expertIDs, views: views)
            let arguments = try groupedMoE.makeStreamedArgumentBuffer(device: ctx.device,
                                                                      binding: binding)
            let tileCommand = try makeCommandBuffer("prefill-moe.layer\(L)")
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
            committed.append(tileCommand)
            inFlight.append((tileCommand, Set(plan.assignedSlots)))
        }
        return committed
    }

    private func waitForTile(_ command: MTLCommandBuffer) throws {
        command.waitUntilCompleted()
        try checkCommandBufferError(command)
    }

    private func qmm(_ command: MTLCommandBuffer, _ view: TensorView,
                     x: MTLBuffer, xOffset: Int = 0, y: MTLBuffer, yOffset: Int = 0,
                     rows: Int, cols: Int, count: Int, kind: String = "") {
        if usesTensorQMM, !exactQMMKinds.contains(kind), let prefillTensorQMM,
           prefillTensorQMM.encode(commandBuffer: command,
                                   weights: view.buffer, weightsOffset: Int(view.offset),
                                   scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                                   biases: view.buffer, biasesOffset: Int(view.biasOffset),
                                   x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                                   m: count, n: rows, k: cols).path
            == .affineThreadgroupF16 {
            return
        }
        prefillQMM.encode(commandBuffer: command,
                          weights: view.buffer, weightsOffset: Int(view.offset),
                          scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                          biases: view.buffer, biasesOffset: Int(view.biasOffset),
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          t: count, n: rows, k: cols)
    }
}
