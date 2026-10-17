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
/// state is restored from a snapshot taken every `snapshotInterval` positions,
/// then the tokens up to the target are replayed.
public final class LFM2MoEForwardRunner: ContinuableLogitProducer, ContextWindowReporting,
                                         @unchecked Sendable {
    public let maxContext: Int
    public private(set) var position: Int = 0
    /// Rewinding to an earlier position is supported; see `prepareForContinuation`.
    public let supportsRewind = true
    static let snapshotInterval = 64

    /// Token fed at each position so far.
    private var fedTokens: [Int32] = []
    /// Tokens to replay before the next `produce`, after a rewind.
    private var pendingReplay: ArraySlice<Int32> = []
    /// Conv state before position `k * snapshotInterval`, all layers packed.
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
    private let routingWeights: MTLBuffer
    private let denseGate: MTLBuffer
    private let denseUp: MTLBuffer
    private let denseAct: MTLBuffer
    private let kCache: [Int: MTLBuffer]
    private let vCache: [Int: MTLBuffer]
    private let convState: [Int: MTLBuffer]

    private let eps: Float = 1e-5
    private static let streamedSlots = MoE.maxStreamedExperts

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
        let anchor = (expectedPosition / Self.snapshotInterval) * Self.snapshotInterval
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
            try await step(token: replay, into: logits)
        }
        guard requested == position else {
            throw PrefillError.prefillCursorMismatch(
                "LFM2 decode expected position \(position), got \(requested)")
        }
        try await step(token: token, into: logits)
    }

    private func step(token: Int32, into logits: MTLBuffer) async throws {
        if position % Self.snapshotInterval == 0, position > 0 {
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
                let experts = try route(layer: L)
                cb = try makeCommandBuffer("moe.layer\(L)")
                try await encodeRoutedExperts(cb, layer: L, experts: experts)
            }
            elementwise.encodeResidualAdd(commandBuffer: cb, hidden: hidden,
                                          delta: ffnOut, count: cfg.hiddenSize)
        }

        let finalNorm = try lfmResident("model.embedding_norm.weight")
        rms.encodeBF16W(commandBuffer: cb, x: hidden,
                        weight: finalNorm.buffer, weightOffset: Int(finalNorm.offset),
                        out: normed, d: D, eps: eps)
        encodeLMHead(cb, table: embedding, logits: logits)
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

    /// CPU router over the committed `normed` activations. Returns the top-k
    /// expert indices and writes their weights into `routingWeights`.
    private func route(layer L: Int) throws -> [Int] {
        let gate = try lfmResident("model.layers.\(L).feed_forward.gate.weight")
        let bias = try lfmResident("model.layers.\(L).feed_forward.expert_bias")
        let E = cfg.numExperts
        let D = cfg.hiddenSize
        let x = normed.contents().assumingMemoryBound(to: Float16.self)
        let w = gate.buffer.contents().advanced(by: Int(gate.offset))
            .assumingMemoryBound(to: Float.self)
        let b = bias.buffer.contents().advanced(by: Int(bias.offset))
            .assumingMemoryBound(to: Float.self)
        let xs = [Float](unsafeUninitializedCapacity: D) { out, count in
            for d in 0..<D { out[d] = Float(x[d]) }
            count = D
        }
        var scores = [Float](repeating: 0, count: E)
        vDSP_mmul(w, 1, xs, 1, &scores, 1, vDSP_Length(E), 1, vDSP_Length(D))
        for e in 0..<E { scores[e] = 1 / (1 + exp(-scores[e])) }
        let selected = (0..<E)
            .sorted { scores[$0] + b[$0] > scores[$1] + b[$1] }
            .prefix(cfg.topKExperts)
        let experts = Array(selected)
        let total = experts.reduce(Float(0)) { $0 + scores[$1] } + 1e-6
        let out = routingWeights.contents().assumingMemoryBound(to: Float16.self)
        for slot in 0..<Self.streamedSlots {
            out[slot] = slot < experts.count ? Float16(scores[experts[slot]] / total) : 0
        }
        return experts
    }

    private func encodeRoutedExperts(_ cb: MTLCommandBuffer, layer L: Int,
                                     experts: [Int]) async throws {
        let packed = L - cfg.numDenseLayers
        let blobs = try await model.fetchRoutedExperts(layer: packed, experts: experts)
        let unique = blobs.map(\.buffer)
        let routed = (0..<Self.streamedSlots).map { unique[$0 % unique.count] }
        let offsets = model.routedExpertOffsets(layer: packed)
        let topK = UInt32(Self.streamedSlots)
        let D = UInt32(cfg.hiddenSize)
        let F = UInt32(cfg.moeIntermediateSize)
        let argBuffer = moe.makeReusedRoutedArgumentBuffer(routedBlobs: routed, topK: topK)
        moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                routedArgBuffer: argBuffer,
                                                routedBlobs: routed,
                                                routedOffsets: offsets,
                                                x: normed,
                                                acts: moeActs,
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
