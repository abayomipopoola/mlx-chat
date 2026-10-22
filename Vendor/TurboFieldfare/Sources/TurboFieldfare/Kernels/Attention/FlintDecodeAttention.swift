import Metal

/// Flint decode attention with learned sinks: split-KV over position chunks,
/// then one merge pass per query head (`flint.metal`). Head dim is fixed at
/// 128 and each KV head serves at most 32 query heads.
public final class FlintDecodeAttention {
    public static let headDim = 128
    /// Positions per chunk before the chunk count hits `maxChunks`.
    static let targetChunkLength = 32
    static let maxChunks = 128

    private let partial: MTLComputePipelineState
    private let combine: MTLComputePipelineState
    private let maxima: MTLBuffer
    private let normalizers: MTLBuffer
    private let outputs: MTLBuffer
    private let numQueryHeads: Int

    public init(context: MetalContext, numQueryHeads: Int) throws {
        partial = try context.pipeline("flint_attention_partial", privateModule: "flint")
        combine = try context.pipeline("flint_attention_combine", privateModule: "flint")
        let slots = max(numQueryHeads, 1) * Self.maxChunks
        guard let maxima = context.device.makeBuffer(length: slots * 4, options: .storageModePrivate),
              let normalizers = context.device.makeBuffer(length: slots * 4,
                                                          options: .storageModePrivate),
              let outputs = context.device.makeBuffer(length: slots * Self.headDim * 4,
                                                      options: .storageModePrivate) else {
            throw ModelError.residentBufferWrapFailed
        }
        maxima.label = "flint.attention.max"
        normalizers.label = "flint.attention.sum"
        outputs.label = "flint.attention.partial"
        self.maxima = maxima
        self.normalizers = normalizers
        self.outputs = outputs
        self.numQueryHeads = numQueryHeads
    }

    static func chunkCount(length: Int) -> Int {
        min(max((length + targetChunkLength - 1) / targetChunkLength, 1), maxChunks)
    }

    /// Attends one query token to positions `[kvStart, seqLen)`. With a
    /// `ringCapacity`, position p lives in K/V slot `p % ringCapacity`.
    public func encode(commandBuffer: MTLCommandBuffer,
                       q: MTLBuffer, qOffset: Int = 0, k: MTLBuffer, v: MTLBuffer,
                       sinks: MTLBuffer, sinksOffset: Int,
                       out: MTLBuffer, outOffset: Int = 0,
                       numKVHeads: Int, seqLen: Int, kvStart: Int, scale: Float,
                       ringCapacity: Int = 0) {
        precondition(numQueryHeads % numKVHeads == 0 && numQueryHeads / numKVHeads <= 32)
        precondition(kvStart < seqLen)
        precondition(ringCapacity == 0 || seqLen - kvStart <= ringCapacity)
        let length = seqLen - kvStart
        let chunks = Self.chunkCount(length: length)
        var chunkLength = UInt32((length + chunks - 1) / chunks)
        var numChunks = UInt32(chunks)
        var kvHeads = UInt32(numKVHeads)
        var group = UInt32(numQueryHeads / numKVHeads)
        var seq = UInt32(seqLen)
        var start = UInt32(kvStart)
        var scaleValue = scale
        var ring = UInt32(ringCapacity)

        guard let first = commandBuffer.makeComputeCommandEncoder() else { return }
        first.setComputePipelineState(partial)
        for (index, buffer) in [q, k, v, maxima, normalizers, outputs].enumerated() {
            first.setBuffer(buffer, offset: index == 0 ? qOffset : 0, index: index)
        }
        first.setBytes(&kvHeads, length: 4, index: 6)
        first.setBytes(&group, length: 4, index: 7)
        first.setBytes(&seq, length: 4, index: 8)
        first.setBytes(&start, length: 4, index: 9)
        first.setBytes(&chunkLength, length: 4, index: 10)
        first.setBytes(&numChunks, length: 4, index: 11)
        first.setBytes(&scaleValue, length: 4, index: 12)
        first.setBytes(&ring, length: 4, index: 13)
        first.dispatchThreadgroups(MTLSize(width: numKVHeads * chunks, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: Int(group) * 32,
                                                                  height: 1, depth: 1))
        first.endEncoding()

        guard let second = commandBuffer.makeComputeCommandEncoder() else { return }
        second.setComputePipelineState(combine)
        second.setBuffer(maxima, offset: 0, index: 0)
        second.setBuffer(normalizers, offset: 0, index: 1)
        second.setBuffer(outputs, offset: 0, index: 2)
        second.setBuffer(sinks, offset: sinksOffset, index: 3)
        second.setBuffer(out, offset: outOffset, index: 4)
        second.setBytes(&numChunks, length: 4, index: 5)
        second.dispatchThreadgroups(MTLSize(width: numQueryHeads, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: Self.headDim,
                                                                   height: 1, depth: 1))
        second.endEncoding()
    }
}
