import Foundation
import Metal
import MLX
import Testing
import TurboFieldfare

/// Contract: the LFM2 SSD kernels reproduce MLX's own 6-bit affine math, and
/// the short convolution matches the reference recurrence.
@Suite(.serialized) struct LFM2KernelTests {
    private static let dim = 2_048
    private static let rows = 24

    private static func values(_ count: Int, phase: Float) -> [Float] {
        (0..<count).map { sin(Float($0) * 0.0137 + phase) * 0.8 }
    }

    private static func buffer<T>(_ device: MTLDevice, _ values: [T]) throws -> MTLBuffer {
        try #require(values.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 4),
                              options: .storageModeShared)
        })
    }

    private static func run(_ context: MetalContext,
                            _ encode: (MTLComputeCommandEncoder) -> Void) throws {
        let cb = try #require(context.queue.makeCommandBuffer())
        let enc = try #require(cb.makeComputeCommandEncoder())
        encode(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)
    }

    /// 6-bit weights, BF16 scales and biases, exactly as an MLX checkpoint stores them.
    private static func quantized6() -> (w: [UInt32], s: [UInt16], b: [UInt16], dense: [Float]) {
        let weight = MLXArray(values(rows * dim, phase: 0.3), [rows, dim]).asType(.bfloat16)
        let (wq, scales, biases) = MLX.quantized(weight, groupSize: 64, bits: 6)
        let dense = MLX.dequantized(wq, scales: scales, biases: biases!, groupSize: 64, bits: 6)
            .asType(.float32)
        return (wq.asArray(UInt32.self),
                scales.view(dtype: .uint16).asArray(UInt16.self),
                biases!.view(dtype: .uint16).asArray(UInt16.self),
                dense.asArray(Float.self))
    }

    @Test func q6GEMVMatchesMLXDequantizedMatmul() throws {
        let context = try MetalContext()
        let pso = try context.pipeline("lfm2_q6_gemv", privateModule: "lfm2")
        let (w, s, b, dense) = Self.quantized6()
        let x = Self.values(Self.dim, phase: 1.1).map { Float16($0) }
        let device = context.device
        let y = try #require(device.makeBuffer(length: Self.rows * 2, options: .storageModeShared))
        let wBuf = try Self.buffer(device, w)
        let sBuf = try Self.buffer(device, s)
        let bBuf = try Self.buffer(device, b)
        let xBuf = try Self.buffer(device, x)
        try Self.run(context) { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(wBuf, offset: 0, index: 0)
            enc.setBuffer(sBuf, offset: 0, index: 1)
            enc.setBuffer(bBuf, offset: 0, index: 2)
            enc.setBuffer(xBuf, offset: 0, index: 3)
            enc.setBuffer(y, offset: 0, index: 4)
            var rows = UInt32(Self.rows), dim = UInt32(Self.dim)
            enc.setBytes(&rows, length: 4, index: 5)
            enc.setBytes(&dim, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: (Self.rows + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let out = y.contents().assumingMemoryBound(to: Float16.self)
        for r in 0..<Self.rows {
            var expected: Float = 0
            for d in 0..<Self.dim { expected += dense[r * Self.dim + d] * Float(x[d]) }
            #expect(abs(Float(out[r]) - expected) <= 0.02 * max(1, abs(expected)),
                    "row \(r): \(out[r]) vs \(expected)")
        }
    }

    @Test func q6EmbeddingLookupMatchesMLXDequantizedRow() throws {
        let context = try MetalContext()
        let pso = try context.pipeline("lfm2_embed_lookup_q6", privateModule: "lfm2")
        let (w, s, b, dense) = Self.quantized6()
        let device = context.device
        let out = try #require(device.makeBuffer(length: Self.dim * 2, options: .storageModeShared))
        let wBuf = try Self.buffer(device, w)
        let sBuf = try Self.buffer(device, s)
        let bBuf = try Self.buffer(device, b)
        let token = 17
        try Self.run(context) { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(wBuf, offset: 0, index: 0)
            enc.setBuffer(sBuf, offset: 0, index: 1)
            enc.setBuffer(bBuf, offset: 0, index: 2)
            enc.setBuffer(out, offset: 0, index: 3)
            var t = UInt32(token), dim = UInt32(Self.dim)
            enc.setBytes(&t, length: 4, index: 4)
            enc.setBytes(&dim, length: 4, index: 5)
            enc.dispatchThreads(MTLSize(width: Self.dim, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let row = out.contents().assumingMemoryBound(to: Float16.self)
        for d in 0..<Self.dim {
            #expect(abs(Float(row[d]) - dense[token * Self.dim + d]) <= 2e-3)
        }
    }

    @Test func shortConvDecodeMatchesCausalReference() throws {
        let context = try MetalContext()
        let pso = try context.pipeline("lfm2_short_conv_decode", privateModule: "lfm2")
        let device = context.device
        let dim = 64
        let taps = 3
        let weights = (0..<(dim * taps)).map { Float(($0 % 7) - 3) * 0.25 }
        let weightBits = weights.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        let wBuf = try Self.buffer(device, weightBits)
        let state = try #require(device.makeBuffer(length: (taps - 1) * dim * 2,
                                                   options: .storageModeShared))
        memset(state.contents(), 0, state.length)
        let y = try #require(device.makeBuffer(length: dim * 2, options: .storageModeShared))
        var history: [[Float]] = []
        for step in 0..<4 {
            let bcx = (0..<(3 * dim)).map { Float16(sin(Float($0 + step * 31) * 0.21)) }
            let bcxBuf = try Self.buffer(device, bcx)
            try Self.run(context) { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(bcxBuf, offset: 0, index: 0)
                enc.setBuffer(wBuf, offset: 0, index: 1)
                enc.setBuffer(state, offset: 0, index: 2)
                enc.setBuffer(y, offset: 0, index: 3)
                var d = UInt32(dim), k = UInt32(taps)
                enc.setBytes(&d, length: 4, index: 4)
                enc.setBytes(&k, length: 4, index: 5)
                enc.dispatchThreads(MTLSize(width: dim, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            let u = (0..<dim).map { Float(bcx[$0]) * Float(bcx[2 * dim + $0]) }
            history.append(u)
            let out = y.contents().assumingMemoryBound(to: Float16.self)
            for c in 0..<dim {
                var conv: Float = 0
                for k in 0..<taps {
                    let t = history.count - taps + k
                    let value = t >= 0 ? Float(Float16(history[t][c])) : 0
                    conv += weights[c * taps + k] * value
                }
                let expected = Float(bcx[dim + c]) * conv
                #expect(abs(Float(out[c]) - expected) <= 2e-3, "step \(step) channel \(c)")
            }
        }
    }

    @Test func groupedFullAttentionKernelCompiles() throws {
        let context = try MetalContext()
        _ = try context.pipeline("attention_decode_full_grouped_vec_partial")
    }
}
