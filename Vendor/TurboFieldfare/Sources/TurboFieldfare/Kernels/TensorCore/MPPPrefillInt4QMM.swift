import Foundation
import Metal

public final class MPPPrefillInt4QMM {
    public enum Variant: String, Sendable {
        case control
        case apple10V1 = "apple10-v1"
        case apple10BF16 = "apple10-bf16"
    }

    public enum Path: String, Sendable {
        case affineThreadgroupF16 = "affine-threadgroup-f16"
        case unavailable
        case fallback
    }

    public struct PathMetadata: Sendable, Equatable {
        public let path: Path
    }

    static let tileM = 64
    static let tileN = 32
    static let tileK = Quantization.groupSize

    private var pipeline: MTLComputePipelineState?
    public let unavailableReason: String?

    init(context: MetalContext, variant: Variant = .control) {
        _ = variant
        do {
            let library = try Self.compileTensorOpsLibrary(device: context.device)
            guard let function = library.makeFunction(
                name: "mpp_prefill_affine_threadgroup_f16") else {
                throw MetalError.missingFunction("mpp_prefill_affine_threadgroup_f16")
            }
            self.pipeline = try context.device.makeComputePipelineState(function: function)
            self.unavailableReason = nil
        } catch {
            self.pipeline = nil
            self.unavailableReason = String(describing: error)
        }
    }

    var isAvailable: Bool {
        pipeline != nil
    }

    @discardableResult
    func encode(commandBuffer: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       m: Int,
                       n: Int,
                       k: Int) -> PathMetadata {
        guard m > 0,
              n > 0,
              k > 0,
              k.isMultiple(of: Self.tileK),
              weightsOffset >= 0,
              scalesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              biasesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              xOffset.isMultiple(of: MemoryLayout<Float16>.stride),
              yOffset.isMultiple(of: MemoryLayout<Float16>.stride),
              let pipeline,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return PathMetadata(path: .fallback)
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = UInt32(m)
        var nValue = UInt32(n)
        var kValue = UInt32(k)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + Self.tileN - 1) / Self.tileN,
                    height: (m + Self.tileM - 1) / Self.tileM,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth * 4,
                                           height: 1,
                                           depth: 1))
        encoder.endEncoding()
        return PathMetadata(path: .affineThreadgroupF16)
    }

    private static func compileTensorOpsLibrary(device: MTLDevice) throws -> MTLLibrary {
        try MetalContext.moduleLibrary(device: device, module: "tensorops")
    }
}
