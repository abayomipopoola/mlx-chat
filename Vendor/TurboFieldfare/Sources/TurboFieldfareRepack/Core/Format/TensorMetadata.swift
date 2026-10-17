import Foundation

/// One physical tensor in a safetensors shard. Coordinates are absolute file
/// offsets so the writer can map and copy a single tensor without re-parsing
/// the shard header.
public struct SourceTensor: Sendable, Hashable {
    public enum Dtype: UInt8, Sendable, Hashable {
        case u32  = 0
        case bf16 = 1
        case fp16 = 2
        case fp32 = 3

        var elementBytes: Int {
            switch self { case .u32: 4; case .bf16: 2; case .fp16: 2; case .fp32: 4 }
        }
    }

    public let name: String
    public let shardPath: String
    public let dtype: Dtype
    public let shape: [UInt64]
    public let absoluteOffset: UInt64
    public let sizeBytes: UInt64

    public init(name: String, shardPath: String, dtype: Dtype,
                shape: [UInt64], absoluteOffset: UInt64, sizeBytes: UInt64) {
        self.name = name
        self.shardPath = shardPath
        self.dtype = dtype
        self.shape = shape
        self.absoluteOffset = absoluteOffset
        self.sizeBytes = sizeBytes
    }
}

/// Bit-width override resolved from `config.json -> quantization`.
public struct QuantSpec: Sendable, Hashable {
    public let bits: Int

    public init(bits: Int) {
        self.bits = bits
    }
}
