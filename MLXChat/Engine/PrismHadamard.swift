import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM

// MARK: - Rotation

/// A signed, blockwise Walsh–Hadamard rotation `R = H·S / √n` of an array's last axis.
/// Runs in float32 and returns the input's dtype, like PrismML's reference runtime.
struct SignedBlockHadamard: Sendable, Equatable {
    let blockSize: Int
    private let scale: Float

    init(blockSize: Int) {
        precondition(Self.isValidBlockSize(blockSize), "blockSize must be a power of two")
        self.blockSize = blockSize
        self.scale = 1 / Float(blockSize).squareRoot()
    }

    static func isValidBlockSize(_ blockSize: Int) -> Bool {
        blockSize > 0 && blockSize & (blockSize - 1) == 0
    }

    /// `R·x`: flip signs, then transform each block.
    func forward(_ x: MLXArray, signs: MLXArray) -> MLXArray {
        transform(x.asType(.float32) * signs).asType(x.dtype)
    }

    /// `Rᵀ·x`: transform each block, then flip signs.
    func inverse(_ x: MLXArray, signs: MLXArray) -> MLXArray {
        (transform(x.asType(.float32)) * signs).asType(x.dtype)
    }

    private func transform(_ x: MLXArray) -> MLXArray {
        precondition(x.dim(-1) % blockSize == 0, "last axis must be a multiple of blockSize")
        let shape = x.shape
        return hadamardTransform(x.reshaped(-1, blockSize), scale: scale).reshaped(shape)
    }
}

// MARK: - Layers

/// A `QuantizedLinear` whose weights were quantized in a rotated input basis: the input is
/// rotated before the packed matmul. `signs` loads from the checkpoint like any parameter.
final class HadamardQuantizedLinear: QuantizedLinear {
    let rotation: SignedBlockHadamard
    let signs: MLXArray

    init(
        weight: MLXArray, bias: MLXArray? = nil, scales: MLXArray, biases: MLXArray?,
        signs: MLXArray, groupSize: Int, bits: Int, mode: QuantizationMode = .affine,
        rotation: SignedBlockHadamard
    ) {
        self.rotation = rotation
        self.signs = signs
        super.init(
            weight: weight, bias: bias, scales: scales, biases: biases, groupSize: groupSize,
            bits: bits, mode: mode)
        freeze()
    }

    /// A placeholder shaped like the checkpoint's arrays, which the loader fills.
    convenience init(
        replacing linear: Linear, groupSize: Int, bits: Int, mode: QuantizationMode,
        rotation: SignedBlockHadamard
    ) {
        let (outputs, inputs) = linear.shape
        let groups = inputs / groupSize
        self.init(
            weight: MLXArray.zeros([outputs, inputs * bits / 32], dtype: .uint32),
            bias: linear.bias,
            scales: MLXArray.zeros([outputs, groups], dtype: .float16),
            biases: mode == .affine ? MLXArray.zeros([outputs, groups], dtype: .float16) : nil,
            signs: MLXArray.ones([inputs], dtype: .float32),
            groupSize: groupSize, bits: bits, mode: mode, rotation: rotation)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(rotation.forward(x, signs: signs))
    }
}

/// A `QuantizedEmbedding` whose rows were quantized in a rotated basis. Lookup un-rotates the
/// dequantized rows; `asLinear` rotates its input so a tied head shares the table.
final class HadamardQuantizedEmbedding: QuantizedEmbedding {
    let signs: MLXArray
    let rotation: SignedBlockHadamard

    /// A placeholder shaped like the checkpoint's arrays. The table quantizes lazily and is
    /// replaced by the loader before anything evaluates it.
    init(
        replacing embedding: Embedding, groupSize: Int, bits: Int, mode: QuantizationMode,
        rotation: SignedBlockHadamard
    ) {
        self.signs = MLXArray.ones([embedding.shape.1], dtype: .float32)
        self.rotation = rotation
        super.init(weight: embedding.weight, groupSize: groupSize, bits: bits, mode: mode)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        rotation.inverse(super.callAsFunction(x), signs: signs)
    }

    override func asLinear(_ x: MLXArray) -> MLXArray {
        super.asLinear(rotation.forward(x, signs: signs))
    }
}

// MARK: - Manifest

/// One rotated module named in a Prism pack's `config.json`.
struct HadamardQuantizedModule: Codable, Sendable, Equatable {
    var path: String
    var block: Int
    var embedding: Bool
    var dtype: String?

    init(path: String, block: Int, embedding: Bool = false, dtype: String? = nil) {
        self.path = path
        self.block = block
        self.embedding = embedding
        self.dtype = dtype
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        path = try container.decode(String.self, forKey: .path)
        block = try container.decode(Int.self, forKey: .block)
        embedding = try container.decodeIfPresent(Bool.self, forKey: .embedding) ?? false
        dtype = try container.decodeIfPresent(String.self, forKey: .dtype)
    }
}

/// The rotation manifest PrismML ships beside the base Qwen 3.5 configuration.
struct HadamardQuantizedManifest: Codable, Sendable {
    static let supportedSchemaVersion = 2

    var schemaVersion: Int
    var baseModelType: String?
    var tensorNamespace: String?
    var gdnActivationLayout: String?
    var modules: [HadamardQuantizedModule]
    var quantization: BaseConfiguration.Quantization

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case baseModelType = "base_model_type"
        case tensorNamespace = "tensor_namespace"
        case gdnActivationLayout = "gdn_activation_layout"
        case modules
        case quantization
    }

    init(
        schemaVersion: Int = HadamardQuantizedManifest.supportedSchemaVersion,
        baseModelType: String? = nil, tensorNamespace: String? = nil,
        gdnActivationLayout: String? = nil, modules: [HadamardQuantizedModule],
        quantization: BaseConfiguration.Quantization
    ) {
        self.schemaVersion = schemaVersion
        self.baseModelType = baseModelType
        self.tensorNamespace = tensorNamespace
        self.gdnActivationLayout = gdnActivationLayout
        self.modules = modules
        self.quantization = quantization
    }

    func validate(baseModelType expectedType: String, tensorNamespace expectedNamespace: String)
        throws
    {
        guard schemaVersion == Self.supportedSchemaVersion else {
            throw HadamardQuantizedCheckpointError.unsupportedSchemaVersion(schemaVersion)
        }
        guard baseModelType == expectedType else {
            throw HadamardQuantizedCheckpointError.unsupportedBaseModelType(
                baseModelType, expected: expectedType)
        }
        guard tensorNamespace == expectedNamespace else {
            throw HadamardQuantizedCheckpointError.unsupportedTensorNamespace(
                tensorNamespace, expected: expectedNamespace)
        }
        if let layout = gdnActivationLayout, layout != "grouped" {
            throw HadamardQuantizedCheckpointError.unsupportedActivationLayout(layout)
        }
        guard !modules.isEmpty else {
            throw HadamardQuantizedCheckpointError.emptyManifest
        }
        var paths = Set<String>()
        for module in modules where !paths.insert(module.path).inserted {
            throw HadamardQuantizedCheckpointError.duplicateModule(module.path)
        }
        _ = try activationDType()
    }

    /// The one half-precision dtype the declaring entries agree on, or `nil` if none declares one.
    func activationDType() throws -> DType? {
        let declared = Set(modules.compactMap(\.dtype))
        guard let name = declared.first else { return nil }
        guard declared.count == 1 else {
            throw HadamardQuantizedCheckpointError.mixedActivationDTypes(declared.sorted())
        }
        switch name {
        case "float16": return .float16
        case "bfloat16": return .bfloat16
        default: throw HadamardQuantizedCheckpointError.unsupportedActivationDType(name)
        }
    }
}

enum HadamardQuantizedCheckpointError: LocalizedError, Equatable {
    case unsupportedSchemaVersion(Int)
    case unsupportedBaseModelType(String?, expected: String)
    case unsupportedTensorNamespace(String?, expected: String)
    case unsupportedActivationLayout(String)
    case emptyManifest
    case duplicateModule(String)
    case mixedActivationDTypes([String])
    case unsupportedActivationDType(String)
    case moduleNotFound(String)
    case moduleNotSubstitutable(String, found: String)
    case unsupportedBlockSize(String, block: Int, width: Int)
    case invalidSigns(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedSchemaVersion(let version):
            "Rotated checkpoint schema_version \(version) is not supported "
                + "(expected \(HadamardQuantizedManifest.supportedSchemaVersion))."
        case .unsupportedBaseModelType(let type, let expected):
            "Rotated checkpoint base_model_type \(type ?? "nil") is not \(expected)."
        case .unsupportedTensorNamespace(let namespace, let expected):
            "Rotated checkpoint tensor_namespace \(namespace ?? "nil") is not \(expected)."
        case .unsupportedActivationLayout(let layout):
            "Rotated checkpoint gdn_activation_layout \(layout) needs a permutation "
                + "this loader does not apply."
        case .emptyManifest:
            "Rotated checkpoint declares no modules."
        case .duplicateModule(let path):
            "Rotated checkpoint lists module \(path) more than once."
        case .mixedActivationDTypes(let names):
            "Rotated checkpoint modules disagree on their activation dtype: "
                + names.joined(separator: ", ") + "."
        case .unsupportedActivationDType(let name):
            "Rotated checkpoint activation dtype \(name) is not supported "
                + "(expected float16 or bfloat16)."
        case .moduleNotFound(let path):
            "Rotated checkpoint names a module the model does not have: \(path)."
        case .moduleNotSubstitutable(let path, let found):
            "Rotated checkpoint module \(path) is a \(found), not a plain Linear or Embedding."
        case .unsupportedBlockSize(let path, let block, let width):
            "Rotated checkpoint module \(path) declares block \(block) for input width \(width); "
                + "a block must be a power of two that divides the width."
        case .invalidSigns(let path):
            "Rotated checkpoint module \(path) has signs other than -1 and +1."
        }
    }
}

// MARK: - Checkpoint policy

/// Applies a validated manifest to a model: rotated placeholders before the weights load, then
/// sign checks and the activation-dtype cast once they have.
struct HadamardQuantizedCheckpointPolicy: Sendable {
    let manifest: HadamardQuantizedManifest
    /// What the model prepends to the manifest's paths, such as `"language_model."`.
    let pathPrefix: String
    let activationDType: DType?

    init(
        manifest: HadamardQuantizedManifest, baseModelType: String, tensorNamespace: String,
        pathPrefix: String = ""
    ) throws {
        try manifest.validate(baseModelType: baseModelType, tensorNamespace: tensorNamespace)
        self.manifest = manifest
        self.pathPrefix = pathPrefix
        self.activationDType = try manifest.activationDType()
    }

    /// Replaces each manifest module with a rotated placeholder. The loader's quantize pass
    /// skips them because they already conform to `Quantized`.
    func substituteModules(in model: Module) throws {
        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let quantization = manifest.quantization
        var replacements = [(String, Module)]()
        for entry in manifest.modules {
            let path = pathPrefix + entry.path
            guard let module = modules[path] else {
                throw HadamardQuantizedCheckpointError.moduleNotFound(path)
            }
            let replacement: Module
            switch module {
            case let embedding as Embedding
            where entry.embedding && type(of: module) == Embedding.self:
                replacement = HadamardQuantizedEmbedding(
                    replacing: embedding, groupSize: quantization.groupSize,
                    bits: quantization.bits, mode: quantization.mode,
                    rotation: try Self.rotation(entry, at: path, width: embedding.shape.1))
            case let linear as Linear where !entry.embedding && type(of: module) == Linear.self:
                replacement = HadamardQuantizedLinear(
                    replacing: linear, groupSize: quantization.groupSize,
                    bits: quantization.bits, mode: quantization.mode,
                    rotation: try Self.rotation(entry, at: path, width: linear.shape.1))
            default:
                throw HadamardQuantizedCheckpointError.moduleNotSubstitutable(
                    path, found: "\(type(of: module))")
            }
            replacements.append((path, replacement))
        }
        model.update(modules: ModuleChildren.unflattened(replacements))
    }

    /// Checks the loaded sign vectors and casts unpacked floating-point tensors to the
    /// activation dtype. Packs store norms and small projections in float32 beside float16
    /// packed modules, which would otherwise promote the residual stream to float32. The
    /// packed modules' own tensors and `A_log`, which the recurrence reads in float32, keep
    /// their stored dtype.
    func prepareLoaded(_ model: Module) throws {
        let packed = Set(manifest.modules.map { pathPrefix + $0.path })
        let parameters = model.parameters().flattened()

        let signs = parameters.filter { key, _ in
            key.hasSuffix(".signs") && packed.contains(Self.modulePath(of: key))
        }
        if !signs.isEmpty {
            func isValid(_ values: MLXArray) -> MLXArray { (abs(values) .== 1).all() }
            if !stacked(signs.map { isValid($0.1) }).all().item(Bool.self),
               let invalid = signs.first(where: { !isValid($0.1).item(Bool.self) }) {
                throw HadamardQuantizedCheckpointError.invalidSigns(Self.modulePath(of: invalid.0))
            }
        }

        guard let activationDType else { return }
        let casts = parameters.compactMap { key, value -> (String, MLXArray)? in
            guard value.dtype.isFloatingPoint, value.dtype != activationDType,
                  !key.hasSuffix("A_log"), !packed.contains(Self.modulePath(of: key))
            else { return nil }
            return (key, value.asType(activationDType))
        }
        guard !casts.isEmpty else { return }
        model.update(parameters: ModuleParameters.unflattened(casts))
        eval(model)
    }

    private static func rotation(_ entry: HadamardQuantizedModule, at path: String, width: Int)
        throws -> SignedBlockHadamard
    {
        guard SignedBlockHadamard.isValidBlockSize(entry.block), width % entry.block == 0 else {
            throw HadamardQuantizedCheckpointError.unsupportedBlockSize(
                path, block: entry.block, width: width)
        }
        return SignedBlockHadamard(blockSize: entry.block)
    }

    private static func modulePath(of key: String) -> String {
        guard let dot = key.lastIndex(of: ".") else { return key }
        return String(key[..<dot])
    }
}

// MARK: - Registration

/// PrismML's Hadamard-rotated ternary Qwen 3.5 packs (Bonsai 2). The linked mlx-swift-lm has
/// no `prism_hadamard_qwen35` type and its Qwen 3.5 classes are not open, so the stock models
/// are built through this registration and their rotated modules are swapped in place.
enum PrismHadamardQwen35 {
    static let modelType = "prism_hadamard_qwen35"
    static let baseModelType = "qwen3_5"
    static let tensorNamespace = "mlx-vlm-qwen3_5"
    static let pathPrefix = "language_model."

    static func policy(configJSON data: Data) throws -> HadamardQuantizedCheckpointPolicy {
        let manifest = try JSONDecoder.json5().decode(HadamardQuantizedManifest.self, from: data)
        return try HadamardQuantizedCheckpointPolicy(
            manifest: manifest, baseModelType: baseModelType, tensorNamespace: tensorNamespace,
            pathPrefix: pathPrefix)
    }

    static func isPrismConfig(_ data: Data) -> Bool {
        guard let object = try? JSONSerialization.jsonObject(with: data),
              let dict = object as? [String: Any]
        else { return false }
        return dict["model_type"] as? String == modelType
    }

    /// Adds the type to both factories. Idempotent; call before any load.
    static func register() async {
        await LLMTypeRegistry.shared.registerModelType(modelType) { data in
            let policy = try policy(configJSON: data)
            let configuration = try JSONDecoder.json5().decode(
                MLXLLM.Qwen35Configuration.self, from: data)
            let model = Qwen35Model(configuration)
            try policy.substituteModules(in: model)
            return model
        }
        await VLMTypeRegistry.shared.registerModelType(modelType) { data in
            let policy = try policy(configJSON: data)
            let configuration = try JSONDecoder.json5().decode(
                MLXVLM.Qwen35Configuration.self, from: data)
            let model = Qwen35(configuration)
            try policy.substituteModules(in: model)
            return model
        }
    }
}
