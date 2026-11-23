import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM

// mlx-swift-lm 3.33.0 ships `SignedBlockHadamard`, `HadamardQuantizedLinear`,
// `HadamardQuantizedEmbedding`, `HadamardQuantizedManifest`,
// `HadamardQuantizedCheckpointError`, `HadamardQuantizedCheckpointPolicy`,
// `MLXLLM.PrismHadamardQwen35Model`, and `MLXVLM.PrismHadamardQwen35`, and
// registers `prism_hadamard_qwen35` in both registries by default.
//
// These thin extensions keep the app's pre-3.33 helpers and test entry points
// callable with zero symbol collisions against the upstream types.

extension MLXVLM.PrismHadamardQwen35 {
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

    /// Ensures `prism_hadamard_qwen35` is registered in both factories. Idempotent.
    static func register() async {
        await LLMTypeRegistry.shared.registerModelType(modelType) { data in
            let decoder = JSONDecoder.json5()
            let manifest = try decoder.decode(HadamardQuantizedManifest.self, from: data)
            let configuration = try decoder.decode(MLXLLM.Qwen35Configuration.self, from: data)
            return try PrismHadamardQwen35Model(configuration, manifest: manifest)
        }
        await VLMTypeRegistry.shared.registerModelType(modelType) { data in
            let decoder = JSONDecoder.json5()
            let manifest = try decoder.decode(HadamardQuantizedManifest.self, from: data)
            let configuration = try decoder.decode(MLXVLM.Qwen35Configuration.self, from: data)
            return try PrismHadamardQwen35(configuration, manifest: manifest)
        }
    }
}

extension HadamardQuantizedCheckpointPolicy {
    /// Post-load compatibility helper used by tests and legacy call sites.
    func prepareLoaded(_ model: Module) throws {
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let prepared = try prepare(ModelCheckpoint(weights: weights))
        model.update(parameters: ModuleParameters.unflattened(prepared.weights))
        eval(model)
    }
}
