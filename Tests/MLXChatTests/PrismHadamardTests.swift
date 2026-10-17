import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing
@testable import MLXChat

/// Contract: Bonsai 2 (`prism_hadamard_qwen35`) packs load as the stock Qwen 3.5 models with
/// every manifest module rotated, and nothing outside the manifest touched.
@Suite(.serialized) struct PrismHadamardTests {

    // MARK: - Fixtures

    private static func values(_ shape: [Int], phase: Float = 0) -> MLXArray {
        let count = shape.reduce(1, *)
        return MLXArray((0 ..< count).map { sin(Float($0) * 0.37 + phase) }, shape)
    }

    private static func signs(_ n: Int) -> MLXArray {
        MLXArray((0 ..< n).map { ($0 * 7 + 3) % 5 < 2 ? Float(-1) : Float(1) })
    }

    /// Sylvester's Hadamard matrix of order `n`, built explicitly as the reference.
    private static func hadamardMatrix(_ n: Int) -> MLXArray {
        var rows: [[Float]] = [[1]]
        while rows.count < n {
            rows = rows.map { $0 + $0 } + rows.map { $0 + $0.map { -$0 } }
        }
        return MLXArray(rows.flatMap { $0 }, [n, n])
    }

    private static func isClose(_ a: MLXArray, _ b: MLXArray, tolerance: Double = 1e-3) -> Bool {
        a.shape == b.shape && a.allClose(b, rtol: tolerance, atol: tolerance).item(Bool.self)
    }

    private static let blockSize = 128

    private static func manifestModules() -> [[String: Any]] {
        var modules: [[String: Any]] = [
            ["path": "lm_head", "block": blockSize, "embedding": false, "dtype": "float16"],
            ["path": "model.embed_tokens", "block": blockSize, "embedding": true, "dtype": "float16"],
        ]
        for layer in 0 ..< 4 {
            let attention = layer == 3
                ? ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
                : ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"]
            for name in attention + ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
                modules.append([
                    "path": "model.layers.\(layer).\(name)", "block": blockSize,
                    "embedding": false, "dtype": "float16",
                ])
            }
        }
        return modules
    }

    /// A Bonsai-shaped `config.json` with tiny dimensions.
    private static func config(
        modules: [[String: Any]] = manifestModules(),
        overrides: [String: Any] = [:]
    ) throws -> Data {
        var config: [String: Any] = [
            "schema_version": 2,
            "model_type": "prism_hadamard_qwen35",
            "base_model_type": "qwen3_5",
            "tensor_namespace": "mlx-vlm-qwen3_5",
            "gdn_activation_layout": "grouped",
            "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
            "tie_word_embeddings": false,
            "modules": modules,
            "text_config": [
                "model_type": "qwen3_5_text",
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 4,
                "full_attention_interval": 4,
                "layer_types": [
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                ],
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "attn_output_gate": true,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 32,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_value_head_dim": 32,
                "max_position_embeddings": 262_144,
                "rms_norm_eps": 1e-6,
                "vocab_size": 512,
                "tie_word_embeddings": false,
                "rope_parameters": [
                    "mrope_interleaved": true,
                    "mrope_section": [3, 3, 2],
                    "rope_type": "default",
                    "rope_theta": 10_000_000,
                    "partial_rotary_factor": 0.25,
                ],
            ] as [String: Any],
            "vision_config": [
                "model_type": "qwen3_5",
                "depth": 1,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_heads": 2,
                "in_channels": 3,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "out_hidden_size": 256,
                "num_position_embeddings": 64,
                "deepstack_visual_indexes": [Int](),
                "hidden_act": "gelu_pytorch_tanh",
            ] as [String: Any],
        ]
        config.merge(overrides) { _, new in new }
        return try JSONSerialization.data(withJSONObject: config)
    }

    private static func leaves(_ model: Module) -> [String: Module] {
        Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
    }

    // MARK: - Rotation

    @Test func forwardMatchesExplicitSignedHadamard() {
        let block = 8
        let rotation = SignedBlockHadamard(blockSize: block)
        let x = Self.values([3, 16])
        let signs = Self.signs(16)
        let h = Self.hadamardMatrix(block) / Float(block).squareRoot()
        let expected = matmul((x * signs).reshaped(-1, block), h).reshaped(3, 16)

        #expect(Self.isClose(rotation.forward(x, signs: signs), expected))
    }

    @Test func inverseUndoesForwardAndKeepsDType() {
        let rotation = SignedBlockHadamard(blockSize: 32)
        let x = Self.values([2, 5, 64]).asType(.float16)
        let signs = Self.signs(64)
        let rotated = rotation.forward(x, signs: signs)

        #expect(rotated.dtype == .float16)
        #expect(Self.isClose(rotation.inverse(rotated, signs: signs), x, tolerance: 1e-2))
    }

    // MARK: - Layers

    @Test func linearRotatesTheInputBeforeThePackedMatmul() {
        let (inputs, outputs, groupSize, bits) = (64, 8, 32, 2)
        let rotation = SignedBlockHadamard(blockSize: 32)
        let (weight, scales, biases) = quantized(
            Self.values([outputs, inputs]), groupSize: groupSize, bits: bits)
        let signs = Self.signs(inputs)
        let layer = HadamardQuantizedLinear(
            weight: weight, scales: scales, biases: biases, signs: signs,
            groupSize: groupSize, bits: bits, rotation: rotation)
        let x = Self.values([2, inputs], phase: 1)

        let expected = quantizedMM(
            rotation.forward(x, signs: signs), weight, scales: scales, biases: biases,
            transpose: true, groupSize: groupSize, bits: bits)
        let unrotated = quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits)

        #expect(Self.isClose(layer(x), expected))
        #expect(!Self.isClose(layer(x), unrotated, tolerance: 1e-2))
    }

    @Test func embeddingUnrotatesRowsAndRotatesForTheHead() {
        let (count, dimensions, groupSize, bits) = (16, 64, 32, 2)
        let rotation = SignedBlockHadamard(blockSize: 32)
        let table = Self.values([count, dimensions])
        let (weight, scales, biases) = quantized(table, groupSize: groupSize, bits: bits)
        let signs = Self.signs(dimensions)
        let layer = HadamardQuantizedEmbedding(
            replacing: Embedding(weight: table), groupSize: groupSize, bits: bits,
            mode: .affine, rotation: rotation)
        layer.update(parameters: ModuleParameters.unflattened(["signs": signs]))

        let ids = MLXArray([3, 5, 1, 3], [2, 2])
        let rows = dequantized(
            weight[ids.flattened()], scales: scales[ids.flattened()],
            biases: biases.map { $0[ids.flattened()] }, groupSize: groupSize, bits: bits)
        let expectedRows = rotation.inverse(rows, signs: signs).reshaped(2, 2, dimensions)
        #expect(Self.isClose(layer(ids), expectedRows))

        let x = Self.values([3, dimensions], phase: 2)
        let expectedLogits = quantizedMM(
            rotation.forward(x, signs: signs), weight, scales: scales, biases: biases,
            transpose: true, groupSize: groupSize, bits: bits)
        #expect(Self.isClose(layer.asLinear(x), expectedLogits))
    }

    // MARK: - Registry

    @Test func visionFactoryBuildsQwen35WithEveryManifestModuleRotated() async throws {
        await PrismHadamardQwen35.register()
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: Self.config(), modelType: PrismHadamardQwen35.modelType)
        try expectRotated(model)
    }

    @Test func textFactoryBuildsQwen35WithEveryManifestModuleRotated() async throws {
        await PrismHadamardQwen35.register()
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Self.config(), modelType: PrismHadamardQwen35.modelType)
        try expectRotated(model)
    }

    private func expectRotated(_ model: any LanguageModel) throws {
        let leaves = Self.leaves(model)
        for entry in Self.manifestModules() {
            let path = "language_model." + (entry["path"] as! String)
            let module = try #require(leaves[path], "missing \(path)")
            if entry["embedding"] as! Bool {
                #expect(module is HadamardQuantizedEmbedding, "\(path) is \(type(of: module))")
            } else {
                #expect(module is HadamardQuantizedLinear, "\(path) is \(type(of: module))")
            }
        }
        let keys = Set(model.parameters().flattened().map(\.0))
        #expect(keys.contains("language_model.model.layers.0.linear_attn.in_proj_qkv.signs"))
        #expect(keys.contains("language_model.model.embed_tokens.signs"))
        #expect(keys.contains("language_model.lm_head.scales"))
        #expect(!keys.contains("language_model.model.layers.0.linear_attn.in_proj_a.signs"))
        let rotated = leaves.values.filter {
            $0 is HadamardQuantizedLinear || $0 is HadamardQuantizedEmbedding
        }
        #expect(rotated.count == Self.manifestModules().count)
    }

    // MARK: - Post-load preparation

    @Test func prepareLoadedCastsUnpackedFloatsAndKeepsPackedAndALog() async throws {
        await PrismHadamardQwen35.register()
        let data = try Self.config()
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: PrismHadamardQwen35.modelType)
        let policy = try PrismHadamardQwen35.policy(configJSON: data)
        try policy.prepareLoaded(model)

        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let prefix = "language_model.model.layers.0."
        #expect(parameters["language_model.model.norm.weight"]?.dtype == .float16)
        #expect(parameters[prefix + "linear_attn.in_proj_a.weight"]?.dtype == .float16)
        #expect(parameters[prefix + "linear_attn.A_log"]?.dtype == .float32)
        #expect(parameters[prefix + "linear_attn.in_proj_qkv.weight"]?.dtype == .uint32)
        #expect(parameters[prefix + "linear_attn.in_proj_qkv.signs"]?.dtype == .float32)
    }

    @Test func prepareLoadedRejectsSignsOtherThanPlusOrMinusOne() async throws {
        await PrismHadamardQwen35.register()
        let data = try Self.config()
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: PrismHadamardQwen35.modelType)
        let path = "language_model.model.layers.1.mlp.up_proj"
        model.update(parameters: ModuleParameters.unflattened([
            path + ".signs": MLXArray.ones([256], dtype: .float32) * 0.5
        ]))
        let policy = try PrismHadamardQwen35.policy(configJSON: data)

        #expect(throws: HadamardQuantizedCheckpointError.invalidSigns(path)) {
            try policy.prepareLoaded(model)
        }
    }

    // MARK: - Manifest validation

    @Test func manifestOutsideTheSupportedContractIsRejected() throws {
        let cases: [([String: Any], HadamardQuantizedCheckpointError)] = [
            (["schema_version": 1], .unsupportedSchemaVersion(1)),
            (["tensor_namespace": "mlx-lm-qwen3_5"],
             .unsupportedTensorNamespace("mlx-lm-qwen3_5", expected: "mlx-vlm-qwen3_5")),
            (["base_model_type": "llama"], .unsupportedBaseModelType("llama", expected: "qwen3_5")),
            (["gdn_activation_layout": "interleaved"], .unsupportedActivationLayout("interleaved")),
        ]
        for (overrides, expected) in cases {
            let data = try Self.config(overrides: overrides)
            #expect(throws: expected) { try PrismHadamardQwen35.policy(configJSON: data) }
        }
    }

    @Test func blockThatDoesNotDivideTheWidthIsRejected() async throws {
        await PrismHadamardQwen35.register()
        var modules = Self.manifestModules()
        modules[0]["block"] = 512
        let data = try Self.config(modules: modules)

        await #expect(throws: HadamardQuantizedCheckpointError.unsupportedBlockSize(
            "language_model.lm_head", block: 512, width: 256)) {
            _ = try await VLMTypeRegistry.shared.createModel(
                configuration: data, modelType: PrismHadamardQwen35.modelType)
        }
    }

    @Test func manifestNamingAMissingModuleIsRejected() async throws {
        await PrismHadamardQwen35.register()
        var modules = Self.manifestModules()
        modules.append(["path": "model.layers.9.mlp.up_proj", "block": 128, "dtype": "float16"])
        let data = try Self.config(modules: modules)

        await #expect(throws: HadamardQuantizedCheckpointError.moduleNotFound(
            "language_model.model.layers.9.mlp.up_proj")) {
            _ = try await LLMTypeRegistry.shared.createModel(
                configuration: data, modelType: PrismHadamardQwen35.modelType)
        }
    }

    @Test func prismConfigIsRecognisedOnlyByItsModelType() throws {
        #expect(PrismHadamardQwen35.isPrismConfig(try Self.config()))
        #expect(!PrismHadamardQwen35.isPrismConfig(Data(#"{"model_type":"qwen3_5"}"#.utf8)))
        #expect(!PrismHadamardQwen35.isPrismConfig(Data("not json".utf8)))
    }
}
