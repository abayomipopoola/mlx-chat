import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXChat

/// Contract: Spark-X2.5/X2.6 MLX conversions (`spark2_5`) load through the app's port, cached
/// decoding (rotating sliding-window cache, full cache) matches a full forward, and PyTorch
/// quantized builds such as XHToken's FP8 release are refused at import.
@Suite(.serialized) struct Spark25Tests {

    // MARK: - Fixtures

    private static func config(slidingWindow: Int = 4) -> [String: Any] {
        [
            "model_type": "spark2_5",
            "vocab_size": 50,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "intermediate_size": 64,
            "rms_norm_eps": 1e-6,
            "hidden_act": "gelu",
            "headwise_attn_output_gate": true,
            "gate_attn_act_mode": "sigmoid",
            "layer_types": ["sliding_attention", "sliding_attention", "full_attention",
                            "sliding_attention"],
            "rope_parameters": [
                "full_attention": ["partial_rotary_factor": 0.25, "rope_theta": 5_000_000],
                "sliding_attention": ["partial_rotary_factor": 1.0, "rope_theta": 10_000],
            ],
            "sliding_window": slidingWindow,
            "tie_word_embeddings": true,
        ]
    }

    private static func json(_ object: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: object)
    }

    private static func values(_ shape: [Int], phase: Float = 0) -> MLXArray {
        let count = shape.reduce(1, *)
        return MLXArray((0 ..< count).map { sin(Float($0) * 0.37 + phase) }, shape)
    }

    private static func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private static func makeModel(slidingWindow: Int = 4) async throws -> Spark25Model {
        await Spark25.register()
        MLXRandom.seed(11)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: json(config(slidingWindow: slidingWindow)),
            modelType: Spark25.modelType)
        let spark = try #require(model as? Spark25Model)
        eval(spark)
        return spark
    }

    // MARK: - Support gate

    @Test func mlxConversionImportsWithThinkingToggle() throws {
        var config = Self.config()
        config["quantization"] = ["group_size": 64, "bits": 8, "mode": "affine"]
        config["quantization_config"] = ["group_size": 64, "bits": 8, "mode": "affine"]
        let data = try Self.json(config)
        #expect(ModelSupport.unsupportedReason(configJSON: data) == nil)
        let imported = ModelCatalog.importedModel(
            repoID: "code-and-canvas/Spark-X2.6-4B-MLX-oQ8e-fp16",
            supportsVision: ModelSupport.visionCapable(configJSON: data), configJSON: data)
        #expect(!imported.supportsVision)
        #expect(imported.thinking == .thinkTags(startsInside: true, toggleable: true))
    }

    /// XHToken/Spark-X2.6-4B-FP8: compressed-tensors FP8 weights MLX can't read.
    @Test func pytorchFP8BuildIsRefusedAtImport() throws {
        var config = Self.config()
        config["quantization_config"] = [
            "quant_method": "compressed-tensors", "format": "float-quantized",
        ]
        let reason = ModelSupport.unsupportedReason(configJSON: try Self.json(config))
        #expect(reason?.contains("PyTorch compressed-tensors build") == true, "\(reason ?? "nil")")
    }

    @Test func configRejectsOtherActivations() throws {
        var config = Self.config()
        config["hidden_act"] = "silu"
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(Spark25Configuration.self, from: Self.json(config))
        }
    }

    // MARK: - RoPE

    /// Hand-written reference of `compute_rope_cos_sin` + `apply_rotary_pos_emb`: the first
    /// `partial * dim` dimensions rotate with that width's frequencies, dimension i with
    /// i + width/2, and the rest pass through.
    @Test(arguments: [Float(0.25), 1])
    func ropeMatchesPartialReference(partial: Float) {
        let dim = 16, length = 3, offset = 5, base: Float = 5_000_000
        let rope = makeSpark25RoPE(
            headDim: dim,
            parameters: ["rope_theta": .float(base), "partial_rotary_factor": .float(partial)])
        let x = Self.values([1, 2, length, dim])
        let actual = rope(x, offset: offset)

        let width = Int(Float(dim) * partial)
        let half = width / 2
        let input = x.asArray(Float.self)
        var expected = input
        for row in 0 ..< 2 * length {
            let position = Float(offset + row % length)
            for i in 0 ..< half {
                let invFreq = 1 / pow(base, Float(2 * i) / Float(width))
                let (c, s) = (cos(position * invFreq), sin(position * invFreq))
                let a = input[row * dim + i], b = input[row * dim + i + half]
                expected[row * dim + i] = a * c - b * s
                expected[row * dim + i + half] = b * c + a * s
            }
        }
        #expect(Self.maxDifference(actual, MLXArray(expected, x.shape)) < 1e-4)
    }

    // MARK: - Weights

    @Test func modelParametersMatchTheCheckpointLayout() async throws {
        let model = try await Self.makeModel()
        let keys = Set(model.parameters().flattened().map(\.0))
        for key in [
            "model.embedding.weight", "model.norm.weight",
            "model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_k_v_proj.weight", "model.layers.0.self_attn.g_proj.weight",
            "model.layers.0.self_attn.out_proj.weight", "model.layers.2.mlp.gate_proj.weight",
            "model.layers.2.mlp.up_proj.weight", "model.layers.2.mlp.down_proj.weight",
        ] {
            #expect(keys.contains(key), "missing \(key)")
        }
        #expect(!keys.contains { $0.hasPrefix("lm_head") })
        let sanitized = model.sanitize(weights: [
            "model.norm.weight": MLXArray.zeros([32]), "lm_head.weight": MLXArray.zeros([50, 32]),
        ])
        #expect(Set(sanitized.keys) == ["model.norm.weight"])
    }

    /// The 4-bit oQ conversions keep some layers at other widths and name them in
    /// `quantization`; those names must match the module paths.
    @Test func mixedBitCheckpointLoadsWithPerLayerBits() async throws {
        let sixBit = "model.layers.0.mlp.down_proj"
        let source = try await Self.makeModel()
        quantize(model: source) { path, _ in
            (groupSize: 32, bits: path == sixBit ? 6 : 4, mode: .affine)
        }
        let weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("spark-mixed-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try save(arrays: weights, metadata: ["format": "mlx"],
                 url: directory.appendingPathComponent("model.safetensors"))

        var config = Self.config()
        config["quantization"] = [
            "group_size": 32, "bits": 4, "mode": "affine",
            sixBit: ["group_size": 32, "bits": 6, "mode": "affine"],
        ]
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: Self.json(config))
        let model = try await Self.makeModel()
        try loadWeights(modelDirectory: directory, model: model,
                        perLayerQuantization: base.perLayerQuantization)

        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        #expect((leaves[sixBit] as? QuantizedLinear)?.bits == 6)
        #expect((leaves["model.layers.1.mlp.down_proj"] as? QuantizedLinear)?.bits == 4)
        let tokens = MLXArray([Int32(3), 17, 8]).reshaped(1, -1)
        #expect(Self.maxDifference(model(tokens, cache: nil), source(tokens, cache: nil)) == 0)
    }

    // MARK: - Caching

    /// Prefill in two chunks, then decode one token at a time past the sliding window: every
    /// step's logits must match an uncached forward. On the CPU with compilation off, for the
    /// reasons given in `LFM3Tests.cachedDecodingMatchesFullForward`.
    @Test(arguments: [4, 64])
    func cachedDecodingMatchesFullForward(slidingWindow: Int) async throws {
        let model = try await Self.makeModel(slidingWindow: slidingWindow)
        compile(enable: false)
        defer { compile(enable: true) }
        let tokens: [Int32] = [3, 17, 8, 42, 1, 9, 30, 12, 5, 26, 44, 7]
        Device.withDefaultDevice(.cpu) {
            let full = model(MLXArray(tokens).reshaped(1, -1), cache: nil)

            let cache = model.newCache(parameters: nil)
            #expect(cache[0] is RotatingKVCache)
            #expect(cache[2] is KVCacheSimple)

            var start = 0
            for chunk in [5, 3, 1, 1, 1, 1] {
                let slice = Array(tokens[start ..< start + chunk])
                let logits = model(MLXArray(slice).reshaped(1, -1), cache: cache)
                let reference = full[0..., start ..< start + chunk, 0...]
                let difference = Self.maxDifference(logits, reference)
                #expect(difference < 1e-4, "chunk at \(start) drifted by \(difference)")
                start += chunk
            }
            #expect(start == tokens.count)
        }
    }

    // MARK: - Real checkpoint

    /// Opt-in (`MLXCHAT_RUN_SPARK_SMOKE=1`): `code-and-canvas/Spark-X2.6-4B-MLX-oQ8e-fp16`
    /// through `MLXEngine` with the thinking toggle on and off, plus a prompt longer than the
    /// 512-token sliding window and a follow-up that reuses the session cache.
    @Test(.timeLimit(.minutes(10)))
    func mlxConversionAnswers() async throws {
        let id = "code-and-canvas/Spark-X2.6-4B-MLX-oQ8e-fp16"
        guard ProcessInfo.processInfo.environment["MLXCHAT_RUN_SPARK_SMOKE"] == "1",
              let directory = ModelStore.localMLXModelDirectory(for: id)
        else { return }

        let config = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        #expect(ModelSupport.unsupportedReason(configJSON: config) == nil)
        let model = ModelCatalog.importedModel(
            repoID: id, supportsVision: ModelSupport.visionCapable(configJSON: config),
            configJSON: config)
        let engine = MLXEngine(model: model)
        try await engine.load { _ in }
        defer { engine.unload() }

        let defaults = UserDefaults.standard
        let savedThinking = defaults.object(forKey: Keys.thinkingEnabled)
        defer { defaults.set(savedThinking, forKey: Keys.thinkingEnabled) }

        for thinkingOn in [true, false] {
            defaults.set(thinkingOn, forKey: Keys.thinkingEnabled)
            let conversation = UUID()
            var history: [EngineMessage] = []
            func ask(_ question: String) async throws -> (reply: String, thinking: String) {
                history.append(EngineMessage(role: .user, text: question))
                var reply = "", thinking = "", reclassified = false
                for try await event in engine.stream(
                    history: history, systemPrompt: "You are a helpful assistant.",
                    temperature: 0, conversationID: conversation
                ) {
                    switch event {
                    case .delta(let delta): reply += delta
                    case .thinkingDelta(let delta): thinking += delta
                    case .thinkingReclassified: reclassified = true
                    default: break
                    }
                }
                print("Spark thinking=\(thinkingOn) >", question.suffix(60),
                      "=> [\(thinking.count) chars thinking]", reply)
                #expect(!reclassified)
                #expect(!reply.contains("<think>") && !reply.contains("</think>"))
                #expect(thinkingOn ? !thinking.isEmpty : thinking.isEmpty)
                history.append(EngineMessage(role: .assistant, text: reply))
                return (reply, thinking)
            }

            #expect(try await ask("How many letters r are in the word strawberry? "
                + "End with the number.").reply.contains("3"))
            guard !thinkingOn else { continue }
            let filler = (1 ... 120).map { "Item \($0) is a plain grey stone." }
                .joined(separator: " ")
            #expect(try await ask(
                filler + " Ignore the items. What colour is the sky on a clear day? One word.")
                .reply.localizedCaseInsensitiveContains("blue"))
            #expect(try await ask("Which word did I ask you to count letters in? One word.")
                .reply.localizedCaseInsensitiveContains("strawberry"))
        }
    }
}
