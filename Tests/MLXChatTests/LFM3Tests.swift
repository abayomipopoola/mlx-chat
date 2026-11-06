import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXChat

/// Contract: LFM3 and LFM3-Omni load as text models through the app's LFM3 port, and cached
/// decoding (conv state, rotating sliding-window cache, full cache) matches a full forward.
@Suite(.serialized) struct LFM3Tests {

    // MARK: - Fixtures

    private static func textConfig(slidingWindow: Int = 4) -> [String: Any] {
        [
            "model_type": "lfm3",
            "vocab_size": 50,
            "hidden_size": 32,
            "num_hidden_layers": 5,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 64,
            "norm_eps": 1e-5,
            "conv_bias": false,
            "conv_L_cache": 7,
            "layer_types": ["conv", "full_attention", "conv", "sliding_attention", "full_attention"],
            "rope_parameters": [
                "full_attention": [
                    "partial_rotary_factor": 0.5, "rope_theta": 1_000_000.0, "rope_type": "default",
                ],
                "sliding_attention": ["rope_theta": 10_000, "rope_type": "default"],
            ],
            "sliding_window": slidingWindow,
            "tie_word_embeddings": true,
            "use_sandwich_norm": true,
        ]
    }

    private static func omniConfig() -> [String: Any] {
        [
            "model_type": "lfm3_omni",
            "eos_token_id": [124_900, 125_014],
            "quantization": ["group_size": 64, "bits": 8, "mode": "affine"],
            "vision_config": ["model_type": "lfm3_vl_vision", "hidden_size": 1024],
            "audio_encoder_config": ["model_type": "lfm3_omni_encoder"],
            "text_config": textConfig(),
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

    private static func makeModel(slidingWindow: Int = 4) async throws -> LFM3NestedModel {
        await LFM3.register()
        MLXRandom.seed(7)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: json(omniConfig().merging(
                ["text_config": textConfig(slidingWindow: slidingWindow)]) { $1 }),
            modelType: "lfm3_omni")
        let lfm3 = try #require(model as? LFM3NestedModel)
        eval(lfm3)
        return lfm3
    }

    // MARK: - Support gate

    @Test func omniConfigIsSupportedAsTextOnly() throws {
        let data = try Self.json(Self.omniConfig())
        #expect(ModelSupport.unsupportedReason(configJSON: data) == nil)
        #expect(!ModelSupport.visionCapable(configJSON: data))
        let imported = ModelCatalog.importedModel(
            repoID: "LiquidAI/LFM3-Omni-2B-MLX-8bit",
            supportsVision: ModelSupport.visionCapable(configJSON: data), configJSON: data)
        #expect(!imported.supportsVision)
        // LFM3-Omni closes <think> at once even when the template opens it, so no toggle.
        #expect(imported.thinking == nil)
    }

    /// LFM3-1.7B's MLX config carries an empty `vision_config`.
    @Test func textConfigImportsWithThinkingToggle() throws {
        let data = try Self.json(Self.textConfig().merging(["vision_config": [:]]) { $1 })
        #expect(ModelSupport.unsupportedReason(configJSON: data) == nil)
        let imported = ModelCatalog.importedModel(
            repoID: "LiquidAI/LFM3-1.7B-MLX-8bit",
            supportsVision: ModelSupport.visionCapable(configJSON: data), configJSON: data)
        #expect(!imported.supportsVision)
        #expect(imported.thinking == .thinkTags(startsInside: true, toggleable: true))
        #expect(imported.extraEOSTokens.contains("<|im_end|>"))
    }

    @Test func configReadsTextConfigOrTopLevel() throws {
        let nested = try LFM3Configuration.decode(configJSON: Self.json(Self.omniConfig()))
        let flat = try LFM3Configuration.decode(configJSON: Self.json(Self.textConfig()))
        for config in [nested, flat] {
            #expect(config.hiddenLayers == 5)
            #expect(config.headDim == 8)
            #expect(config.isSliding(3) && config.isAttention(1) && !config.isAttention(0))
            #expect(config.ropeParameters["full_attention"]?["partial_rotary_factor"]?.asFloat() == 0.5)
        }
    }

    // MARK: - RoPE

    /// Hand-written reference of `Lfm3RotaryEmbedding` + `apply_rotary_pos_emb`: the first
    /// `partial * dim / 2` inverse frequencies follow the full-dimension progression, the rest are
    /// zero, and dimension i rotates with i + dim/2.
    @Test(arguments: [Float(0.5), 1])
    func ropeMatchesZeroTailReference(partial: Float) {
        let dim = 8, length = 3, offset = 5, base: Float = 1_000_000
        let rope = makeLFM3RoPE(
            headDim: dim,
            parameters: ["rope_theta": .float(base), "partial_rotary_factor": .float(partial)])
        let x = Self.values([1, 2, length, dim])
        let actual = rope(x, offset: offset)

        let angles = Int(partial * Float(dim)) / 2
        let half = dim / 2
        let input = x.asArray(Float.self)
        var expected = input
        for row in 0 ..< 2 * length {
            let position = Float(offset + row % length)
            for i in 0 ..< half {
                let invFreq = i < angles ? 1 / pow(base, Float(2 * i) / Float(dim)) : 0
                let (c, s) = (cos(position * invFreq), sin(position * invFreq))
                let a = input[row * dim + i], b = input[row * dim + i + half]
                expected[row * dim + i] = a * c - b * s
                expected[row * dim + i + half] = b * c + a * s
            }
        }
        #expect(Self.maxDifference(actual, MLXArray(expected, x.shape)) < 1e-5)
    }

    // MARK: - Weights

    @Test func sanitizeKeepsTheLanguageBackboneOnly() async throws {
        let model = try await Self.makeModel()
        let weights = model.sanitize(weights: [
            "language_model.model.embed_tokens.weight": MLXArray.zeros([50, 32]),
            "language_model.model.layers.0.conv.conv.weight": MLXArray.zeros([32, 1, 7]),
            "language_model.lm_head.weight": MLXArray.zeros([50, 32]),
            "vision_tower.encoder.weight": MLXArray.zeros([2]),
            "depth_decoder.mask_embedding": MLXArray.zeros([2]),
            "codebook_embedding.weight": MLXArray.zeros([2]),
        ])
        #expect(Set(weights.keys) == [
            "language_model.model.embed_tokens.weight",
            "language_model.model.layers.0.conv.conv.weight",
        ])
        #expect(weights["language_model.model.layers.0.conv.conv.weight"]?.shape == [32, 7, 1])
    }

    @Test func sanitizeAcceptsWeightsWithoutTheWrapper() async throws {
        let model = try await Self.makeModel()
        let weights = model.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([50, 32]),
            "lm_head.weight": MLXArray.zeros([50, 32]),
        ])
        #expect(Set(weights.keys) == ["language_model.model.embed_tokens.weight"])
    }

    @Test func bothTypesBuildTheNestedModel() async throws {
        await LFM3.register()
        for (type, config) in [("lfm3", Self.textConfig()), ("lfm3_omni", Self.omniConfig())] {
            let model = try await LLMTypeRegistry.shared.createModel(
                configuration: Self.json(config), modelType: type)
            #expect(model is LFM3NestedModel, "\(type) built \(Swift.type(of: model))")
        }
    }

    @Test func modelParametersMatchTheCheckpointLayout() async throws {
        let model = try await Self.makeModel()
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for key in [
            "model.embed_tokens.weight", "model.embedding_norm.weight",
            "model.layers.0.conv.conv.weight", "model.layers.0.conv.in_proj.weight",
            "model.layers.0.post_operator_norm.weight", "model.layers.0.post_ffn_norm.weight",
            "model.layers.1.self_attn.gate_proj.weight", "model.layers.1.self_attn.q_norm.weight",
            "model.layers.1.self_attn.o_proj.weight", "model.layers.3.feed_forward.w2.weight",
        ] {
            #expect(parameters["language_model." + key] != nil, "missing \(key)")
        }
        #expect(!parameters.keys.contains { $0.hasPrefix("language_model.lm_head") })
        #expect(parameters["language_model.model.layers.0.conv.conv.weight"]?.shape == [32, 7, 1])
    }

    /// The 4-bit LFM3-Omni build keeps some layers at 6 bits and names them, prefix and all, in
    /// `quantization`. Loading must give those layers 6 bits: the names only match when the
    /// module paths keep `language_model.`.
    @Test func mixedBitCheckpointLoadsWithPerLayerBits() async throws {
        let sixBit = "language_model.model.layers.0.feed_forward.w2"
        let source = try await Self.makeModel()
        quantize(model: source) { path, _ in
            (groupSize: 32, bits: path == sixBit ? 6 : 4, mode: .affine)
        }
        var weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        weights["vision_tower.encoder.weight"] = MLXArray.zeros([2])

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("lfm3-mixed-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try save(arrays: weights, metadata: ["format": "mlx"],
                 url: directory.appendingPathComponent("model.safetensors"))

        var config = Self.omniConfig()
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
        #expect((leaves["language_model.model.layers.1.feed_forward.w2"] as? QuantizedLinear)?
            .bits == 4)
        let tokens = MLXArray([Int32(3), 17, 8]).reshaped(1, -1)
        #expect(Self.maxDifference(model(tokens, cache: nil), source(tokens, cache: nil)) == 0)
    }

    // MARK: - Caching

    /// Prefill in two chunks, then decode one token at a time past the sliding window: every
    /// step's logits must match an uncached forward over the whole sequence. Runs on the CPU:
    /// the GPU's matrix-matrix and matrix-vector kernels round float32 differently, which would
    /// hide a real cache bug behind kernel noise. The sandboxed test host can't build fused CPU
    /// kernels, so compilation is off for the duration.
    @Test(arguments: [4, 64])
    func cachedDecodingMatchesFullForward(slidingWindow: Int) async throws {
        let model = try await Self.makeModel(slidingWindow: slidingWindow)
        compile(enable: false)
        defer { compile(enable: true) }
        let tokens: [Int32] = [3, 17, 8, 42, 1, 9, 30, 12, 5, 26, 44, 7]
        Device.withDefaultDevice(.cpu) {
            let full = model(MLXArray(tokens).reshaped(1, -1), cache: nil)

            let cache = model.newCache(parameters: nil)
            #expect(cache[3] is RotatingKVCache)
            #expect(cache[1] is KVCacheSimple)
            #expect(cache[0] is MambaCache)

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

    /// Opt-in (`MLXCHAT_RUN_LFM3_SMOKE=1`): greedy chat through `MLXEngine` on each downloaded
    /// LFM3-Omni build, with a prompt longer than the sliding window and a second turn that
    /// reuses the session cache. The 4-bit build mixes in 6-bit layers.
    @Test(.timeLimit(.minutes(10)), arguments: [
        "LiquidAI/LFM3-Omni-2B-MLX-8bit", "LiquidAI/LFM3-Omni-2B-MLX-4bit",
    ])
    func omniCheckpointAnswersInText(id: String) async throws {
        guard ProcessInfo.processInfo.environment["MLXCHAT_RUN_LFM3_SMOKE"] == "1",
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

        let conversation = UUID()
        var history: [EngineMessage] = []
        func ask(_ question: String) async throws -> String {
            history.append(EngineMessage(role: .user, text: question))
            var reply = ""
            for try await event in engine.stream(
                history: history, systemPrompt: "You are a helpful assistant.",
                temperature: 0, conversationID: conversation
            ) {
                switch event {
                case .delta(let delta): reply += delta
                case .thinkingDelta, .thinkingReclassified:
                    Issue.record("LFM3 has no thinking config, so nothing should be split out")
                default: break
                }
            }
            print("LFM3 \(id.suffix(4)) >", question.suffix(60), "=>", reply)
            // The template pre-closes the reasoning span; its tags must not leak.
            #expect(!reply.contains("<think>") && !reply.contains("</think>"))
            history.append(EngineMessage(role: .assistant, text: reply))
            return reply
        }

        #expect(try await ask("What is the capital of France? Answer in one word.")
            .localizedCaseInsensitiveContains("paris"))
        let filler = (1 ... 60).map { "Item \($0) is a plain grey stone." }.joined(separator: " ")
        #expect(try await ask(
            filler + " Ignore the items. What colour is the sky on a clear day? One word.")
            .localizedCaseInsensitiveContains("blue"))
        #expect(try await ask("Which city did I ask about in my first message? One word.")
            .localizedCaseInsensitiveContains("paris"))
    }

    /// Opt-in (`MLXCHAT_RUN_LFM3_SMOKE=1`): `LiquidAI/LFM3-1.7B-MLX-8bit` through `MLXEngine`
    /// with the app's thinking toggle on and off. On, the template opens <think> and the reasoning
    /// arrives as thinking; off, it pre-closes the span and the reply is plain text.
    @Test(.timeLimit(.minutes(10)))
    func textCheckpointThinksOnlyWhenToggledOn() async throws {
        let id = "LiquidAI/LFM3-1.7B-MLX-8bit"
        guard ProcessInfo.processInfo.environment["MLXCHAT_RUN_LFM3_SMOKE"] == "1",
              let directory = ModelStore.localMLXModelDirectory(for: id)
        else { return }

        let config = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let model = ModelCatalog.importedModel(
            repoID: id, supportsVision: ModelSupport.visionCapable(configJSON: config),
            configJSON: config)
        #expect(model.thinking != nil)
        let engine = MLXEngine(model: model)
        try await engine.load { _ in }
        defer { engine.unload() }

        let defaults = UserDefaults.standard
        let savedThinking = defaults.object(forKey: Keys.thinkingEnabled)
        defer { defaults.set(savedThinking, forKey: Keys.thinkingEnabled) }

        for thinkingOn in [true, false] {
            defaults.set(thinkingOn, forKey: Keys.thinkingEnabled)
            let question = "How many letters r are in the word strawberry? End with the number."
            var reply = "", thinking = "", reclassified = false
            for try await event in engine.stream(
                history: [EngineMessage(role: .user, text: question)],
                systemPrompt: "You are a helpful assistant.",
                temperature: 0, conversationID: UUID()
            ) {
                switch event {
                case .delta(let delta): reply += delta
                case .thinkingDelta(let delta): thinking += delta
                case .thinkingReclassified: reclassified = true
                default: break
                }
            }
            print("LFM3-1.7B thinking=\(thinkingOn): [\(thinking.count) chars thinking]",
                  thinking.suffix(300), "=> reply:", reply)
            #expect(!reclassified)
            #expect(!reply.contains("<think>") && !reply.contains("</think>"))
            #expect(!reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            if thinkingOn {
                #expect(!thinking.isEmpty)
                #expect(reply.contains("3"))
            } else {
                #expect(thinking.isEmpty)
            }
        }
    }
}
