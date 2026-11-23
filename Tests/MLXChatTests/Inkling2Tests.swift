// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing
@testable import MLXChat

// MARK: - Helpers

private func referenceRelativeAttention(
    q: MLXArray, k: MLXArray, v: MLXArray, rel: MLXArray, sinks: MLXArray, scale: Float,
    window: Int, distance: MLXArray
) -> MLXArray {
    let (b, hq, l, d) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
    let (hkv, s) = (k.dim(1), k.dim(2))
    let extent = rel.dim(-1)
    let group = hq / hkv
    let kk = repeated(k.asType(.float32), count: group, axis: 1)
    let vv = repeated(v.asType(.float32), count: group, axis: 1)
    var logits = matmul(q.asType(.float32), kk.swappedAxes(-1, -2)) * scale
    let index = broadcast(clip(distance, min: 0, max: extent - 1), to: [b, hq, l, s])
    var bias = takeAlong(rel.asType(.float32).transposed(0, 2, 1, 3), index, axis: -1)
    bias = which(distance .< extent, bias, MLXArray(Float(0)))
    var visible = distance .>= 0
    if window > 0 {
        visible = logicalAnd(visible, distance .< window)
    }
    logits = which(visible, logits + bias, MLXArray(-Float.infinity))
    let sinkLogits = sinks.asType(.float32).transposed(0, 2, 1)[.ellipsis, .newAxis]
    let weights = softmax(concatenated([logits, sinkLogits], axis: -1), axis: -1)[
        .ellipsis, ..<s]
    return matmul(weights, vv).transposed(0, 2, 1, 3).reshaped(b, l, hq, d)
}

private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

private final class InklingTestTokenizer: Tokenizer, @unchecked Sendable {
    static let controls = [
        "<|message_model|>": 200_001, "<|content_text|>": 200_004,
        "<|content_model_end_sampling|>": 200_006, "<|content_thinking|>": 200_008,
        "<|end_message|>": 200_010, "<|content_invoke_tool_json|>": 200_049,
    ]
    private var nextID = 1
    private var textToID = InklingTestTokenizer.controls
    private var idToText = Dictionary(
        uniqueKeysWithValues: InklingTestTokenizer.controls.map { ($1, $0) })

    func id(_ text: String) -> Int {
        if let id = textToID[text] { return id }
        let id = nextID
        nextID += 1
        textToID[text] = id
        idToText[id] = text
        return id
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [id(text)] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { idToText[$0] }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { textToID[token] }
    func convertIdToToken(_ id: Int) -> String? { idToText[id] }
    var bosToken: String? { nil }
    var eosToken: String? { "<|content_model_end_sampling|>" }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

// MARK: - Configuration, Registration & Checkpoint Sanitization

@Suite("Inkling2 configuration and checkpoint")
struct Inkling2ConfigurationTests {

    static let miniRawJSON = """
        {
          "model_type": "inkling2_loop",
          "eos_token_id": 200006,
          "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4"
          },
          "text_config": {
            "hidden_size": 2304, "num_hidden_layers": 24, "vocab_size": 201024,
            "unpadded_vocab_size": 200058, "num_attention_heads": 24, "num_key_value_heads": 8,
            "head_dim": 128, "swa_head_dim": 128, "swa_num_attention_heads": 24,
            "swa_num_key_value_heads": 8, "sliding_window_size": 1024,
            "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
            "d_rel": 16, "rel_extent": 1024, "log_scaling_n_floor": 8192,
            "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-06,
            "qk_l2_clip_threshold": 22.627416997969522, "rel_l2_clip_threshold": 8.0,
            "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 6, "n_routed_experts": 128,
            "num_experts_per_tok": 8, "intermediate_size": 1152, "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0, "route_scale": 8.0, "final_logit_softcapping": 30.0,
            "loop_config": {
              "core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64,
              "is_varloop": false
            }
          },
          "vision_config": {
            "native_patch_size": 20, "adapter_merge_size": 2, "projection_hidden_dim": 4096,
            "hidden_size": 1024, "num_hidden_layers": 23, "num_attention_heads": 16
          }
        }
        """

    @Test("Both inkling2 and inkling2_loop are registered in VLM and LLM registries")
    func bothTypesAreRegistered() async throws {
        await Inkling2.register()
        for type in ["inkling2", "inkling2_loop"] {
            #expect(ModelSupport.supportedModelTypes.contains(type))
            #expect(await VLMTypeRegistry.shared.contains(type))
            #expect(await LLMTypeRegistry.shared.contains(type))
        }
        let tiny = Data(Inkling2ModelTests.tinyJSON.utf8)
        let vlm = try await VLMTypeRegistry.shared.createModel(
            configuration: tiny, modelType: "inkling2_loop")
        let llm = try await LLMTypeRegistry.shared.createModel(
            configuration: tiny, modelType: "inkling2")
        #expect(vlm is Inkling2)
        #expect(llm is Inkling2)
    }

    @Test("Raw config with quantization_config and no quantization key decodes and passes support gate")
    func rawModelOptConfigDecodesWithoutQuantizationKey() throws {
        let data = Data(Self.miniRawJSON.utf8)
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        #expect(base.modelType == "inkling2_loop")
        #expect(base.perLayerQuantization == nil)
        #expect(base.effectiveEOSTokenIds == [200006])

        let config = try JSONDecoder.json5().decode(Inkling2Configuration.self, from: data)
        let text = config.textConfiguration
        let visits = text.layerVisits
        #expect(visits.count == 48)
        #expect(Array(visits.prefix(7)) == [0, 1, 2, 3, 4, 5, 6])
        #expect(Array(visits[18 ..< 20]) == [6, 7])
        #expect(Array(visits.suffix(6)) == Array(18 ..< 24))
        #expect(config.imageTokenId == 200054)
        #expect(config.visionConfiguration?.blockSize == 40)
        #expect(text.isSliding(0) && !text.isSliding(5) && !text.isSliding(23))

        #expect(ModelSupport.unsupportedReason(configJSON: data) == nil)
        #expect(ModelSupport.visionCapable(configJSON: data))

        let mlxConverted = Data(#"{"model_type":"inkling2","quantization":{"bits":4,"group_size":64,"mode":"affine"},"vision_config":{}}"#.utf8)
        #expect(ModelSupport.unsupportedReason(configJSON: mlxConverted) == nil)

        // Non-Inkling modelopt/compressed-tensors still rejected.
        let rejected = Data(#"{"model_type":"qwen3","quantization_config":{"quant_method":"modelopt"}}"#.utf8)
        #expect(ModelSupport.unsupportedReason(configJSON: rejected) != nil)
    }

    @Test("The processor configuration reads the vision settings of config.json")
    func processorConfiguration() throws {
        let config = try JSONDecoder().decode(
            Inkling2ProcessorConfiguration.self, from: Data(Self.miniRawJSON.utf8))
        #expect(config.patchSize == 20)
        #expect(config.mergeSize == 2)
        #expect(config.imageTokenId == 200054)
    }

    @Test("Expert formats follow the tensor shapes")
    func expertFormats() {
        let experts = 4
        let nvfp4: [String: MLXArray] = [
            "e.weight": MLXArray.zeros([experts, 8, 64 / 8], dtype: .uint32),
            "e.scales": MLXArray.zeros([experts, 8, 64 / 16], dtype: .uint8),
            "e.global_scale": MLXArray.ones([experts]),
        ]
        #expect(
            Inkling2ExpertFormat(path: "e", weights: nvfp4, inputDims: 64)
                == Inkling2ExpertFormat(mode: .nvfp4, bits: 4, groupSize: 16, hasGlobalScale: true))

        let mxfp8: [String: MLXArray] = [
            "e.weight": MLXArray.zeros([experts, 8, 64 / 4], dtype: .uint32),
            "e.scales": MLXArray.zeros([experts, 8, 64 / 32], dtype: .uint8),
        ]
        #expect(
            Inkling2ExpertFormat(path: "e", weights: mxfp8, inputDims: 64)
                == Inkling2ExpertFormat(mode: .mxfp8, bits: 8, groupSize: 32, hasGlobalScale: false)
        )
        #expect(Inkling2ExpertFormat(path: "missing", weights: mxfp8, inputDims: 64) == nil)
    }

    @Test("Raw Thinking Machines keys map cleanly onto the MLX Inkling2 layout")
    func rawKeyMappingAndNVFP4GlobalScaleAndConvTranspose() throws {
        let dims = 64
        let experts = 4
        let inner = 32

        let amax = MLXArray([Float(1.5), 2.5, 3.0, 4.5])
        let denom = MLXArray([Float(2.0)])
        let convOrig = MLXArray((0 ..< (dims * 4)).map { Float($0) }, [dims, 1, 4]).asType(.bfloat16)
        let qCross = MLXArray((0 ..< (4 * dims)).map { Float($0) }, [4, dims])
        let kvCross = MLXArray((0 ..< (4 * dims)).map { Float($0 + 1000) }, [4, dims])

        let rawWeights: [String: MLXArray] = [
            "model.llm.embed.weight": MLXArray.zeros([50, dims], dtype: .bfloat16),
            "model.llm.embed_norm.weight": MLXArray.ones([dims], dtype: .bfloat16),
            "model.llm.norm.weight": MLXArray.ones([dims], dtype: .bfloat16),
            "model.llm.unembed.weight": MLXArray.zeros([50, dims], dtype: .bfloat16),
            "model.llm.attn_res.attn_queries.0": MLXArray.ones([dims], dtype: .float32),
            "model.llm.attn_res.mlp_queries.0": MLXArray.ones([dims], dtype: .float32),
            "model.llm.attn_res.output_query": MLXArray.ones([dims], dtype: .float32),
            "model.llm.middle_cycle_cross_pass_queries": qCross,
            "model.llm.middle_cycle_cross_pass_kv_queries": kvCross,
            "model.llm.layers.0.attn.wq_du.weight": MLXArray.zeros([128, dims], dtype: .bfloat16),
            "model.llm.layers.0.attn.wk_dv.weight": MLXArray.zeros([64, dims], dtype: .bfloat16),
            "model.llm.layers.0.attn.wv_dv.weight": MLXArray.zeros([64, dims], dtype: .bfloat16),
            "model.llm.layers.0.attn.wr_du.weight": MLXArray.zeros([32, dims], dtype: .bfloat16),
            "model.llm.layers.0.attn.wo_ud.weight": MLXArray.zeros([dims, 128], dtype: .bfloat16),
            "model.llm.layers.0.attn.gate_attn.weight": MLXArray.zeros([128, dims], dtype: .bfloat16),
            "model.llm.layers.0.attn.attn_sink_proj.weight": MLXArray.zeros([4, dims], dtype: .bfloat16),
            "model.llm.layers.0.attn.k_sconv.weight": convOrig,
            "model.llm.layers.0.attn_sconv.weight": convOrig,
            "model.llm.layers.0.mlp_sconv.weight": convOrig,
            "model.llm.layers.0.attn_norm.weight": MLXArray.ones([dims], dtype: .bfloat16),
            "model.llm.layers.0.mlp_norm.weight": MLXArray.ones([dims], dtype: .bfloat16),
            "model.llm.layers.3.attn.k_sconv.static_w": MLXArray.zeros([4, 64], dtype: .bfloat16),
            "model.llm.layers.3.attn.k_sconv.weight_proj": MLXArray.zeros([8, dims], dtype: .bfloat16),
            "model.llm.layers.3.q_input_norm.weight": MLXArray.ones([dims], dtype: .bfloat16),
            "model.llm.layers.2.controllers.attn.down_weight": MLXArray.zeros([8, 2 * dims], dtype: .bfloat16),
            "model.llm.layers.2.controllers.attn.forget_weight": MLXArray.zeros([dims, 8], dtype: .bfloat16),
            "model.llm.layers.2.controllers.attn.gain_weight": MLXArray.zeros([dims, 8], dtype: .bfloat16),
            "model.llm.layers.0.mlp.gate.weight": MLXArray.zeros([experts, dims], dtype: .bfloat16),
            "model.llm.layers.0.mlp.gate.bias": MLXArray.zeros([experts], dtype: .float32),
            "model.llm.layers.0.mlp.experts.w13_weight": MLXArray.zeros([experts, 2 * inner, dims / 2], dtype: .uint8),
            "model.llm.layers.0.mlp.experts.w13_weight_scale": MLXArray.zeros([experts, 2 * inner, dims / 16], dtype: .uint8),
            "model.llm.layers.0.mlp.experts.w13_weight_amax": amax,
            "model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator": denom,
            "model.llm.layers.5.mlp.experts.w2_weight": MLXArray.zeros([experts, dims, inner], dtype: .uint8),
            "model.llm.layers.5.mlp.experts.w2_weight_scale": MLXArray.zeros([experts, dims, inner / 32], dtype: .uint8),
            "model.visual.backbone.embeddings.position_embedding": MLXArray.zeros([16, 32], dtype: .bfloat16),
            "model.visual.backbone.ln_pre.weight": MLXArray.ones([32], dtype: .bfloat16),
            "model.visual.backbone.layers.0.ln_1.weight": MLXArray.ones([32], dtype: .bfloat16),
            "model.visual.backbone.layers.0.ln_2.weight": MLXArray.ones([32], dtype: .bfloat16),
            "model.visual.backbone.layers.0.ls_1.gamma": MLXArray.ones([32], dtype: .bfloat16),
            "model.visual.backbone.layers.0.mlp.c_fc.weight": MLXArray.zeros([64, 32], dtype: .bfloat16),
            "model.visual.backbone.layers.0.mlp.c_proj.weight": MLXArray.zeros([32, 64], dtype: .bfloat16),
            "model.visual.proj_in.weight": MLXArray.zeros([64, 128], dtype: .bfloat16),
            "model.visual.proj_out.weight": MLXArray.zeros([dims, 64], dtype: .bfloat16),
            "mtp.layers.0.weight": MLXArray.zeros([dims], dtype: .bfloat16),
        ]

        let mapped = Inkling2.sanitizeCheckpointWeights(rawWeights, includeVision: true)

        // Top-level and attn_res (query 0 maps to layer 1).
        #expect(mapped["language_model.embed_tokens.weight"] != nil)
        #expect(mapped["lm_head.weight"] != nil)
        #expect(mapped["language_model.layers.1.attention_input.query"] != nil)
        #expect(mapped["language_model.layers.0.mlp_input.query"] != nil)
        #expect(mapped["language_model.final_input.query"] != nil)
        #expect(mapped["mtp.layers.0.weight"] == nil)

        // Attention, norms, dynamic conv, controllers.
        #expect(mapped["language_model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(mapped["language_model.layers.0.self_attn.sink_proj.weight"] != nil)
        #expect(mapped["language_model.layers.0.input_layernorm.weight"] != nil)
        #expect(mapped["language_model.layers.0.post_attention_layernorm.weight"] != nil)
        #expect(mapped["language_model.layers.3.self_attn.k_sconv.static_weight"] != nil)
        #expect(mapped["language_model.layers.3.self_attn.k_sconv.weight_proj.weight"] != nil)
        #expect(mapped["language_model.layers.3.q_input_layernorm.weight"] != nil)
        #expect(mapped["language_model.layers.2.q_message_gate.down_proj.weight"] != nil)
        #expect(mapped["language_model.layers.2.q_message_gate.forget_gate_proj.weight"]?.dtype == .float32)
        #expect(mapped["language_model.layers.2.q_message_gate.message_gate_proj.weight"]?.dtype == .float32)
        #expect(mapped["language_model.layers.0.mlp.gate.weight"]?.dtype == .float32)
        #expect(mapped["language_model.layers.0.mlp.gate.e_score_correction_bias"] != nil)

        // Conv weights transposed from [C, 1, 4] to [C, 4, 1].
        let convMapped = try #require(mapped["language_model.layers.0.self_attn.k_sconv.conv1d.weight"])
        #expect(convMapped.shape == [dims, 4, 1])
        #expect(maxDifference(convMapped, convOrig.swappedAxes(1, 2)) == 0)
        #expect(mapped["language_model.layers.0.post_attention_convolution.conv1d.weight"]?.shape == [dims, 4, 1])
        #expect(mapped["language_model.layers.0.post_mlp_convolution.conv1d.weight"]?.shape == [dims, 4, 1])

        // Cross-pass queries interleaved [q, k, v, mlp] with shape [2, 4, dims].
        let cross = try #require(mapped["language_model.message_input.query"])
        #expect(cross.shape == [2, 4, dims])
        #expect(maxDifference(cross[0, 0], qCross[0]) == 0)
        #expect(maxDifference(cross[0, 1], kvCross[0]) == 0)
        #expect(maxDifference(cross[0, 2], kvCross[1]) == 0)
        #expect(maxDifference(cross[0, 3], qCross[1]) == 0)

        // NVFP4 global_scale == amax * (6.0 * 448.0 / outer_scale_denominator) and uint32 view.
        let gateUpWeight = try #require(mapped["language_model.layers.0.mlp.experts.gate_up_proj.weight"])
        #expect(gateUpWeight.dtype == .uint32)
        #expect(gateUpWeight.shape == [experts, 2 * inner, dims / 8])
        let globalScale = try #require(mapped["language_model.layers.0.mlp.experts.gate_up_proj.global_scale"])
        let expectedGlobal = amax * (Float(6.0 * 448.0) / denom)
        #expect(maxDifference(globalScale, expectedGlobal) < 1e-4)

        // MXFP8 layer has uint32 view, uint8 scales, and no global_scale.
        let mxfp8Weight = try #require(mapped["language_model.layers.5.mlp.experts.down_proj.weight"])
        #expect(mxfp8Weight.dtype == .uint32)
        #expect(mxfp8Weight.shape == [experts, dims, inner / 4])
        #expect(mapped["language_model.layers.5.mlp.experts.down_proj.global_scale"] == nil)

        // Vision keys mapped onto vision_tower.*
        #expect(mapped["vision_tower.embeddings.position_embedding.weight"] != nil)
        #expect(mapped["vision_tower.pre_layernorm.weight"] != nil)
        #expect(mapped["vision_tower.layers.0.layernorm_before.weight"] != nil)
        #expect(mapped["vision_tower.layers.0.layernorm_after.weight"] != nil)
        #expect(mapped["vision_tower.layers.0.lambda_1"] != nil)
        #expect(mapped["vision_tower.layers.0.mlp.fc1.weight"] != nil)
        #expect(mapped["vision_tower.layers.0.mlp.fc2.weight"] != nil)
        #expect(mapped["vision_tower.merger.linear_1.weight"] != nil)
        #expect(mapped["vision_tower.merger.linear_2.weight"] != nil)
    }

    @Test("reasoning_effort carries high or none and frame sanitizer strips control markers")
    func reasoningEffortAndFrameSanitizer() {
        let on = Inkling2.additionalContext(thinkingEnabled: true)
        let off = Inkling2.additionalContext(thinkingEnabled: false)
        #expect(on["reasoning_effort"] as? String == "high")
        #expect(off["reasoning_effort"] as? String == "none")

        let engineOn = MLXEngine.additionalContext(
            isInkling2: true, thinking: .inklingFrames, thinkingEnabled: true)
        let engineOff = MLXEngine.additionalContext(
            isInkling2: true, thinking: .inklingFrames, thinkingEnabled: false)
        #expect(engineOn?["reasoning_effort"] as? String == "high")
        #expect(engineOff?["reasoning_effort"] as? String == "high")

        // Streamed piece-by-piece across marker boundaries.
        let raw = "<|message_model|><|content_text|>Hello <|not_a_marker|> world!<|end_message|><|content_model_end_sampling|>"
        var sanitizer = InklingFrameSanitizer()
        var out = ""
        for ch in raw {
            out += sanitizer.consume(String(ch))
        }
        out += sanitizer.finish()
        #expect(out == "Hello <|not_a_marker|> world!")
    }

    @Test("Catalog includes Inkling 2 Mini entries with 1M context, vision, and always-on thinking")
    func catalogEntries() {
        for id in [
            "mlx-community/Inkling-2-Mini-4bit",
            "mlx-streamed/inkling-2-mini-4bit",
            "thinkingmachines/Inkling-2-Mini",
        ] {
            let model = ModelCatalog.model(for: id)
            #expect(model != nil, "missing catalog model \(id)")
            #expect(model?.ctxTokens == 1_048_576)
            #expect(model?.supportsVision == true)
            #expect(model?.supportsThinking == true)
            #expect(model?.supportsThinkingToggle == false)
            #expect(model?.extraEOSTokens.contains("<|content_model_end_sampling|>") == true)
            #expect(model.map { ModelCatalog.familyID(for: $0) } == "inkling2")
        }
        #expect(ModelCatalog.family(for: "inkling2") != nil)
    }
}

// MARK: - Kernels

@Suite("Inkling2 kernels")
struct Inkling2KernelTests {

    @Test("Depth attention matches the softmax over the sources", arguments: [1, 2, 4])
    func depthAttention(queriesPerGroup: Int) {
        MLXRandom.seed(1)
        let blocks = MLXRandom.normal([3, 2, 5, 256])
        let partial = MLXRandom.normal([2, 5, 256])
        let queries = MLXRandom.normal([4, 256])
        let scale = 1 / Float(256).squareRoot()
        let out = inkling2DepthAttention(
            blocks: blocks, partial: partial, queries: queries, scale: scale,
            queriesPerGroup: queriesPerGroup)

        let sources = movedAxis(
            concatenated([blocks, partial.expandedDimensions(axis: 0)]), source: 0,
            destination: -2)
        let weights = softmax(matmul(sources, queries.transposed()) * scale, axis: -2)
        let expected = matmul(weights.swappedAxes(-1, -2), sources)
        #expect(out.shape == [2, 5, 4, 256])
        #expect(maxDifference(out, expected) < 1e-2)
    }

    @Test(
        "Prefill attention matches the reference",
        arguments: [
            (window: 0, cached: 0, keep: 0, evicted: 0),
            (window: 16, cached: 0, keep: 0, evicted: 0),
            (window: 16, cached: 23, keep: 0, evicted: 0),
            (window: 0, cached: 23, keep: 4, evicted: 3),
        ])
    func prefillAttention(window: Int, cached: Int, keep: Int, evicted: Int) {
        MLXRandom.seed(2)
        let (b, hq, hkv, l, d, extent) = (1, 4, 2, 40, 64, 24)
        let s = cached + l
        let q = (MLXRandom.normal([b, hq, l, d]) * 0.3).asType(.bfloat16)
        let k = (MLXRandom.normal([b, hkv, s, d]) * 0.3).asType(.bfloat16)
        let v = MLXRandom.normal([b, hkv, s, d]).asType(.bfloat16)
        let rel = MLXRandom.normal([b, l, hq, extent]).asType(.bfloat16)
        let sinks = MLXRandom.normal([b, l, hq]).asType(.bfloat16)

        let out = inkling2RelativeAttention(
            q: q, k: k, v: v, rel: rel, sinks: sinks, scale: 0.125, window: window,
            keyStart: MLXArray([Int32(-evicted)]), keep: keep, evicted: evicted)
        let queries = MLXArray.arange(l, dtype: .int32) + Int32(s - l + evicted)
        let slots = MLXArray.arange(s, dtype: .int32)
        let positions = which(slots .< Int32(keep), slots, slots + Int32(evicted))
        let distance = queries[0..., .newAxis] - positions[.newAxis]
        let expected = referenceRelativeAttention(
            q: q, k: k, v: v, rel: rel, sinks: sinks, scale: 0.125, window: window,
            distance: distance)
        #expect(maxDifference(out, expected) < 2e-2)
    }

    @Test(
        "Decode attention matches the reference",
        arguments: [
            (keys: 37, ring: false, window: 0, keep: 0, evicted: 0),
            (keys: 37, ring: true, window: 0, keep: 0, evicted: 0),
            (keys: 64, ring: true, window: 48, keep: 0, evicted: 0),
            (keys: 3000, ring: false, window: 0, keep: 0, evicted: 0),
            (keys: 1500, ring: true, window: 1100, keep: 0, evicted: 0),
            (keys: 37, ring: false, window: 0, keep: 4, evicted: 2),
            (keys: 37, ring: true, window: 0, keep: 4, evicted: 2),
            (keys: 3000, ring: true, window: 0, keep: 4, evicted: 900),
        ])
    func decodeAttention(keys s: Int, ring: Bool, window: Int, keep: Int, evicted: Int) {
        MLXRandom.seed(3)
        let (b, hq, hkv, d, extent) = (2, 6, 2, 64, 40)
        let q = (MLXRandom.normal([b, hq, 1, d]) * 0.3).asType(.bfloat16)
        let kBuffer = (MLXRandom.normal([b, hkv, s + 5, d]) * 0.3).asType(.bfloat16)
        let vBuffer = MLXRandom.normal([b, hkv, s + 5, d]).asType(.bfloat16)
        let k = kBuffer[0..., 0..., ..<s, 0...]
        let v = vBuffer[0..., 0..., ..<s, 0...]
        let relative = (MLXRandom.normal([b, hq, 8]) * 0.25).asType(.bfloat16)
        let table = MLXRandom.normal([8, extent]).asType(.bfloat16)
        let rel = matmul(relative.asType(.float32), table.asType(.float32)).expandedDimensions(
            axis: 1)
        let sinks = MLXRandom.normal([b, 1, hq]).asType(.bfloat16)

        let ringIndex = ring ? 11 : nil
        var ringK = k
        var ringV = v
        if let ringIndex {
            let n = s - keep
            let order = MLXArray(
                (0 ..< s).map { Int32($0 < keep ? $0 : keep + ($0 - ringIndex + n) % n) })
            ringK = k.take(order, axis: 2)
            ringV = v.take(order, axis: 2)
        }
        let gate = (MLXRandom.normal([b, 1, hq * d]) * 0.5).asType(.bfloat16)
        let out = inkling2DecodeAttention(
            q: q.squeezed(axis: 2), k: ringK, v: ringV, relative: relative, table: table,
            sinks: sinks, gate: gate, scale: 0.125, ringIndex: ringIndex, window: window,
            keep: keep, evicted: evicted)

        let slots = MLXArray.arange(s, dtype: .int32)
        let distance =
            (MLXArray(Int32(s - 1)) - slots + which(slots .< Int32(keep), Int32(evicted), Int32(0)))[
                .newAxis]
        let expected = referenceRelativeAttention(
            q: q, k: k, v: v, rel: rel, sinks: sinks, scale: 0.125, window: window,
            distance: distance)
        let gated =
            expected.asType(.float32).reshaped(b, hq * d)
            * sigmoid(gate.asType(.float32)).reshaped(b, hq * d)
        #expect(out.shape == [b, hq * d])
        #expect(maxDifference(out, gated) < 1e-2)
    }

    @Test("Expert routing picks the top bias-corrected scores")
    func expertRouting() {
        MLXRandom.seed(4)
        let logits = MLXRandom.normal([3, 2, 128]) * 3
        let bias = MLXRandom.normal([128]) * 0.1
        let (indices, weights) = inkling2ExpertRouting(
            logits: logits, bias: bias, topK: 8, scoreEps: 1e-20, routeScale: 8)

        let scores = sigmoid(logits)
        let order = argSort(-(scores + bias), axis: -1)[.ellipsis, ..<8]
        let selected = takeAlong(scores, order, axis: -1)
        let expected = selected / selected.sum(axis: -1, keepDims: true) * 8
        #expect(
            indices.asType(.int32).asArray(Int32.self) == order.asType(.int32).asArray(Int32.self))
        #expect(maxDifference(weights, expected) < 1e-5)
    }

    @Test(
        "The fused expert kernels match the gathered quantized matmuls",
        arguments: [
            (QuantizationMode.nvfp4, 4, 16), (.mxfp8, 8, 32), (.affine, 4, 64), (.affine, 8, 32),
        ])
    func fusedExperts(mode: QuantizationMode, bits: Int, groupSize: Int) {
        MLXRandom.seed(8)
        let (rows, d, inner, experts, k) = (2, 128, 64, 6, 3)
        func quantize(_ shape: [Int]) -> Inkling2ExpertTensors {
            let dense = (MLXRandom.normal(shape) * 0.1).asType(.bfloat16)
            let (weight, scales, biases) = MLX.quantized(
                dense, groupSize: groupSize, bits: bits, mode: mode)
            let global = mode == .nvfp4 ? MLXRandom.uniform(2000 ..< 3000, [experts]) : nil
            return .init(
                weight: weight, scales: scales, biases: biases, globalScale: global,
                format: .init(
                    mode: mode, bits: bits, groupSize: groupSize, hasGlobalScale: global != nil))
        }
        let gateUp = quantize([experts, 2 * inner, d])
        let down = quantize([experts, d, inner])
        let x = MLXRandom.normal([rows, d]).asType(.bfloat16)
        let indices = MLXArray([0, 4, 2, 5, 1, 4] as [UInt32]).reshaped(rows, k)
        let weights = MLXRandom.uniform(0 ..< 1, [rows, k])
        let residual = MLXRandom.normal([rows, d])
        let state = MLXRandom.normal([rows, 3, d]).asType(.bfloat16)
        let convolution = (MLXRandom.normal([d, 4, 1]) * 0.5).asType(.bfloat16)
        func gathered(_ x: MLXArray, _ experts: Inkling2ExpertTensors) -> MLXArray {
            gatherQuantizedMM(
                x, experts.weight, scales: experts.scales, biases: experts.biases,
                rhsIndices: indices, transpose: true, groupSize: groupSize, bits: bits,
                mode: mode, globalScale: experts.globalScale
            ).squeezed(axis: -2)
        }

        let activations = inkling2ExpertGateUp(
            x, experts: gateUp, indices: indices, softcaps: MLXArray([Float(4), 25]))
        let g = gathered(x.reshaped(rows, 1, 1, d), gateUp).asType(.float32)
        let (gate, up) = (
            g[.ellipsis, .stride(from: 0, by: 2)], g[.ellipsis, .stride(from: 1, by: 2)]
        )
        let expectedActivations = tanh(gate / 4) * 4 * sigmoid(gate) * tanh(up / 25) * 25
        #expect(activations.shape == [rows, k, inner])
        #expect(maxDifference(activations, expectedActivations) < 5e-2)

        let fused = inkling2ExpertDown(
            activations, experts: down, indices: indices, weights: weights, residual: residual,
            state: state, convolutionWeight: convolution)
        let expected = inkling2ExpertEpilogue(
            outputs: gathered(activations.reshaped(rows, k, 1, inner), down), weights: weights,
            residual: residual, state: state, convolutionWeight: convolution)
        #expect(maxDifference(fused.hidden, expected.hidden) < 2e-2)
        #expect(maxDifference(fused.state, expected.state) < 2e-2)
    }

    @Test(
        "The prefill convolution matches the causal convolution of its inputs",
        arguments: [(37, true, false), (2, true, true), (19, false, true), (37, true, true)])
    func rowConvolution(length l: Int, hasState: Bool, experts: Bool) {
        MLXRandom.seed(9)
        let (b, c, k) = (2, 64, 3)
        let state = hasState ? MLXRandom.normal([b, 3, c]).asType(.bfloat16) : nil
        let weight = (MLXRandom.normal([c, 4, 1]) * 0.5).asType(.bfloat16)
        let base = MLXRandom.normal([b, l, c])

        var x = MLXRandom.normal([b, l, c]).asType(.bfloat16)
        var routing: (weights: MLXArray, order: MLXArray?)?
        var input = x
        if experts {
            let outputs = MLXRandom.normal([b * l * k, c]).asType(.bfloat16)
            let weights = MLXRandom.uniform(0 ..< 1, [b, l, k])
            let order = MLXArray((0 ..< (b * l * k)).reversed().map { UInt32($0) })
            routing = (weights, order)
            x = outputs
            input =
                (outputs[order].reshaped(b, l, k, c).asType(.float32)
                * weights.expandedDimensions(axis: -1)).sum(axis: 2).asType(.bfloat16)
        }
        let (out, nextState) = inkling2RowConvolution(
            x, state: state, weight: weight, base: base, outputType: .float32, experts: routing)

        let window = concatenated(
            [state ?? MLXArray.zeros([b, 3, c], dtype: .bfloat16), input], axis: 1)
        let taps = weight.squeezed(axis: -1).transposed().asType(.float32)
        var acc = window[0..., 0 ..< l, 0...].asType(.float32) * taps[0]
        for tap in 1 ..< 4 {
            acc = acc + window[0..., tap ..< (tap + l), 0...].asType(.float32) * taps[tap]
        }
        let y =
            ((acc.asType(.bfloat16).asType(.float32) + input.asType(.float32))
            / Float(5).squareRoot()).asType(.bfloat16)
        #expect(out.shape == [b, l, c])
        #expect(maxDifference(out, base + y.asType(.float32)) < 2e-2)
        #expect(maxDifference(nextState, window[0..., l...]) < 2e-2)
    }

    @Test("The prefill convolution adds the dynamic taps of each head")
    func dynamicRowConvolution() {
        MLXRandom.seed(10)
        let (b, l, c, heads) = (2, 21, 64, 4)
        let x = MLXRandom.normal([b, l, c]).asType(.bfloat16)
        let state = MLXRandom.normal([b, 3, c]).asType(.bfloat16)
        let staticTaps = (MLXRandom.normal([4, c]) * 0.5).asType(.bfloat16)
        let taps = (MLXRandom.normal([b, l, heads * 4 + 3]) * 0.5).asType(.bfloat16)[
            .ellipsis, ..<(heads * 4)]
        let (out, nextState) = inkling2RowConvolution(
            x, state: state, weight: staticTaps, outputType: .bfloat16, dynamicTaps: taps)

        let window = concatenated([state, x], axis: 1).asType(.float32)
        let perChannel = repeated(
            taps.asType(.float32).reshaped(b, l, heads, 1, 4), count: c / heads, axis: 3
        )
        .reshaped(b, l, c, 4)
        var acc = MLXArray.zeros([b, l, c])
        for lag in 0 ..< 4 {
            let weights = staticTaps[lag].asType(.float32) + perChannel[.ellipsis, lag]
            acc = acc + window[0..., (3 - lag) ..< (3 - lag + l), 0...] * weights
        }
        let expected = ((acc + x.asType(.float32)) / 3).asType(.bfloat16)
        #expect(maxDifference(out, expected) < 2e-2)
        #expect(maxDifference(nextState, window[0..., l...]) == 0)
    }

    @Test(
        "The prefill prologue clips, scales and normalizes the attention inputs",
        arguments: [false, true])
    func prefillPrologue(full: Bool) {
        MLXRandom.seed(11)
        let (b, l, hq, hkv, dh, dr) = (2, 5, 4, 2, 64, 16)
        let projection = (MLXRandom.normal([b, l, hq * (dh + dr) + 8]) * 4).asType(.bfloat16)
        let q = projection[.ellipsis, ..<(hq * dh)]
        let r = projection[.ellipsis, (hq * dh) ..< (hq * (dh + dr))]
        let k = (MLXRandom.normal([b, l, hkv * dh]) * 4).asType(.bfloat16)
        let v = (MLXRandom.normal([b, l, hkv * dh]) * 4).asType(.bfloat16)
        let qScale = MLXRandom.uniform(0.5 ..< 1.5, [hq, dh])
        let rScale = MLXRandom.uniform(0.5 ..< 1.5, [hq, dr])
        let tau = full ? MLXRandom.uniform(1 ..< 1.5, [l]) : nil
        let out = inkling2PrefillPrologue(
            q: q, r: r, k: k, v: v, qScale: qScale, rScale: rScale, tau: tau, heads: hq,
            kvHeads: hkv, headDim: dh, relativeDim: dr, qkThreshold: 20, relativeThreshold: 8)

        func clipped(_ x: MLXArray, _ scale: MLXArray?, _ threshold: Float) -> MLXArray {
            var states = x.asType(.float32)
            if let scale { states = states * scale }
            let norm = sqrt(states.square().sum(axis: -1, keepDims: true))
            var y = (states * minimum(threshold / (norm + 1e-6), 1)).asType(.bfloat16)
            if let tau { y = (y.asType(.float32) * tau.reshaped(1, l, 1, 1)).asType(.bfloat16) }
            return y
        }
        let keys = k.reshaped(b, l, hkv, dh).asType(.float32)
        let norm = sqrt(keys.square().sum(axis: -1, keepDims: true))
        let values = v.reshaped(b, l, hkv, dh).asType(.float32)
        #expect(maxDifference(out[0], clipped(q.reshaped(b, l, hq, dh), qScale, 20)) < 5e-2)
        #expect(maxDifference(out[1], clipped(r.reshaped(b, l, hq, dr), rScale, 8)) < 5e-2)
        #expect(maxDifference(out[2], keys * minimum(20 / (norm + 1e-6), 1)) < 5e-2)
        #expect(
            maxDifference(
                out[3], values * rsqrt(values.square().mean(axis: -1, keepDims: true) + 1e-6))
                < 5e-2)
    }

    @Test("Message controllers match the gated blend", arguments: [(3, false), (1, true)])
    func messageControllers(streams: Int, f32Copy: Bool) {
        MLXRandom.seed(6)
        let (rows, d, rank, first) = (2, 320, 8, 1)
        let inputs = MLXRandom.normal([streams, rows, d])
        let messages = MLXRandom.normal([rows, 6, d])
        let down = (MLXRandom.normal([streams, rank, 2 * d]) * 0.1).asType(.bfloat16)
        let forget = (MLXRandom.normal([streams, rank, d]) * 0.3).asType(.bfloat16)
        let gain = (MLXRandom.normal([streams, rank, d]) * 0.3).asType(.bfloat16)
        let normWeight = MLXRandom.uniform(0.5 ..< 1.5, [streams, d]).asType(.bfloat16)
        let (out, outF32) = inkling2MessageControllers(
            inputs: inputs, messages: messages, first: first, down: down, forget: forget,
            gain: gain, normWeight: normWeight, eps: 1e-6, f32Copy: f32Copy)

        func rmsNormed(_ x: MLXArray) -> MLXArray {
            x * rsqrt(x.square().mean(axis: -1, keepDims: true) + 1e-6)
        }
        let message = messages[0..., first ..< (first + streams), 0...].transposed(1, 0, 2)
        let pair = concatenated([rmsNormed(inputs), rmsNormed(message)], axis: -1)
            .asType(.bfloat16).asType(.float32)
        let h = matmul(pair, down.asType(.float32).transposed(0, 2, 1))
            .asType(.bfloat16).asType(.float32)
        let blend =
            (1 - tanh(matmul(h, forget.asType(.float32)))) * inputs
            + tanh(matmul(h, gain.asType(.float32))) * message
        let expected = normWeight.asType(.float32).expandedDimensions(axis: 1) * rmsNormed(blend)

        #expect(out.dtype == .bfloat16)
        #expect(maxDifference(out, expected) < 2e-2)
        #expect((outF32 != nil) == f32Copy)
        if let outF32 {
            #expect(maxDifference(outF32, out) == 0)
        }
    }

    @Test("The depth mix matches the softmax over the blocks and the residual")
    func depthMix() {
        MLXRandom.seed(7)
        let (rows, d, k, q) = (2, 320, 3, 3)
        let blocks = MLXRandom.normal([k, rows, d])
        let base = MLXRandom.normal([rows, d]).asType(.bfloat16)
        let x = MLXRandom.normal([rows, d]).asType(.bfloat16)
        let state = MLXRandom.normal([rows, 3, d]).asType(.bfloat16)
        let weight = (MLXRandom.normal([d, 4, 1]) * 0.5).asType(.bfloat16)
        let queries = MLXRandom.normal([q, d]) * 0.2
        let normWeight = MLXRandom.uniform(0.5 ..< 1.5, [q, d]).asType(.bfloat16)
        let mix = inkling2DepthMix(
            blocks: blocks, base: base, convolution: (x, state, weight), queries: queries,
            normWeight: normWeight, f32Copy: true, eps: 1e-6)

        let window = concatenated([state, x.expandedDimensions(axis: 1)], axis: 1)
            .asType(.float32)
        let taps = weight.asType(.float32).reshaped(d, 4).transposed()
        let convolved = (window * taps).sum(axis: 1).asType(.bfloat16).asType(.float32)
        let residual =
            base.asType(.float32)
            + ((convolved + x.asType(.float32)) / Float(5).squareRoot())
            .asType(.bfloat16).asType(.float32)
        let normed = residual * rsqrt(residual.square().mean(axis: -1, keepDims: true) + 1e-6)
        let sources = concatenated([blocks, normed.expandedDimensions(axis: 0)])
        let weights = softmax(
            matmul(sources, queries.transposed()) / Float(d).squareRoot(), axis: 0)
        let readout =
            (weights.expandedDimensions(axis: -1)
            * sources.expandedDimensions(axis: 2)).sum(axis: 0)
        let expected =
            normWeight.asType(.float32)
            * readout * rsqrt(readout.square().mean(axis: -1, keepDims: true) + 1e-6)

        #expect(maxDifference(mix.residual, residual) < 1e-4)
        #expect(maxDifference(mix.normed, normed) < 1e-4)
        #expect(maxDifference(mix.readout, expected) < 2e-2)
        #expect(maxDifference(mix.readoutF32!, mix.readout) == 0)
        let expectedState = concatenated(
            [state[0..., 1...], x.expandedDimensions(axis: 1)], axis: 1)
        #expect(maxDifference(mix.state!, expectedState) == 0)
    }
}

// MARK: - Model & Sliding Cache

@Suite("Inkling2 model")
struct Inkling2ModelTests {

    static let tinyJSON = """
        {
          "model_type": "inkling2_loop",
          "text_config": {
            "hidden_size": 64, "num_hidden_layers": 6, "vocab_size": 50,
            "unpadded_vocab_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 32, "sliding_window_size": 8, "local_layer_ids": [0, 1, 2, 4],
            "d_rel": 8, "rel_extent": 16, "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-06, "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8.0,
            "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 2, "n_routed_experts": 4,
            "num_experts_per_tok": 2, "intermediate_size": 32, "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0, "route_scale": 2.0, "final_logit_softcapping": 30.0,
            "loop_config": {
              "core_start": 2, "core_end": 4, "num_passes": 2, "gate_rank": 8,
              "is_varloop": false
            }
          }
        }
        """

    static func makeModel(bfloat16: Bool = false) throws -> Inkling2 {
        let config = try JSONDecoder().decode(
            Inkling2Configuration.self, from: Data(tinyJSON.utf8))
        let model = Inkling2(config)
        MLXRandom.seed(5)
        var parameters = [String: MLXArray]()
        for (key, value) in model.parameters().flattened() where value.dtype == .float32 {
            parameters[key] = MLXRandom.normal(value.shape) * 0.3
        }
        for (key, value) in model.parameters().flattened() where key.hasSuffix(".global_scale") {
            parameters[key] = MLXArray.ones(value.shape)
        }
        for (key, value) in model.parameters().flattened()
        where key.hasSuffix("_proj.weight") && value.dtype == .uint32 {
            let prefix = String(key.dropLast(".weight".count))
            let (rows, packed) = (value.dim(1), value.dim(2))
            let dense = MLXRandom.normal([value.dim(0), rows, packed * 8]) * 0.3
            let (weight, scales, _) = MLX.quantized(
                dense, groupSize: 16, bits: 4, mode: .nvfp4)
            parameters[key] = weight
            parameters["\(prefix).scales"] = scales
        }
        if bfloat16 {
            let float32Suffixes = [
                ".query", "mlp.gate.weight", "e_score_correction_bias",
                "forget_gate_proj.weight", "message_gate_proj.weight", ".global_scale",
            ]
            for (key, value) in parameters
            where value.dtype == .float32 && !float32Suffixes.contains(where: key.hasSuffix) {
                parameters[key] = value.asType(.bfloat16)
            }
        }
        try model.update(
            parameters: ModuleParameters.unflattened(parameters), verify: .noUnusedKeys)
        if bfloat16 {
            try model.prepare()
        }
        eval(model)
        return model
    }

    @Test("Pinned keys keep their positions in the reference distances")
    func pinnedDistances() {
        let masks = Inkling2DecodeMasks()
        let ring = masks.distance(keys: 7, queries: 1, ringIndex: 6, keep: 2, evicted: 10)
        #expect(ring.asArray(Int32.self) == [16, 15, 3, 2, 1, 0, 4])
        let prefill = masks.distance(keys: 6, queries: 2, ringIndex: nil, keep: 2, evicted: 3)
        #expect(prefill.asArray(Int32.self) == [7, 6, 2, 1, 0, -1, 8, 7, 3, 2, 1, 0])
    }

    @Test("Local Inkling2SlidingKVCache tracks ringWriteIndex identically to RotatingKVCache")
    func slidingCacheRingWriteIndex() {
        let local = Inkling2SlidingKVCache(maxSize: 6, keep: 2, step: 4)
        let rotating = RotatingKVCache(maxSize: 6, keep: 2, step: 4)
        for step in 0 ..< 12 {
            let k = MLXArray.full([1, 1, 1, 4], values: MLXArray(Float(step)))
            let v = k
            let (lk, _) = local.update(keys: k, values: v)
            let (rk, _) = rotating.update(keys: k, values: v)
            let localState = inkling2RingState(for: local, keyLength: lk.dim(2))
            let rotState = inkling2RingState(for: rotating, keyLength: rk.dim(2))
            #expect(localState.ringIndex == rotState.ringIndex)
            #expect(localState.keep == rotState.keep)
            #expect(maxDifference(lk, rk) == 0)
        }
    }

    @Test(
        "A pinned prefix reads the same through the ring and the rewind view",
        arguments: [false, true])
    func pinnedPrefix(bfloat16: Bool) throws {
        let model = try Self.makeModel(bfloat16: bfloat16)
        let tokens = MLXArray((0 ..< 30).map { Int32(($0 * 7 + 3) % 48) }).reshaped(1, -1)
        let capacity = try KVCacheConfiguration.Capacity(maxTokens: 12, preservedPrefixTokens: 4)
        func decode(rewind: Int?) throws -> [MLXArray] {
            let configuration = KVCacheConfiguration(
                capacity: capacity, rewind: try rewind.map { try .init(maxTokens: $0) })
            let cache = try model.newCache(parameters: .init(kvCache: configuration))
            _ = model(tokens[0..., ..<5], cache: cache)
            return (5 ..< 30).map { position in
                let step = model(tokens[0..., position ..< (position + 1)], cache: cache)
                eval(step)
                return step
            }
        }
        let ring = try decode(rewind: nil)
        let view = try decode(rewind: 8)
        let tolerance: Float = bfloat16 ? 0.1 : 1e-2
        for (position, (a, b)) in zip(5..., zip(ring, view)) {
            #expect(maxDifference(a, b) < tolerance, "position \(position)")
        }
    }

    @Test("Decoding through the caches matches the full-sequence forward")
    func incrementalDecoding() throws {
        let model = try Self.makeModel()
        let tokens = MLXArray((0 ..< 21).map { Int32(($0 * 7 + 3) % 48) }).reshaped(1, -1)
        let full = model(tokens, cache: nil)
        #expect(full.shape == [1, 21, 48])

        let cache = try model.newCache(parameters: nil)
        #expect(cache.count == 8)
        let prefill = model(tokens[0..., ..<5], cache: cache)
        #expect(maxDifference(prefill, full[0..., ..<5]) < 1e-2)
        for position in 5 ..< 21 {
            let step = model(tokens[0..., position ..< (position + 1)], cache: cache)
            #expect(
                maxDifference(step, full[0..., position ..< (position + 1)]) < 1e-2,
                "position \(position)")
        }
    }

    @Test("The bfloat16 decode kernels match the full-sequence forward")
    func bfloat16DecodeKernels() throws {
        let model = try Self.makeModel(bfloat16: true)
        let tokens = MLXArray((0 ..< 21).map { Int32(($0 * 7 + 3) % 48) }).reshaped(1, -1)
        let full = model(tokens, cache: nil)

        let cache = try model.newCache(parameters: nil)
        _ = model(tokens[0..., ..<5], cache: cache)
        var worst: Float = 0
        for position in 5 ..< 21 {
            let step = model(tokens[0..., position ..< (position + 1)], cache: cache)
            worst = max(worst, maxDifference(step, full[0..., position ..< (position + 1)]))
        }
        #expect(worst < 0.1)
    }

    @Test("Chunked prefill matches one prefill")
    func chunkedPrefill() throws {
        let model = try Self.makeModel()
        let tokens = MLXArray((0 ..< 30).map { Int32(($0 * 5 + 1) % 48) }).reshaped(1, -1)
        let whole = try model.prepare(
            LMInput(tokens: tokens), cache: model.newCache(parameters: nil), state: nil,
            prefill: .init())
        let chunked = try model.prepare(
            LMInput(tokens: tokens), cache: model.newCache(parameters: nil), state: nil,
            prefill: .init(stepSize: 7))
        guard case .logits(let a) = whole, case .logits(let b) = chunked else {
            Issue.record("prepare did not return logits")
            return
        }
        #expect(a.logits.shape == [1, 1, 48])
        #expect(maxDifference(a.logits, b.logits) < 1e-2)
    }

    @Test("Opt-in real-weight smoke test gated by INKLING2_MODEL_DIR")
    func realWeightSmokeTest() async throws {
        guard let dirPath = ProcessInfo.processInfo.environment["INKLING2_MODEL_DIR"],
              !dirPath.isEmpty
        else { return }
        let directory = URL(fileURLWithPath: dirPath, isDirectory: true)
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        #expect(ModelSupport.unsupportedReason(configJSON: configData) == nil)
        await Inkling2.register()
        let configuration = ModelConfiguration(
            directory: directory, extraEOSTokens: ["<|content_model_end_sampling|>"])
        let container = try await VLMModelFactory.shared.loadContainer(
            from: HubDownloader(ModelStore.hubClient),
            using: HuggingFaceTokenizerLoader(),
            configuration: configuration)
        let isInkling = await container.perform { $0.model is Inkling2 }
        #expect(isInkling)
    }
}

// MARK: - Processor

@Suite("Inkling2 processor")
struct Inkling2ProcessorTests {

    @Test("Images are padded to whole blocks and patchified block by block")
    func patches() throws {
        let processor = Inkling2Processor(
            Inkling2ProcessorConfiguration(), tokenizer: InklingTestTokenizer())
        let image = CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(
            to: CGRect(x: 0, y: 0, width: 50, height: 30))
        let (patches, grid) = try processor.preprocess(image: image, processing: nil)
        #expect(grid.h == 2 && grid.w == 4)
        #expect(patches.shape == [8, 1200])
        let padding = (Float(-1) / 255 - 0.5) / 0.5
        #expect(abs(patches[7].min().item(Float.self) - padding) < 1e-6)
        #expect(abs(patches[0, 0].item(Float.self) - 1) < 1e-3)
        #expect(abs(patches[0, 1].item(Float.self) + 1) < 1e-3)
    }
}
