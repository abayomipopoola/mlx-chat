import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Port of XHToken's `modeling_spark.py` (Spark-X2.5 and X2.6, `spark2_5`). A Llama-style decoder
// with a fused QKV projection, a sigmoid output gate per head, and sliding-window layers between
// full-attention layers that rotate only part of each head. The linked mlx-swift-lm has no Spark,
// so the app registers it.

// MARK: - Configuration

struct Spark25Configuration: Codable, Sendable {
    var vocabularySize: Int
    var hiddenSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var intermediateSize: Int
    var rmsNormEps: Float
    var attentionBias: Bool
    var mlpBias: Bool
    var headwiseGate: Bool
    var gateActivation: String
    var layerTypes: [String]
    var ropeParameters: [String: [String: StringOrNumber]]
    var slidingWindow: Int
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case headwiseGate = "headwise_attn_output_gate"
        case gateActivation = "gate_attn_act_mode"
        case hiddenActivation = "hidden_act"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case slidingWindow = "sliding_window"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? hiddenSize / attentionHeads
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        headwiseGate = try container.decodeIfPresent(Bool.self, forKey: .headwiseGate) ?? false
        gateActivation =
            try container.decodeIfPresent(String.self, forKey: .gateActivation) ?? "sigmoid"
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: hiddenLayers)
        ropeParameters = try container.decodeIfPresent(
            [String: [String: StringOrNumber]].self, forKey: .ropeParameters) ?? [:]
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        // The reference refuses anything else, so a config that differs is not this model.
        let activation = try container.decodeIfPresent(String.self, forKey: .hiddenActivation)
        guard activation == nil || activation == "gelu" else {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenActivation, in: container,
                debugDescription: "hidden_act \(activation ?? "") isn't supported")
        }
        guard ["sigmoid", "silu"].contains(gateActivation) else {
            throw DecodingError.dataCorruptedError(
                forKey: .gateActivation, in: container,
                debugDescription: "gate_attn_act_mode \(gateActivation) isn't supported")
        }
        guard layerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription: "layer_types has \(layerTypes.count) entries for \(hiddenLayers) layers")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(attentionBias, forKey: .attentionBias)
        try container.encode(mlpBias, forKey: .mlpBias)
        try container.encode(headwiseGate, forKey: .headwiseGate)
        try container.encode(gateActivation, forKey: .gateActivation)
        try container.encode("gelu", forKey: .hiddenActivation)
        try container.encode(layerTypes, forKey: .layerTypes)
        try container.encode(ropeParameters, forKey: .ropeParameters)
        try container.encode(slidingWindow, forKey: .slidingWindow)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
    }

    func isSliding(_ layer: Int) -> Bool { layerTypes[layer] == "sliding_attention" }
}

// MARK: - Layers

/// Spark rotates the first `partial_rotary_factor * head_dim` dimensions of each head with the
/// frequencies of that smaller width, and passes the rest through: plain partial RoPE.
func makeSpark25RoPE(headDim: Int, parameters: [String: StringOrNumber]?) -> RoPE {
    let base = parameters?["rope_theta"]?.asFloat() ?? 10_000
    let partial = parameters?["partial_rotary_factor"]?.asFloat() ?? 1
    return RoPE(dimensions: Int(Float(headDim) * partial), traditional: false, base: base)
}

final class Spark25Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let gateActivation: String
    let rope: RoPE

    @ModuleInfo(key: "q_k_v_proj") var qkvProj: Linear
    @ModuleInfo(key: "g_proj") var gateProj: Linear?
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Spark25Configuration, layer: Int) {
        heads = args.attentionHeads
        kvHeads = args.kvHeads
        headDim = args.headDim
        scale = 1 / Float(args.headDim).squareRoot()
        gateActivation = args.gateActivation
        rope = makeSpark25RoPE(
            headDim: args.headDim, parameters: args.ropeParameters[args.layerTypes[layer]])

        let dim = args.hiddenSize
        _qkvProj.wrappedValue = Linear(
            dim, (heads + 2 * kvHeads) * headDim, bias: args.attentionBias)
        if args.headwiseGate {
            _gateProj.wrappedValue = Linear(dim, heads, bias: args.attentionBias)
        }
        _outProj.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qkv = qkvProj(x).split(
            indices: [heads * headDim, (heads + kvHeads) * headDim], axis: -1)
        var queries = qkv[0].reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var keys = qkv[1].reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = qkv[2].reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        var output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)

        // One gate per head, read from the layer input, applied in float32 like the reference.
        if let gateProj {
            let score = gateProj(x).asType(.float32)
            let gate = gateActivation == "silu" ? silu(score) : sigmoid(score)
            output = output * gate.asType(output.dtype).expandedDimensions(axis: -1)
        }
        return outProj(output.reshaped(B, L, -1))
    }
}

final class Spark25MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: Spark25Configuration) {
        _gateProj.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        _upProj.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        _downProj.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
    }

    /// `gelu` here is the exact (erf) GELU, as transformers' ACT2FN["gelu"].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(gelu(gateProj(x)) * upProj(x))
    }
}

final class Spark25DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Spark25Attention
    @ModuleInfo(key: "mlp") var mlp: Spark25MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: Spark25Configuration, layer: Int) {
        _attention.wrappedValue = Spark25Attention(args, layer: layer)
        _mlp.wrappedValue = Spark25MLP(args)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    /// `h` is the float32 residual stream; each branch runs in the weights' dtype.
    func callAsFunction(
        _ h: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let dtype = inputLayerNorm.weight.dtype
        let r = attention(inputLayerNorm(h).asType(dtype), mask: mask, cache: cache)
        let h = h + r.asType(.float32)
        return h + mlp(postAttentionLayerNorm(h).asType(dtype)).asType(.float32)
    }
}

final class Spark25ModelInner: Module {
    let args: Spark25Configuration
    let layers: [Spark25DecoderLayer]

    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: Spark25Configuration) {
        self.args = args
        _embedding.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        layers = (0 ..< args.hiddenLayers).map { Spark25DecoderLayer(args, layer: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let embedded = embedding(inputs)
        // The reference keeps the residual stream in float32 across all layers.
        var h = embedded.asType(.float32)

        // One mask per attention type, built from a cache of that type.
        let fullIndex = args.layerTypes.firstIndex(of: "full_attention")
        let slidingIndex = args.layerTypes.firstIndex(of: "sliding_attention")
        let fullMask = createAttentionMask(h: h, cache: fullIndex.flatMap { cache?[$0] })
        let slidingMask = createAttentionMask(
            h: h, cache: slidingIndex.flatMap { cache?[$0] }, windowSize: args.slidingWindow)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: args.isSliding(i) ? slidingMask : fullMask, cache: cache?[i])
        }
        return norm(h).asType(embedded.dtype)
    }
}

final class Spark25Model: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]
    let configuration: Spark25Configuration

    let model: Spark25ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ args: Spark25Configuration) {
        configuration = args
        vocabularySize = args.vocabularySize
        kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        model = Spark25ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead { return lmHead(out) }
        return model.embedding.asLinear(out)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in
            !(configuration.tieWordEmbeddings && key.hasPrefix("lm_head."))
                && !key.contains("rotary_emb.inv_freq")
        }
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { layer -> KVCache in
            configuration.isSliding(layer)
                ? RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
                : KVCacheSimple()
        }
    }

    var loraLayers: [Module] { model.layers }
}

// MARK: - Registration

enum Spark25 {
    static let modelType = "spark2_5"

    /// Adds `spark2_5` to the LLM factory. Idempotent; call before any load.
    static func register() async {
        await LLMTypeRegistry.shared.registerModelType(modelType) { data in
            Spark25Model(try JSONDecoder.json5().decode(Spark25Configuration.self, from: data))
        }
    }
}
