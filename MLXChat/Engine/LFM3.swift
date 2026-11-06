import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Port of Liquid AI's `modeling_lfm3.py` (transformers 5.19). LFM3 is LFM2 with a third layer
// type (`sliding_attention`), sandwich norms, gated attention, an explicit head size and a
// width-7 short convolution. The linked mlx-swift-lm has no LFM3, so the app registers it.

// MARK: - Configuration

struct LFM3Configuration: Codable, Sendable {
    var vocabularySize: Int
    var hiddenSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var intermediateSize: Int
    var normEps: Float
    var convBias: Bool
    var convLCache: Int
    var layerTypes: [String]
    var ropeParameters: [String: [String: StringOrNumber]]
    var slidingWindow: Int
    var useSandwichNorm: Bool
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case slidingWindow = "sliding_window"
        case useSandwichNorm = "use_sandwich_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? hiddenSize / attentionHeads
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        normEps = try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        convBias = try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        convLCache = try container.decodeIfPresent(Int.self, forKey: .convLCache) ?? 7
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        ropeParameters = try container.decodeIfPresent(
            [String: [String: StringOrNumber]].self, forKey: .ropeParameters) ?? [:]
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 256
        useSandwichNorm = try container.decodeIfPresent(Bool.self, forKey: .useSandwichNorm) ?? false
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        guard layerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription: "layer_types has \(layerTypes.count) entries for \(hiddenLayers) layers")
        }
    }

    /// Reads the text backbone: `text_config` for LFM3-Omni, the top level for plain LFM3.
    static func decode(configJSON data: Data) throws -> LFM3Configuration {
        struct Wrapper: Decodable {
            let textConfig: LFM3Configuration?
            enum CodingKeys: String, CodingKey { case textConfig = "text_config" }
        }
        let decoder = JSONDecoder.json5()
        if let text = try decoder.decode(Wrapper.self, from: data).textConfig {
            return text
        }
        return try decoder.decode(LFM3Configuration.self, from: data)
    }

    func isAttention(_ layer: Int) -> Bool { layerTypes[layer] != "conv" }
    func isSliding(_ layer: Int) -> Bool { layerTypes[layer] == "sliding_attention" }
}

// MARK: - Layers

/// LFM3 rotary embedding for one layer type. Partial RoPE keeps the full head's frequency
/// progression and leaves the tail unrotated, which is what `ProportionalRoPE` computes.
func makeLFM3RoPE(headDim: Int, parameters: [String: StringOrNumber]?) -> RoPELayer {
    let base = parameters?["rope_theta"]?.asFloat() ?? 10_000
    let partial = parameters?["partial_rotary_factor"]?.asFloat() ?? 1
    if partial < 1 {
        return initializeRope(
            dims: headDim, base: base, traditional: false,
            scalingConfig: [
                "rope_type": .string("proportional"), "partial_rotary_factor": .float(partial),
            ],
            maxPositionEmbeddings: nil)
    }
    return RoPE(dimensions: headDim, traditional: false, base: base)
}

final class LFM3Attention: Module {
    let heads: Int
    let kvHeads: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ args: LFM3Configuration, layer: Int) {
        heads = args.attentionHeads
        kvHeads = args.kvHeads
        scale = pow(Float(args.headDim), -0.5)
        rope = makeLFM3RoPE(
            headDim: args.headDim, parameters: args.ropeParameters[args.layerTypes[layer]])

        let dim = args.hiddenSize
        _qProj.wrappedValue = Linear(dim, heads * args.headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, kvHeads * args.headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, kvHeads * args.headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * args.headDim, dim, bias: false)
        _gateProj.wrappedValue = Linear(dim, heads, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.normEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(qProj(x).reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)

        // One sigmoid gate per head, read from the layer input.
        let gate = sigmoid(gateProj(x)).expandedDimensions(axis: -1)
        return oProj((output * gate).reshaped(B, L, -1))
    }
}

final class LFM3ShortConv: Module {
    let hiddenSize: Int
    let lCache: Int

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: LFM3Configuration) {
        hiddenSize = args.hiddenSize
        lCache = args.convLCache
        _conv.wrappedValue = Conv1d(
            inputChannels: args.hiddenSize, outputChannels: args.hiddenSize,
            kernelSize: lCache, groups: args.hiddenSize, bias: args.convBias)
        _inProj.wrappedValue = Linear(args.hiddenSize, 3 * args.hiddenSize, bias: args.convBias)
        _outProj.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: args.convBias)
    }

    func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let parts = inProj(x).split(parts: 3, axis: -1)
        let (b, c, h) = (parts[0], parts[1], parts[2])
        var bx = b * h

        let state = cache?[0]
            ?? MLXArray.zeros([bx.dim(0), lCache - 1, hiddenSize], dtype: bx.dtype)
        bx = concatenated([state, bx], axis: -2)
        if let cache {
            cache[0] = contiguous(bx[0..., (bx.dim(1) - (lCache - 1))..., 0...])
            cache.advance(x.dim(1))
        }

        return outProj(c * conv(bx))
    }
}

final class LFM3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hidden: Int) {
        _w1.wrappedValue = Linear(dim, hidden, bias: false)
        _w2.wrappedValue = Linear(hidden, dim, bias: false)
        _w3.wrappedValue = Linear(dim, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

final class LFM3DecoderLayer: Module {
    let isAttention: Bool

    @ModuleInfo(key: "self_attn") var attention: LFM3Attention?
    @ModuleInfo(key: "conv") var conv: LFM3ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm?

    init(_ args: LFM3Configuration, layer: Int) {
        isAttention = args.isAttention(layer)
        if isAttention {
            _attention.wrappedValue = LFM3Attention(args, layer: layer)
        } else {
            _conv.wrappedValue = LFM3ShortConv(args)
        }
        _feedForward.wrappedValue = LFM3MLP(dim: args.hiddenSize, hidden: args.intermediateSize)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        if args.useSandwichNorm {
            _postOperatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
            _postFFNNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = isAttention
            ? attention!(operatorNorm(x), mask: mask, cache: cache)
            : conv!(operatorNorm(x), cache: cache as? MambaCache)
        if let postOperatorNorm { r = postOperatorNorm(r) }
        let h = x + r

        var f = feedForward(ffnNorm(h))
        if let postFFNNorm { f = postFFNNorm(f) }
        return h + f
    }
}

final class LFM3ModelInner: Module {
    let args: LFM3Configuration
    let layers: [LFM3DecoderLayer]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(_ args: LFM3Configuration) {
        self.args = args
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        layers = (0 ..< args.hiddenLayers).map { LFM3DecoderLayer(args, layer: $0) }
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        // One mask per attention type, built from a cache of that type.
        let fullIndex = args.layerTypes.firstIndex(of: "full_attention")
        let slidingIndex = args.layerTypes.firstIndex(of: "sliding_attention")
        let fullMask = createAttentionMask(h: h, cache: fullIndex.flatMap { cache?[$0] })
        let slidingMask = createAttentionMask(
            h: h, cache: slidingIndex.flatMap { cache?[$0] }, windowSize: args.slidingWindow)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: args.isSliding(i) ? slidingMask : fullMask, cache: cache?[i])
        }
        // LFM names its final norm `embedding_norm`.
        return embeddingNorm(h)
    }
}

final class LFM3Model: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]
    let configuration: LFM3Configuration

    let model: LFM3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ args: LFM3Configuration) {
        configuration = args
        vocabularySize = args.vocabularySize
        kvHeads = (0 ..< args.hiddenLayers).map { args.isAttention($0) ? args.kvHeads : 0 }
        model = LFM3ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead { return lmHead(out) }
        return model.embedTokens.asLinear(out)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, param) in weights {
            if configuration.tieWordEmbeddings, key.hasPrefix("lm_head.") { continue }
            var value = param
            // PyTorch keeps depthwise conv weights as [out, 1, kernel]; MLX wants [out, kernel, 1].
            if key.hasSuffix("conv.conv.weight"), param.ndim == 3, param.dim(2) > param.dim(1) {
                value = param.transposed(0, 2, 1)
            }
            sanitized[key] = value
        }
        return sanitized
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { layer -> KVCache in
            if configuration.isSliding(layer) {
                RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
            } else if configuration.isAttention(layer) {
                KVCacheSimple()
            } else {
                MambaCache()
            }
        }
    }

    var loraLayers: [Module] { model.layers }
}

/// An LFM3 text backbone kept under `language_model`, as Liquid's MLX checkpoints store it
/// (LFM3 and LFM3-Omni alike). The module paths must match the checkpoint's: per-layer
/// quantization (the 4-bit Omni build's 6-bit layers) is looked up by those exact names.
final class LFM3NestedModel: Module, LLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") var languageModel: LFM3Model

    var vocabularySize: Int { languageModel.vocabularySize }
    var kvHeads: [Int] { languageModel.kvHeads }

    init(_ args: LFM3Configuration) {
        _languageModel.wrappedValue = LFM3Model(args)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    /// Keeps only the text backbone; LFM3-Omni's vision, audio and speech-decoder weights next
    /// to it aren't used by text chat. Conversions without the wrapper load as they are.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let prefix = "language_model."
        let nested = weights.keys.contains { $0.hasPrefix(prefix) }
        let backbone = nested
            ? Dictionary(uniqueKeysWithValues: weights.compactMap { key, value in
                key.hasPrefix(prefix) ? (String(key.dropFirst(prefix.count)), value) : nil
            })
            : weights
        return Dictionary(uniqueKeysWithValues: languageModel.sanitize(weights: backbone).map {
            (prefix + $0.key, $0.value)
        })
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    var loraLayers: [Module] { languageModel.loraLayers }
}

// MARK: - Registration

/// Text chat for LFM3 checkpoints. LFM3-Omni's vision, audio input and speech output need
/// their own encoders and decoders, so the app loads only its language backbone.
enum LFM3 {
    /// Adds `lfm3` and `lfm3_omni` to the LLM factory. Idempotent; call before any load.
    static func register() async {
        for type in ["lfm3", "lfm3_omni"] {
            await LLMTypeRegistry.shared.registerModelType(type) { data in
                LFM3NestedModel(try LFM3Configuration.decode(configJSON: data))
            }
        }
    }
}
