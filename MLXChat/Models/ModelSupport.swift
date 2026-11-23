import Foundation

/// Whether a local model can run in this build of MLX Chat.
///
/// Rejects configs the bundled MLX runtime cannot load: unknown `model_type`
/// (would throw cleanly) and out-of-range affine quantization (would SIGTRAP
/// via the mlx C error handler). Callers must check before any load attempt.
enum ModelSupport {
    /// `model_type` keys the bundled MLX runtime can instantiate (mirror of
    /// LLMTypeRegistry ∪ VLMTypeRegistry at the linked mlx-swift-lm version;
    /// ModelSupportTests pins this list against the real registries).
    static let supportedModelTypes: Set<String> = [
        "mistral",
        "mixtral",
        "llama",
        "phi",
        "phi3",
        "phimoe",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "gemma3n",
        "gemma4",
        "gemma4_unified",
        "gemma4_text",
        "qwen2",
        "qwen3",
        "qwen3_moe",
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_text",
        // Registered by the app (PrismHadamardQwen35), not by mlx-swift-lm.
        "prism_hadamard_qwen35",
        // Registered by the app (LFM3); lfm3_omni loads its text backbone only.
        "lfm3",
        "lfm3_omni",
        // Registered by the app (Spark25): XHToken's Spark-X2.5 and X2.6.
        "spark2_5",
        // Registered by the app (Inkling2): Thinking Machines' Inkling 2 Mini.
        "inkling2",
        "inkling2_loop",
        "minicpm",
        "starcoder2",
        "cohere",
        "openelm",
        "internlm2",
        "deepseek_v3",
        "granite",
        "granitemoehybrid",
        "mimo",
        "mimo_v2_flash",
        "minimax",
        "glm4",
        "glm4_moe",
        "glm4_moe_lite",
        "acereason",
        "falcon_h1",
        "bitnet",
        "smollm3",
        "ernie4_5",
        "lfm2",
        "baichuan_m1",
        "exaone4",
        "gpt_oss",
        "lille-130m",
        "olmoe",
        "olmo2",
        "olmo3",
        "bailing_moe",
        "lfm2_moe",
        "nanochat",
        "nemotron_h",
        "afmoe",
        "jamba",
        "mamba2",
        "mistral3",
        "apertus",
        "nemotron_labs_diffusion",
        // VLM-only types (also present in VLMTypeRegistry)
        "paligemma",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "idefics3",
        "smolvlm",
        "fastvlm",
        "llava_qwen2",
        "pixtral",
        "lfm2_vl",
        "lfm2-vl",
        "glm_ocr",
    ]

    /// Model types that are vision-language only (not ambiguous with text LLMs).
    private static let visionOnlyModelTypes: Set<String> = [
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "paligemma",
        "idefics3",
        "smolvlm",
        "fastvlm",
        "llava_qwen2",
        "pixtral",
        "lfm2_vl",
        "lfm2-vl",
        "glm_ocr",
    ]

    /// Types the app runs as text-only models even with a `vision_config`: LFM3
    /// ships an empty one, and LFM3-Omni's vision encoder isn't in this build.
    /// Neither may load through the VLM factory.
    private static let textOnlyModelTypes: Set<String> = ["lfm3", "lfm3_omni"]

    private static let supportedBits: Set<Int> = [2, 3, 4, 5, 6, 8]
    private static let supportedGroupSizes: Set<Int> = [32, 64, 128]

    /// Checks a parsed config.json: model_type against `supportedModelTypes`,
    /// quantization (`quantization` and `text_config.quantization`) against the
    /// vendored mlx C core's affine limits — bits {2,3,4,5,6,8}, group sizes
    /// {32,64,128}, mode "affine" only.
    static func unsupportedReason(configJSON data: Data) -> String? {
        guard let object = try? JSONSerialization.jsonObject(with: data),
              let dict = object as? [String: Any]
        else {
            return "Couldn't read the model config."
        }

        guard let modelType = dict["model_type"] as? String,
              !modelType.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return "Model config is missing a model type."
        }

        if !supportedModelTypes.contains(modelType) {
            return "Architecture “\(modelType)” isn't supported."
        }

        // A PyTorch quantized build (FP8, GPTQ, AWQ…) names its method in
        // `quantization_config` (top-level or under `text_config`) and has no MLX
        // `quantization` block. MLX can't read its weights, except Inkling 2's native
        // `modelopt` NVFP4/MXFP8 checkpoints, which `Inkling2.prepareCheckpoint` maps
        // directly onto MLX quantized kernels.
        let textConfig = dict["text_config"] as? [String: Any]
        let hasMLXQuantization = dict["quantization"] != nil || textConfig?["quantization"] != nil
        if !hasMLXQuantization {
            let quantConfig = (dict["quantization_config"] as? [String: Any])
                ?? (textConfig?["quantization_config"] as? [String: Any])
            if let method = quantConfig?["quant_method"] as? String {
                let allowsModelOpt = method == "modelopt" && Inkling2.modelTypes.contains(modelType)
                if !allowsModelOpt {
                    return "This is a PyTorch \(method) build, not an MLX one. Look for an MLX version of this model."
                }
            }
        }

        if let reason = quantizationReason(dict["quantization"] as? [String: Any]) {
            return reason
        }
        if let textConfig,
           let reason = quantizationReason(textConfig["quantization"] as? [String: Any]) {
            return reason
        }

        return nil
    }

    /// Same check for a model directory on disk (reads its config.json).
    static func unsupportedReason(directory: URL) -> String? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL) else {
            return "Couldn't read the model config."
        }
        return unsupportedReason(configJSON: data)
    }

    /// Same check by model id; nil when the model isn't on disk.
    static func unsupportedReason(modelID: String) -> String? {
        guard let directory = ModelStore.localMLXModelDirectory(for: modelID) else {
            return nil
        }
        return unsupportedReason(directory: directory)
    }

    /// True when the model is a vision-language model: `config.json` has a
    /// non-empty `vision_config` dictionary, or `model_type` is in the VLM-only set.
    /// Shared types (qwen3_5, gemma3/4, mistral3) are vision only when
    /// `vision_config` is non-empty. Types in `textOnlyModelTypes` never are.
    static func visionCapable(configJSON data: Data) -> Bool {
        guard let object = try? JSONSerialization.jsonObject(with: data),
              let dict = object as? [String: Any]
        else {
            return false
        }
        let modelType = dict["model_type"] as? String
        if let modelType, textOnlyModelTypes.contains(modelType) {
            return false
        }
        if let visionConfig = dict["vision_config"] as? [String: Any], !visionConfig.isEmpty {
            return true
        }
        if let modelType, visionOnlyModelTypes.contains(modelType) {
            return true
        }
        return false
    }

    // MARK: - Quantization

    private static func quantizationReason(
        _ quant: [String: Any]?,
        defaultMode: String = "affine"
    ) -> String? {
        guard let quant else { return nil }

        let mode = (quant["mode"] as? String) ?? defaultMode
        if let rawBits = quant["bits"], !(rawBits is NSNull), intValue(rawBits) == nil {
            return "Quantization bit width isn't supported."
        }
        if let rawGroup = quant["group_size"], !(rawGroup is NSNull), intValue(rawGroup) == nil {
            return "Quantization group size isn't supported."
        }
        let bits = intValue(quant["bits"])
        let groupSize = intValue(quant["group_size"])

        switch mode {
        case "affine":
            if let bits, !supportedBits.contains(bits) {
                return "\(bits)-bit quantization isn't supported."
            }
            if let groupSize, !supportedGroupSizes.contains(groupSize) {
                return "Group size \(groupSize) isn't supported."
            }
        case "nvfp4":
            if let bits, bits != 4 {
                return "\(bits)-bit NVFP4 quantization isn't supported."
            }
            if let groupSize, groupSize != 16 {
                return "NVFP4 group size \(groupSize) isn't supported."
            }
        case "mxfp4":
            if let bits, bits != 4 {
                return "\(bits)-bit MXFP4 quantization isn't supported."
            }
            if let groupSize, groupSize != 32 {
                return "MXFP4 group size \(groupSize) isn't supported."
            }
        case "mxfp8":
            if let bits, bits != 8 {
                return "\(bits)-bit MXFP8 quantization isn't supported."
            }
            if let groupSize, groupSize != 32 {
                return "MXFP8 group size \(groupSize) isn't supported."
            }
        default:
            return "Quantization mode “\(mode)” isn't supported."
        }

        // Validate per-layer mixed-precision override sub-dictionaries (e.g. "lm_head").
        for key in quant.keys.sorted() {
            guard let nested = quant[key] as? [String: Any],
                  nested["bits"] != nil || nested["group_size"] != nil || nested["mode"] != nil
            else { continue }
            if let reason = quantizationReason(nested, defaultMode: mode) {
                return reason
            }
        }
        return nil
    }

    private static func intValue(_ any: Any?) -> Int? {
        if any is Bool { return nil }
        if let number = any as? NSNumber {
            guard CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
            let double = number.doubleValue
            guard double.isFinite, double.rounded(.towardZero) == double else { return nil }
            return Int(exactly: double)
        }
        if let string = any as? String {
            return Int(string)
        }
        return nil
    }
}
