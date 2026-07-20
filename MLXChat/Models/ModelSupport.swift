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

        if let reason = quantizationReason(dict["quantization"] as? [String: Any]) {
            return reason
        }
        if let textConfig = dict["text_config"] as? [String: Any],
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
        guard let directory = ModelStore.localModelDirectory(for: modelID) else {
            return nil
        }
        return unsupportedReason(directory: directory)
    }

    /// True when the model is a vision-language model: `config.json` has a
    /// `vision_config` dictionary, or `model_type` is in the VLM-only set.
    /// Shared types (qwen3_5, gemma3/4, mistral3) are vision only when
    /// `vision_config` is present.
    static func visionCapable(configJSON data: Data) -> Bool {
        guard let object = try? JSONSerialization.jsonObject(with: data),
              let dict = object as? [String: Any]
        else {
            return false
        }
        if dict["vision_config"] is [String: Any] {
            return true
        }
        if let modelType = dict["model_type"] as? String,
           visionOnlyModelTypes.contains(modelType) {
            return true
        }
        return false
    }

    // MARK: - Quantization

    private static func quantizationReason(_ quant: [String: Any]?) -> String? {
        guard let quant else { return nil }

        if let mode = quant["mode"] as? String, mode != "affine" {
            return "Quantization mode “\(mode)” isn't supported."
        }

        if let bits = intValue(quant["bits"]), !supportedBits.contains(bits) {
            return "\(bits)-bit quantization isn't supported."
        }

        if let groupSize = intValue(quant["group_size"]), !supportedGroupSizes.contains(groupSize) {
            return "Group size \(groupSize) isn't supported."
        }

        return nil
    }

    private static func intValue(_ any: Any?) -> Int? {
        switch any {
        case let i as Int: return i
        case let n as NSNumber: return n.intValue
        case let s as String: return Int(s)
        default: return nil
        }
    }
}
