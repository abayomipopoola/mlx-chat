import Foundation

/// A model the app can download or register and run. Standard MLX model IDs are
/// Hugging Face repository IDs; streamed installs use a stable app-local ID.
struct CatalogModel: Identifiable, Hashable, Codable {
    enum Category: String, Codable {
        case general, coding, reasoning, imported
    }

    enum Backend: String, Codable {
        /// Standard MLX / mlx-swift-lm loading. This remains the default for
        /// existing persisted models and every dense model in the catalog.
        case mlx
        /// TurboFieldfare's MoE runtime. Common weights stay resident; routed
        /// experts are fetched from a registered `.gturbo` directory.
        case streamedMoE
    }

    let id: String
    let displayName: String
    let sizeGB: Double
    let ctxTokens: Int
    let quantLabel: String
    let category: Category
    let blurb: String
    let extraEOSTokens: Set<String>
    let backend: Backend
    /// How the model expresses chain-of-thought reasoning in its stream, or `nil`
    /// for a non-reasoning model. Drives both stream routing and the thinking toggle.
    let thinking: ThinkingConfig?
    /// True when the model can accept image inputs (VLM). Persisted; legacy
    /// UserDefaults entries without the key decode as `false`.
    let supportsVision: Bool
    /// Optional folder inside the Hugging Face repository that should be
    /// downloaded and flattened into the app's local model directory.
    let sourceSubfolder: String?
    /// Hugging Face repository to download when `id` is an app-local variant
    /// identity such as `owner/repo@5bit`. Nil means the id is the repo id.
    let sourceRepoID: String?
    /// Non-nil when the catalog should show a known variant but must not offer
    /// its download with the MLX runtime linked into this build.
    let installRestriction: String?

    init(
        id: String,
        displayName: String,
        sizeGB: Double,
        ctxTokens: Int,
        quantLabel: String,
        category: Category,
        blurb: String,
        extraEOSTokens: Set<String>,
        thinking: ThinkingConfig?,
        supportsVision: Bool = false,
        backend: Backend = .mlx,
        sourceSubfolder: String? = nil,
        sourceRepoID: String? = nil,
        installRestriction: String? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.sizeGB = sizeGB
        self.ctxTokens = ctxTokens
        self.quantLabel = quantLabel
        self.category = category
        self.blurb = blurb
        self.extraEOSTokens = extraEOSTokens
        self.backend = backend
        self.thinking = thinking
        self.supportsVision = supportsVision
        self.sourceSubfolder = sourceSubfolder
        self.sourceRepoID = sourceRepoID
        self.installRestriction = installRestriction
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        displayName = try c.decode(String.self, forKey: .displayName)
        sizeGB = try c.decode(Double.self, forKey: .sizeGB)
        ctxTokens = try c.decode(Int.self, forKey: .ctxTokens)
        quantLabel = try c.decode(String.self, forKey: .quantLabel)
        category = try c.decode(Category.self, forKey: .category)
        blurb = try c.decode(String.self, forKey: .blurb)
        extraEOSTokens = try c.decode(Set<String>.self, forKey: .extraEOSTokens)
        backend = try c.decodeIfPresent(Backend.self, forKey: .backend) ?? .mlx
        thinking = try c.decodeIfPresent(ThinkingConfig.self, forKey: .thinking)
        supportsVision = try c.decodeIfPresent(Bool.self, forKey: .supportsVision) ?? false
        sourceSubfolder = try c.decodeIfPresent(String.self, forKey: .sourceSubfolder)
        sourceRepoID = try c.decodeIfPresent(String.self, forKey: .sourceRepoID)
        installRestriction = try c.decodeIfPresent(String.self, forKey: .installRestriction)
    }

    func withSupportsVision(_ value: Bool) -> CatalogModel {
        guard value != supportsVision else { return self }
        return CatalogModel(
            id: id,
            displayName: displayName,
            sizeGB: sizeGB,
            ctxTokens: ctxTokens,
            quantLabel: quantLabel,
            category: category,
            blurb: blurb,
            extraEOSTokens: extraEOSTokens,
            thinking: thinking,
            supportsVision: value,
            backend: backend,
            sourceSubfolder: sourceSubfolder,
            sourceRepoID: sourceRepoID,
            installRestriction: installRestriction)
    }

    var huggingFaceRepoID: String { sourceRepoID ?? id }

    var hugginFaceURL: URL {
        let suffix = sourceSubfolder.map { "/tree/main/\($0)" } ?? ""
        return URL(string: "https://huggingface.co/\(huggingFaceRepoID)\(suffix)")!
    }

    /// Reasoning can be switched off per turn via the model's `enable_thinking`
    /// template kwarg (Qwen3 family, Ternary Bonsai, Gemma 4). R1 distills
    /// reason unconditionally.
    var supportsThinking: Bool { thinking != nil }

    var supportsThinkingToggle: Bool { thinking?.toggleable == true }

    /// TurboFieldfare's FP16 KV cache uses the SSD runtime setting (4K–32K).
    /// The model supports 256K, but exposing that full cache would defeat the
    /// low-memory backend. Dense MLX models keep their existing limit.
    var runtimeContextTokens: Int {
        backend == .streamedMoE ? StreamedMoEInstall.runtimeContextTokens : ctxTokens
    }

    /// Disambiguates the ordinary MLX checkpoint from its separately repacked
    /// SSD-streamed install in the compact chat picker.
    var pickerDisplayName: String {
        if backend == .streamedMoE { return "SSD · \(displayName)" }
        if StreamedMoEInstall.supportsSSDStreaming(sourceModelID: id) {
            return "MLX · \(displayName)"
        }
        return displayName
    }
}

/// How a model emits chain-of-thought reasoning inline in its output stream.
///
/// The reasoning span is delimited by markers the model writes itself; these
/// differ by family. Qwen3, Ternary Bonsai, and DeepSeek-R1 use
/// `<think>…</think>`; Gemma 4 uses a harmony-style thought channel,
/// `<|channel>thought … <channel|>`. All reach the
/// app as text because MLX decodes the stream with `skipSpecialTokens: false`.
struct ThinkingConfig: Hashable, Codable {
    /// Marker that opens the reasoning span (ignored when `startsInside`).
    let open: String
    /// Marker that closes it; the visible answer follows.
    let close: String
    /// The chat template pre-opens the span, so the stream begins already inside
    /// reasoning with no opening marker present (Qwen3.5 with thinking on).
    let startsInside: Bool
    /// Reasoning can be disabled per turn via the `enable_thinking` template kwarg.
    let toggleable: Bool

    /// Inline `<think>…</think>` tags (Qwen3 family, DeepSeek-R1 distills).
    static func thinkTags(startsInside: Bool = false, toggleable: Bool) -> ThinkingConfig {
        ThinkingConfig(open: "<think>", close: "</think>",
                       startsInside: startsInside, toggleable: toggleable)
    }

    /// Gemma 4 harmony-style thought channel: `<|channel>thought … <channel|>`.
    static let gemmaChannel = ThinkingConfig(
        open: "<|channel>thought", close: "<channel|>",
        startsInside: false, toggleable: true)
}

/// A curated group of related models, shown as one tappable card in Manage Models.
struct ModelFamily: Identifiable {
    enum Badge: String {
        case new = "New"
        case thinking = "Thinking"
        case coding = "Coding"
        case reasoning = "Reasoning"
        case recommended = "Recommended"
        case vision = "Vision"
    }

    let id: String
    let name: String
    let icon: String
    let blurb: String
    let badges: [Badge]
}

enum ModelCatalog {
    /// Display order of the family cards in Manage Models.
    static let families: [ModelFamily] = [
        ModelFamily(
            id: "lfm25", name: "LFM2.5", icon: "brain.head.profile",
            blurb: "Liquid AI's compact 2.6B reasoning model with a 128K context window.",
            badges: [.new, .thinking, .recommended]),
        ModelFamily(
            id: "qwen36", name: "Qwen3.6", icon: "bubble.left.and.bubble.right",
            blurb: "Qwen's 35B-A3B hybrid MoE model. Use standard MLX for vision or SSD streaming for a much smaller memory footprint.",
            badges: [.new, .thinking, .recommended, .vision]),
        ModelFamily(
            id: "qwen35", name: "Qwen3.5", icon: "bubble.left.and.bubble.right",
            blurb: "A compact Qwen generation with strong all-round quality, thinking mode, and 256K context.",
            badges: [.new, .thinking, .recommended]),
        ModelFamily(
            id: "ornith", name: "Ornith", icon: "bird",
            blurb: "Ornith 1.5 Qwen 3.5 family models with thinking mode and 256K context. The 35B-A3B MoE can use standard MLX or SSD streaming.",
            badges: [.new, .thinking]),
        ModelFamily(
            id: "bonsai", name: "Bonsai 2", icon: "leaf",
            blurb: "PrismML's ternary Qwen models. Close to full-precision quality in a fraction of the memory, with thinking mode, vision, and 256K context.",
            badges: [.new, .thinking, .vision]),
        ModelFamily(
            id: "gemma4", name: "Gemma 4", icon: "bubble.left.and.bubble.right",
            blurb: "The latest generation of open models from Google. Optimized for multi-turn chat and advanced reasoning, with 256K context.",
            badges: [.new, .recommended, .vision]),
        ModelFamily(
            id: "qwen3", name: "Qwen3", icon: "message",
            blurb: "Previous Qwen generation with thinking mode. Great quality per gigabyte.",
            badges: [.thinking]),
        ModelFamily(
            id: "llama", name: "Llama 3", icon: "message",
            blurb: "Meta's open instruct models. Reliable all-rounders with 128K context.",
            badges: []),
        ModelFamily(
            id: "phi", name: "Phi-4", icon: "brain",
            blurb: "Microsoft's reasoning-focused models.",
            badges: [.reasoning]),
        ModelFamily(
            id: "gemma", name: "Gemma 3", icon: "message",
            blurb: "Google's lightweight open models, tuned for efficiency.",
            badges: []),
        ModelFamily(
            id: "coder", name: "Qwen2.5 Coder", icon: "chevron.left.forwardslash.chevron.right",
            blurb: "Code-specialized Qwen models for local development.",
            badges: [.coding]),
        ModelFamily(
            id: "qwen25vl", name: "Qwen2.5 VL", icon: "eye",
            blurb: "Vision-language models that can see images and answer about them.",
            badges: [.new, .vision]),
        ModelFamily(
            id: "deepseek", name: "DeepSeek R1", icon: "brain.head.profile",
            blurb: "Reasoning distills that show their thinking step by step.",
            badges: [.thinking, .reasoning]),
        ModelFamily(
            id: "imported", name: "Imported", icon: "shippingbox",
            blurb: "Models you added by Hugging Face repo id.",
            badges: []),
    ]

    static func family(for id: String) -> ModelFamily? {
        families.first { $0.id == id }
    }

    /// Family a model belongs to (prefix match; check longer prefixes first).
    static func familyID(for model: CatalogModel) -> String {
        if model.category == .imported { return "imported" }
        let name = model.displayName
        if name.hasPrefix("LFM2.5") { return "lfm25" }
        if name.hasPrefix("Qwen3.6") { return "qwen36" }
        if name.hasPrefix("Qwen3.5") { return "qwen35" }
        if name.hasPrefix("Ornith") { return "ornith" }
        if name.hasPrefix("Bonsai") { return "bonsai" }
        if name.hasPrefix("Qwen3") { return "qwen3" }
        if name.hasPrefix("Qwen2.5 VL") { return "qwen25vl" }
        if name.hasPrefix("Qwen2.5 Coder") { return "coder" }
        if name.hasPrefix("Llama") { return "llama" }
        if name.hasPrefix("Phi") { return "phi" }
        if name.hasPrefix("Gemma 4") { return "gemma4" }
        if name.hasPrefix("Gemma") { return "gemma" }
        if name.hasPrefix("DeepSeek") { return "deepseek" }
        return "imported"
    }

    /// Curated entries. Repos, sizes, `max_position_embeddings`, and MLX architecture
    /// support (LLMTypeRegistry at mlx-swift-lm 3.31.4) verified against HF/source.
    static let models: [CatalogModel] = [
        CatalogModel(
            id: "LiquidAI/LFM2.5-2.6B-MLX", displayName: "LFM2.5 2.6B 4-bit",
            sizeGB: 1.47, ctxTokens: 131_072, quantLabel: "4-bit · LiquidAI", category: .general,
            blurb: "Official compact reasoning model, optimized for fast on-device use.",
            extraEOSTokens: ["<|im_end|>"],
            thinking: .thinkTags(startsInside: true, toggleable: false),
            sourceSubfolder: "4bit"),
        CatalogModel(
            id: "LiquidAI/LFM2.5-2.6B-MLX@5bit", displayName: "LFM2.5 2.6B 5-bit",
            sizeGB: 1.76, ctxTokens: 131_072, quantLabel: "5-bit · LiquidAI", category: .general,
            blurb: "Official 5-bit precision for a small quality increase over 4-bit.",
            extraEOSTokens: ["<|im_end|>"],
            thinking: .thinkTags(startsInside: true, toggleable: false),
            sourceSubfolder: "5bit", sourceRepoID: "LiquidAI/LFM2.5-2.6B-MLX"),
        CatalogModel(
            id: "LiquidAI/LFM2.5-2.6B-MLX@6bit", displayName: "LFM2.5 2.6B 6-bit",
            sizeGB: 2.04, ctxTokens: 131_072, quantLabel: "6-bit · LiquidAI", category: .general,
            blurb: "Official 6-bit precision balancing memory use and fidelity.",
            extraEOSTokens: ["<|im_end|>"],
            thinking: .thinkTags(startsInside: true, toggleable: false),
            sourceSubfolder: "6bit", sourceRepoID: "LiquidAI/LFM2.5-2.6B-MLX"),
        CatalogModel(
            id: "LiquidAI/LFM2.5-2.6B-MLX@8bit", displayName: "LFM2.5 2.6B 8-bit",
            sizeGB: 2.67, ctxTokens: 131_072, quantLabel: "8-bit · LiquidAI", category: .general,
            blurb: "Official high-fidelity quantized checkpoint.",
            extraEOSTokens: ["<|im_end|>"],
            thinking: .thinkTags(startsInside: true, toggleable: false),
            sourceSubfolder: "8bit", sourceRepoID: "LiquidAI/LFM2.5-2.6B-MLX"),
        CatalogModel(
            id: "LiquidAI/LFM2.5-2.6B-MLX@bf16", displayName: "LFM2.5 2.6B BF16",
            sizeGB: 5.02, ctxTokens: 131_072, quantLabel: "BF16 · LiquidAI", category: .general,
            blurb: "Official full-precision checkpoint for maximum fidelity.",
            extraEOSTokens: ["<|im_end|>"],
            thinking: .thinkTags(startsInside: true, toggleable: false),
            sourceSubfolder: "bf16", sourceRepoID: "LiquidAI/LFM2.5-2.6B-MLX"),
        CatalogModel(
            id: "LiquidAI/LFM2.5-2.6B-MLX@mxfp4", displayName: "LFM2.5 2.6B MXFP4",
            sizeGB: 1.46, ctxTokens: 131_072, quantLabel: "MXFP4 · LiquidAI", category: .general,
            blurb: "Official microscaling FP4 checkpoint.",
            extraEOSTokens: ["<|im_end|>"],
            thinking: .thinkTags(startsInside: true, toggleable: false),
            sourceSubfolder: "mxfp4", sourceRepoID: "LiquidAI/LFM2.5-2.6B-MLX",
            installRestriction: "MXFP4 isn't supported by the MLX Swift runtime in this build."),
        CatalogModel(
            id: "LiquidAI/LFM2.5-2.6B-MLX@mxfp8", displayName: "LFM2.5 2.6B MXFP8",
            sizeGB: 2.59, ctxTokens: 131_072, quantLabel: "MXFP8 · LiquidAI", category: .general,
            blurb: "Official microscaling FP8 checkpoint.",
            extraEOSTokens: ["<|im_end|>"],
            thinking: .thinkTags(startsInside: true, toggleable: false),
            sourceSubfolder: "mxfp8", sourceRepoID: "LiquidAI/LFM2.5-2.6B-MLX",
            installRestriction: "MXFP8 isn't supported by the MLX Swift runtime in this build."),
        CatalogModel(
            id: "LiquidAI/LFM2.5-2.6B-MLX@nvfp4", displayName: "LFM2.5 2.6B NVFP4",
            sizeGB: 1.53, ctxTokens: 131_072, quantLabel: "NVFP4 · LiquidAI", category: .general,
            blurb: "Official NVIDIA FP4 checkpoint.",
            extraEOSTokens: ["<|im_end|>"],
            thinking: .thinkTags(startsInside: true, toggleable: false),
            sourceSubfolder: "nvfp4", sourceRepoID: "LiquidAI/LFM2.5-2.6B-MLX",
            installRestriction: "NVFP4 isn't supported by the MLX Swift runtime in this build."),
        CatalogModel(
            id: "mlx-community/LFM2.5-2.6B-bf16", displayName: "LFM2.5 2.6B BF16",
            sizeGB: 5.39, ctxTokens: 131_072, quantLabel: "BF16 · mlx-community", category: .general,
            blurb: "Full-precision LFM2.5 checkpoint for maximum quality.",
            extraEOSTokens: ["<|im_end|>"],
            thinking: .thinkTags(startsInside: true, toggleable: false)),
        CatalogModel(
            id: StreamedMoEInstall.qwen36.sourceRepoID, displayName: "Qwen3.6 35B A3B",
            sizeGB: 20.4, ctxTokens: 262_144, quantLabel: "4-bit MoE", category: .general,
            blurb: "Hybrid MoE with about 3B active parameters. Use standard MLX or a separate low-memory SSD-streamed install.",
            extraEOSTokens: ["<|im_end|>", "<|endoftext|>"],
            thinking: .thinkTags(startsInside: true, toggleable: true),
            supportsVision: true),
        CatalogModel(
            id: StreamedMoEInstall.qwen36.modelID, displayName: "Qwen3.6 35B A3B",
            sizeGB: 19.5, ctxTokens: 262_144, quantLabel: "4-bit · SSD", category: .general,
            blurb: "Text-only Qwen MoE with a configurable context cache and SSD-streamed experts.",
            extraEOSTokens: ["<|im_end|>", "<|endoftext|>"],
            thinking: .thinkTags(startsInside: true, toggleable: true),
            backend: .streamedMoE),
        CatalogModel(
            id: "mlx-community/Qwen3.5-9B-4bit", displayName: "Qwen3.5 9B",
            sizeGB: 5.95, ctxTokens: 262_144, quantLabel: "4-bit", category: .general,
            blurb: "Latest Qwen3.5. Excellent all-round quality on Apple Silicon.",
            extraEOSTokens: ["<|im_end|>"], thinking: .thinkTags(startsInside: true, toggleable: true)),
        CatalogModel(
            id: "mlx-community/Qwen3.5-2B-4bit", displayName: "Qwen3.5 2B",
            sizeGB: 1.72, ctxTokens: 262_144, quantLabel: "4-bit", category: .general,
            blurb: "Small, fast Qwen3.5 for quick tasks.",
            extraEOSTokens: ["<|im_end|>"], thinking: .thinkTags(startsInside: true, toggleable: true)),
        CatalogModel(
            id: "ornith-ai/Ornith-1.5-9B-MLX-4bit", displayName: "Ornith 1.5 9B",
            sizeGB: 5.06, ctxTokens: 262_144, quantLabel: "4-bit", category: .general,
            blurb: "Dense Ornith 1.5. Qwen 3.5 reasoning with 256K context.",
            extraEOSTokens: ["<|im_end|>", "<|endoftext|>"],
            thinking: .thinkTags(startsInside: true, toggleable: true)),
        CatalogModel(
            id: StreamedMoEInstall.ornith15.sourceRepoID, displayName: "Ornith 1.5 35B A3B",
            sizeGB: 19.5, ctxTokens: 262_144, quantLabel: "4-bit MoE", category: .general,
            blurb: "Mixture-of-experts Ornith 1.5 with about 3B active parameters. Use standard MLX or a separate low-memory SSD-streamed install.",
            extraEOSTokens: ["<|im_end|>", "<|endoftext|>"],
            thinking: .thinkTags(startsInside: true, toggleable: true)),
        CatalogModel(
            id: StreamedMoEInstall.ornith15.modelID, displayName: "Ornith 1.5 35B A3B",
            sizeGB: 19.5, ctxTokens: 262_144, quantLabel: "4-bit · SSD", category: .general,
            blurb: "Text-only Ornith MoE with a configurable context cache and SSD-streamed experts.",
            extraEOSTokens: ["<|im_end|>", "<|endoftext|>"],
            thinking: .thinkTags(startsInside: true, toggleable: true),
            backend: .streamedMoE),
        CatalogModel(
            id: "prism-ml/Ternary-Bonsai-2-9B-mlx-2bit", displayName: "Bonsai 2 9B",
            sizeGB: 3.46, ctxTokens: 262_144, quantLabel: "Ternary 2-bit", category: .general,
            blurb: "Ternary Qwen 3.5 9B with vision. About a fifth of the full-precision size.",
            extraEOSTokens: ["<|im_end|>", "<|endoftext|>"],
            thinking: .thinkTags(startsInside: true, toggleable: true),
            supportsVision: true),
        CatalogModel(
            id: "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit", displayName: "Bonsai 2 27B",
            sizeGB: 8.60, ctxTokens: 262_144, quantLabel: "Ternary 2-bit", category: .general,
            blurb: "Ternary Qwen3.8 27B with vision. Large-model reasoning in under 9 GB.",
            extraEOSTokens: ["<|im_end|>", "<|endoftext|>"],
            thinking: .thinkTags(startsInside: true, toggleable: true),
            supportsVision: true),
        CatalogModel(
            id: "mlx-community/Qwen3-4B-4bit", displayName: "Qwen3 4B",
            sizeGB: 2.26, ctxTokens: 40_960, quantLabel: "4-bit", category: .general,
            blurb: "Solid general model with thinking mode.",
            extraEOSTokens: ["<|im_end|>"], thinking: .thinkTags(toggleable: true)),
        CatalogModel(
            id: "mlx-community/Qwen3-0.6B-4bit", displayName: "Qwen3 0.6B",
            sizeGB: 0.34, ctxTokens: 40_960, quantLabel: "4-bit", category: .general,
            blurb: "Tiny model for instant smoke tests.",
            extraEOSTokens: ["<|im_end|>"], thinking: .thinkTags(toggleable: true)),
        CatalogModel(
            id: "mlx-community/Llama-3.2-3B-Instruct-4bit", displayName: "Llama 3.2 3B",
            sizeGB: 1.81, ctxTokens: 131_072, quantLabel: "4-bit", category: .general,
            blurb: "Meta's compact instruct model.",
            extraEOSTokens: [], thinking: nil),
        CatalogModel(
            id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", displayName: "Llama 3.1 8B",
            sizeGB: 4.52, ctxTokens: 131_072, quantLabel: "4-bit", category: .general,
            blurb: "Meta's classic 8B all-rounder.",
            extraEOSTokens: [], thinking: nil),
        CatalogModel(
            id: "mlx-community/phi-4-4bit", displayName: "Phi-4 14B",
            sizeGB: 8.25, ctxTokens: 16_384, quantLabel: "4-bit", category: .reasoning,
            blurb: "Microsoft's reasoning-focused 14B.",
            extraEOSTokens: [], thinking: nil),
        CatalogModel(
            id: "mlx-community/gemma-4-12B-it-4bit", displayName: "Gemma 4 12B",
            sizeGB: 6.74, ctxTokens: 262_144, quantLabel: "4-bit", category: .general,
            blurb: "Google's latest generation. Strong chat and reasoning with 256K context.",
            extraEOSTokens: [], thinking: .gemmaChannel, supportsVision: true),
        CatalogModel(
            id: StreamedMoEInstall.standard.sourceRepoID, displayName: "Gemma 4 26B A4B",
            sizeGB: 14.3, ctxTokens: 262_144, quantLabel: "4-bit MoE", category: .general,
            blurb: "Google's instruction-tuned MoE model. Download as standard MLX or opt into low-memory SSD streaming.",
            extraEOSTokens: [], thinking: .gemmaChannel, supportsVision: true),
        CatalogModel(
            id: StreamedMoEInstall.standard.modelID, displayName: "Gemma 4 26B A4B",
            sizeGB: 14.3, ctxTokens: 262_144, quantLabel: "4-bit · SSD", category: .general,
            blurb: "MoE model with a configurable 4K–32K KV cache and SSD-streamed experts.",
            extraEOSTokens: [], thinking: .gemmaChannel,
            backend: .streamedMoE),
        CatalogModel(
            id: StreamedMoEInstall.qat.sourceRepoID, displayName: "Gemma 4 26B A4B QAT",
            sizeGB: 14.5, ctxTokens: 262_144, quantLabel: "QAT 4-bit MoE", category: .general,
            blurb: "Quantization-aware trained MoE checkpoint. Use standard MLX or a separate SSD-streamed install.",
            extraEOSTokens: [], thinking: .gemmaChannel, supportsVision: true),
        CatalogModel(
            id: StreamedMoEInstall.qat.modelID, displayName: "Gemma 4 26B A4B QAT",
            sizeGB: 14.5, ctxTokens: 262_144, quantLabel: "QAT 4-bit · SSD", category: .general,
            blurb: "QAT MoE model with a configurable 4K–32K KV cache and SSD-streamed experts.",
            extraEOSTokens: [], thinking: .gemmaChannel,
            backend: .streamedMoE),
        CatalogModel(
            id: "mlx-community/gemma-4-12B-it-qat-4bit", displayName: "Gemma 4 12B QAT",
            sizeGB: 10.99, ctxTokens: 262_144, quantLabel: "QAT 4-bit", category: .general,
            blurb: "Quantization-aware trained: closer to full precision at 4-bit.",
            extraEOSTokens: [], thinking: .gemmaChannel, supportsVision: true),
        CatalogModel(
            id: "mlx-community/gemma-3-1b-it-4bit", displayName: "Gemma 3 1B",
            sizeGB: 0.73, ctxTokens: 32_768, quantLabel: "4-bit", category: .general,
            blurb: "Google's lightweight text model.",
            extraEOSTokens: [], thinking: nil),
        CatalogModel(
            id: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", displayName: "Qwen2.5 Coder 7B",
            sizeGB: 4.28, ctxTokens: 32_768, quantLabel: "4-bit", category: .coding,
            blurb: "Strong local coding assistant.",
            extraEOSTokens: [], thinking: nil),
        CatalogModel(
            id: "mlx-community/Qwen2.5-VL-7B-Instruct-4bit", displayName: "Qwen2.5 VL 7B",
            sizeGB: 5.25, ctxTokens: 128_000, quantLabel: "4-bit", category: .general,
            blurb: "Vision-language model that can answer about images.",
            extraEOSTokens: ["<|im_end|>", "<|endoftext|>"], thinking: nil,
            supportsVision: true),
        CatalogModel(
            id: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit", displayName: "DeepSeek R1 Qwen 7B",
            sizeGB: 4.28, ctxTokens: 131_072, quantLabel: "4-bit", category: .reasoning,
            blurb: "Reasoning distill; shows its thinking.",
            extraEOSTokens: [], thinking: .thinkTags(toggleable: false)),
    ]

    static func model(for id: String) -> CatalogModel? {
        models.first { $0.id == id }
    }

    /// Fallback entry shape for user-imported repos. Known imported models are
    /// enriched with capabilities that cannot be inferred from the repo id alone.
    static func importedModel(repoID: String, supportsVision: Bool = false, configJSON: Data? = nil) -> CatalogModel {
        applyingKnownCapabilities(
            to: CatalogModel(
                id: repoID,
                displayName: repoID.split(separator: "/").last.map(String.init) ?? repoID,
                sizeGB: 0, ctxTokens: 8_192, quantLabel: "MLX", category: .imported,
                blurb: "Imported from HuggingFace.",
                extraEOSTokens: [], thinking: nil,
                supportsVision: supportsVision),
            configJSON: configJSON)
    }

    /// Upgrades persisted imported entries when support for one of their model
    /// families is added after the model was originally imported.
    static func applyingKnownCapabilities(to model: CatalogModel, configJSON: Data? = nil) -> CatalogModel {
        let id = model.id.lowercased()
        let contextTokens: Int
        let extraEOSTokens: Set<String>
        let thinking: ThinkingConfig

        if id == "prism-ml/ternary-bonsai-27b-mlx-2bit" {
            contextTokens = 262_144
            extraEOSTokens = ["<|im_end|>"]
            thinking = .thinkTags(startsInside: true, toggleable: true)
        } else if isOrnithQwenFamily(id) {
            // Ornith 1.0/1.5 are Qwen 3.5 reasoning models (dense 9B and MoE
            // 35B/397B). The chat template pre-opens <think>, so only </think>
            // appears in the decoded stream before the visible answer. The
            // dense 9B checkpoint is not a streamed-MoE source.
            contextTokens = 262_144
            extraEOSTokens = ["<|im_end|>", "<|endoftext|>"]
            thinking = .thinkTags(startsInside: true, toggleable: true)
        } else if let type = modelType(from: configJSON),
                  let inferred = thinkingConfig(forModelType: type) {
            let isPrism = type == PrismHadamardQwen35.modelType
            let usesLongContext = isPrism || type.hasPrefix("qwen3_5") || type.hasPrefix("gemma4")
            contextTokens = usesLongContext ? max(model.ctxTokens, 262_144) : model.ctxTokens
            extraEOSTokens = isPrism ? ["<|im_end|>", "<|endoftext|>"] : ["<|im_end|>"]
            thinking = inferred
        } else {
            return model
        }

        return CatalogModel(
            id: model.id,
            displayName: model.displayName,
            sizeGB: model.sizeGB,
            ctxTokens: contextTokens,
            quantLabel: model.quantLabel,
            category: model.category,
            blurb: model.blurb,
            extraEOSTokens: model.extraEOSTokens.union(extraEOSTokens),
            thinking: thinking,
            supportsVision: model.supportsVision,
            backend: model.backend,
            sourceSubfolder: model.sourceSubfolder,
            sourceRepoID: model.sourceRepoID,
            installRestriction: model.installRestriction)
    }

    /// Ornith 1.0 and 1.5 Hugging Face ids, regardless of owner or quant label.
    static func isOrnithQwenFamily(_ id: String) -> Bool {
        let lower = id.lowercased()
        return lower.contains("/ornith-1.0") || lower.contains("/ornith-1.5")
    }

    static func thinkingConfig(forModelType type: String) -> ThinkingConfig? {
        switch type {
        case "qwen3_5", "qwen3_5_moe", "qwen3_5_text", PrismHadamardQwen35.modelType:
            return .thinkTags(startsInside: true, toggleable: true)
        case "qwen3", "qwen3_moe":
            return .thinkTags(toggleable: true)
        case "gemma4", "gemma4_text", "gemma4_unified":
            return .gemmaChannel
        default:
            return nil
        }
    }

    private static func modelType(from configJSON: Data?) -> String? {
        guard let configJSON,
              let object = try? JSONSerialization.jsonObject(with: configJSON),
              let dict = object as? [String: Any],
              let type = dict["model_type"] as? String
        else { return nil }
        let trimmed = type.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
