import Foundation

/// A model the app can download and run. `id` is the HuggingFace repo id and doubles
/// as the engine/model id everywhere (settings, picker, persistence).
struct CatalogModel: Identifiable, Hashable, Codable {
    enum Category: String, Codable {
        case general, coding, reasoning, imported
    }

    let id: String
    let displayName: String
    let sizeGB: Double
    let ctxTokens: Int
    let quantLabel: String
    let category: Category
    let blurb: String
    let extraEOSTokens: Set<String>
    /// Chat template pre-opens `<think>` in the generation prompt (Qwen3.5 family):
    /// the stream starts inside a think block with no opening tag.
    let startsInsideThink: Bool
    /// Model may emit explicit `<think>…</think>` tags (Qwen3 family, R1 distills).
    let emitsThinkTags: Bool

    var hugginFaceURL: URL { URL(string: "https://huggingface.co/\(id)")! }

    /// Thinking can be switched off via the Qwen3-family `/no_think` soft
    /// switch. R1 distills think unconditionally, so they are NOT toggleable.
    var supportsThinkingToggle: Bool {
        displayName.hasPrefix("Qwen3") && (startsInsideThink || emitsThinkTags)
    }
}

/// A curated group of related models, shown as one tappable card in Manage Models.
struct ModelFamily: Identifiable {
    enum Badge: String {
        case new = "New"
        case thinking = "Thinking"
        case coding = "Coding"
        case reasoning = "Reasoning"
        case recommended = "Recommended"
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
            id: "qwen35", name: "Qwen3.5", icon: "bubble.left.and.bubble.right",
            blurb: "The latest Qwen flagship generation. Strong all-round quality with thinking mode and 256K context.",
            badges: [.new, .thinking, .recommended]),
        ModelFamily(
            id: "gemma4", name: "Gemma 4", icon: "bubble.left.and.bubble.right",
            blurb: "The latest generation of open models from Google. Optimized for multi-turn chat and advanced reasoning, with 256K context.",
            badges: [.new, .recommended]),
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
        if name.hasPrefix("Qwen3.5") { return "qwen35" }
        if name.hasPrefix("Qwen3") { return "qwen3" }
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
            id: "mlx-community/Qwen3.5-9B-4bit", displayName: "Qwen3.5 9B",
            sizeGB: 5.95, ctxTokens: 262_144, quantLabel: "4-bit", category: .general,
            blurb: "Latest Qwen3.5. Excellent all-round quality on Apple Silicon.",
            extraEOSTokens: ["<|im_end|>"], startsInsideThink: true, emitsThinkTags: true),
        CatalogModel(
            id: "mlx-community/Qwen3.5-2B-4bit", displayName: "Qwen3.5 2B",
            sizeGB: 1.72, ctxTokens: 262_144, quantLabel: "4-bit", category: .general,
            blurb: "Small, fast Qwen3.5 for quick tasks.",
            extraEOSTokens: ["<|im_end|>"], startsInsideThink: true, emitsThinkTags: true),
        CatalogModel(
            id: "mlx-community/Qwen3-4B-4bit", displayName: "Qwen3 4B",
            sizeGB: 2.26, ctxTokens: 40_960, quantLabel: "4-bit", category: .general,
            blurb: "Solid general model with thinking mode.",
            extraEOSTokens: ["<|im_end|>"], startsInsideThink: false, emitsThinkTags: true),
        CatalogModel(
            id: "mlx-community/Qwen3-0.6B-4bit", displayName: "Qwen3 0.6B",
            sizeGB: 0.34, ctxTokens: 40_960, quantLabel: "4-bit", category: .general,
            blurb: "Tiny model for instant smoke tests.",
            extraEOSTokens: ["<|im_end|>"], startsInsideThink: false, emitsThinkTags: true),
        CatalogModel(
            id: "mlx-community/Llama-3.2-3B-Instruct-4bit", displayName: "Llama 3.2 3B",
            sizeGB: 1.81, ctxTokens: 131_072, quantLabel: "4-bit", category: .general,
            blurb: "Meta's compact instruct model.",
            extraEOSTokens: [], startsInsideThink: false, emitsThinkTags: false),
        CatalogModel(
            id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", displayName: "Llama 3.1 8B",
            sizeGB: 4.52, ctxTokens: 131_072, quantLabel: "4-bit", category: .general,
            blurb: "Meta's classic 8B all-rounder.",
            extraEOSTokens: [], startsInsideThink: false, emitsThinkTags: false),
        CatalogModel(
            id: "mlx-community/phi-4-4bit", displayName: "Phi-4 14B",
            sizeGB: 8.25, ctxTokens: 16_384, quantLabel: "4-bit", category: .reasoning,
            blurb: "Microsoft's reasoning-focused 14B.",
            extraEOSTokens: [], startsInsideThink: false, emitsThinkTags: false),
        CatalogModel(
            id: "mlx-community/gemma-4-12B-it-4bit", displayName: "Gemma 4 12B",
            sizeGB: 6.74, ctxTokens: 262_144, quantLabel: "4-bit", category: .general,
            blurb: "Google's latest generation. Strong chat and reasoning with 256K context.",
            extraEOSTokens: [], startsInsideThink: false, emitsThinkTags: false),
        CatalogModel(
            id: "mlx-community/gemma-4-12B-it-qat-4bit", displayName: "Gemma 4 12B QAT",
            sizeGB: 10.99, ctxTokens: 262_144, quantLabel: "QAT 4-bit", category: .general,
            blurb: "Quantization-aware trained: closer to full precision at 4-bit.",
            extraEOSTokens: [], startsInsideThink: false, emitsThinkTags: false),
        CatalogModel(
            id: "mlx-community/gemma-3-1b-it-4bit", displayName: "Gemma 3 1B",
            sizeGB: 0.73, ctxTokens: 32_768, quantLabel: "4-bit", category: .general,
            blurb: "Google's lightweight text model.",
            extraEOSTokens: [], startsInsideThink: false, emitsThinkTags: false),
        CatalogModel(
            id: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", displayName: "Qwen2.5 Coder 7B",
            sizeGB: 4.28, ctxTokens: 32_768, quantLabel: "4-bit", category: .coding,
            blurb: "Strong local coding assistant.",
            extraEOSTokens: [], startsInsideThink: false, emitsThinkTags: false),
        CatalogModel(
            id: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit", displayName: "DeepSeek R1 Qwen 7B",
            sizeGB: 4.28, ctxTokens: 131_072, quantLabel: "4-bit", category: .reasoning,
            blurb: "Reasoning distill; shows its thinking.",
            extraEOSTokens: [], startsInsideThink: false, emitsThinkTags: true),
    ]

    static func model(for id: String) -> CatalogModel? {
        models.first { $0.id == id }
    }

    /// Fallback entry shape for user-imported repos.
    static func importedModel(repoID: String) -> CatalogModel {
        CatalogModel(
            id: repoID,
            displayName: repoID.split(separator: "/").last.map(String.init) ?? repoID,
            sizeGB: 0, ctxTokens: 8_192, quantLabel: "MLX", category: .imported,
            blurb: "Imported from HuggingFace.",
            extraEOSTokens: [], startsInsideThink: false, emitsThinkTags: false)
    }
}
