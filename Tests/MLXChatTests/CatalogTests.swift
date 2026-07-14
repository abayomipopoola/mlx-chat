import Testing
@testable import MLXChat

/// Contract: catalog integrity — ids are unique, the default engine needs no
/// download, and every curated entry carries a plausible size and context window.
@Suite struct CatalogTests {
    @Test func idsAreUnique() {
        let ids = ModelCatalog.models.map(\.id)
        #expect(Set(ids).count == ids.count)
    }

    @Test func defaultEngineIsAppleIntelligence() {
        #expect(Keys.Defaults.selectedModelID == appleIntelligenceEngineID)
    }

    @Test func entriesHavePositiveSizeAndContext() {
        for model in ModelCatalog.models {
            #expect(model.sizeGB > 0, "\(model.id) has no download size")
            #expect(model.ctxTokens > 0, "\(model.id) has no context window")
        }
    }

    @Test func ternaryBonsaiImportedModelSupportsThinkingToggle() {
        let model = ModelCatalog.importedModel(
            repoID: "prism-ml/Ternary-Bonsai-27B-mlx-2bit")

        #expect(model.supportsThinkingToggle)
        #expect(model.thinking?.open == "<think>")
        #expect(model.thinking?.close == "</think>")
        #expect(model.thinking?.startsInside == true)
        #expect(model.ctxTokens == 262_144)
        #expect(model.extraEOSTokens.contains("<|im_end|>"))
    }

    @Test func ternaryBonsaiCapabilitiesUpgradeExistingImport() {
        let staleImport = CatalogModel(
            id: "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
            displayName: "Ternary-Bonsai-27B-mlx-2bit",
            sizeGB: 0,
            ctxTokens: 8_192,
            quantLabel: "MLX",
            category: .imported,
            blurb: "Imported from HuggingFace.",
            extraEOSTokens: [],
            thinking: nil)

        let upgraded = ModelCatalog.applyingKnownCapabilities(to: staleImport)

        #expect(upgraded.supportsThinkingToggle)
        #expect(upgraded.thinking?.startsInside == true)
    }
}
