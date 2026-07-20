import Foundation
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

    @Test func legacyCatalogModelDecodesWithoutSupportsVision() throws {
        // Old UserDefaults payload before the vision field existed.
        let legacy = """
            {
              "id": "mlx-community/Old-Model-4bit",
              "displayName": "Old Model",
              "sizeGB": 1.5,
              "ctxTokens": 8192,
              "quantLabel": "4-bit",
              "category": "general",
              "blurb": "Legacy entry.",
              "extraEOSTokens": [],
              "thinking": null
            }
            """
        let model = try JSONDecoder().decode(CatalogModel.self, from: Data(legacy.utf8))
        #expect(model.supportsVision == false)
        #expect(model.id == "mlx-community/Old-Model-4bit")
    }

    @Test func visionCatalogModelRoundTripsSupportsVision() throws {
        let original = CatalogModel(
            id: "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            displayName: "Qwen2.5 VL 7B",
            sizeGB: 5.25,
            ctxTokens: 128_000,
            quantLabel: "4-bit",
            category: .general,
            blurb: "Vision-language model.",
            extraEOSTokens: ["<|im_end|>"],
            thinking: nil,
            supportsVision: true)
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(CatalogModel.self, from: data)
        #expect(decoded.supportsVision == true)
        #expect(decoded.id == original.id)
        #expect(decoded.displayName == original.displayName)
    }

    @Test func qwen25VLFamilyMapsFromDisplayName() {
        let model = ModelCatalog.model(for: "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
        #expect(model?.supportsVision == true)
        #expect(model.map { ModelCatalog.familyID(for: $0) } == "qwen25vl")
    }
}
