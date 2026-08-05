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
        #expect(model.backend == .mlx)
        #expect(model.sourceRepoID == nil)
        #expect(model.installRestriction == nil)
        #expect(model.id == "mlx-community/Old-Model-4bit")
    }

    @Test func allPinnedMoEVariantsUseStreamedMoEBackend() {
        for variant in StreamedMoEInstall.variants {
            let streamed = ModelCatalog.model(for: variant.modelID)
            #expect(streamed?.backend == .streamedMoE)
            #expect(StreamedMoERuntimeSettings.allowedContextTokens.contains(
                streamed?.runtimeContextTokens ?? 0))
            #expect(streamed?.ctxTokens == 262_144)
            #expect(streamed?.supportsVision == false)
        }

        let dense = ModelCatalog.model(for: "mlx-community/gemma-4-12B-it-4bit")
        #expect(dense?.backend == .mlx)
        #expect(dense?.runtimeContextTokens == dense?.ctxTokens)
    }

    @Test func streamedInstallAndOrdinaryMLXCheckpointHaveDistinctIdentities() {
        for variant in StreamedMoEInstall.variants {
            let streamed = ModelCatalog.model(for: variant.modelID)
            let ordinary = ModelCatalog.model(for: variant.sourceRepoID)

            #expect(variant.modelID != variant.sourceRepoID)
            #expect(streamed?.backend == .streamedMoE)
            #expect(ordinary?.backend == .mlx)
            #expect(streamed?.id != ordinary?.id)
            #expect(streamed?.pickerDisplayName.hasPrefix("SSD ·") == true)
            #expect(ordinary?.pickerDisplayName.hasPrefix("MLX ·") == true)
            #expect(StreamedMoEInstall.supportsSSDStreaming(
                sourceModelID: ordinary?.id ?? ""))
        }
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

    @Test func qwen36OffersVisionMLXAndTextOnlySSDIdentities() {
        let mlx = ModelCatalog.model(for: StreamedMoEInstall.qwen36.sourceRepoID)
        let ssd = ModelCatalog.model(for: StreamedMoEInstall.qwen36.modelID)

        #expect(mlx?.backend == .mlx)
        #expect(mlx?.supportsVision == true)
        #expect(mlx.map { ModelCatalog.familyID(for: $0) } == "qwen36")
        #expect(ssd?.backend == .streamedMoE)
        #expect(ssd?.supportsVision == false)
        #expect(ssd?.supportsThinkingToggle == true)
        #expect(mlx?.pickerDisplayName == "MLX · Qwen3.6 35B A3B")
        #expect(ssd?.pickerDisplayName == "SSD · Qwen3.6 35B A3B")
    }

    @Test func lfm25CatalogOffersEveryOfficialPrecisionAndCommunityBF16() {
        let official = ModelCatalog.model(for: "LiquidAI/LFM2.5-2.6B-MLX")
        let community = ModelCatalog.model(for: "mlx-community/LFM2.5-2.6B-bf16")
        let officialVariants = ModelCatalog.models.filter {
            $0.huggingFaceRepoID == "LiquidAI/LFM2.5-2.6B-MLX"
        }
        let supported = officialVariants.filter { $0.installRestriction == nil }
        let restricted = officialVariants.filter { $0.installRestriction != nil }

        #expect(official?.sourceSubfolder == "4bit")
        #expect(official?.ctxTokens == 131_072)
        #expect(official?.thinking?.startsInside == true)
        #expect(official?.supportsThinkingToggle == false)
        #expect(official.map { ModelCatalog.familyID(for: $0) } == "lfm25")
        #expect(community?.sourceSubfolder == nil)
        #expect(community?.ctxTokens == 131_072)
        #expect(community?.thinking?.startsInside == true)
        #expect(community.map { ModelCatalog.familyID(for: $0) } == "lfm25")
        #expect(officialVariants.count == 8)
        #expect(Set(supported.compactMap(\.sourceSubfolder)) == ["4bit", "5bit", "6bit", "8bit", "bf16"])
        #expect(Set(restricted.compactMap(\.sourceSubfolder)) == ["mxfp4", "mxfp8", "nvfp4"])
    }

    @Test func catalogModelVariantSourceRoundTrips() throws {
        let original = ModelCatalog.model(for: "LiquidAI/LFM2.5-2.6B-MLX@5bit")!
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(CatalogModel.self, from: data)

        #expect(decoded.sourceSubfolder == "5bit")
        #expect(decoded.sourceRepoID == "LiquidAI/LFM2.5-2.6B-MLX")
        #expect(decoded.hugginFaceURL.absoluteString.hasSuffix("/tree/main/5bit"))
    }
}
