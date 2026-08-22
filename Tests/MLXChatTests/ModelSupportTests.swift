import Foundation
import Testing

import MLXLLM
import MLXLMCommon
import MLXVLM
@testable import MLXChat

/// Contract: ModelSupport rejects configs the bundled MLX runtime cannot run.
/// An unknown `model_type` already throws cleanly inside MLX-Swift-LM, but an
/// out-of-range quantization dies as a SIGTRAP via the mlx C error handler
/// (the prism-ml/Bonsai-27B-mlx-1bit crash) — so both must be caught before
/// any load is attempted, and the picker must not offer doomed models.
@Suite struct ModelSupportTests {
    private func json(_ string: String) -> Data { Data(string.utf8) }

    @Test func plainSupportedArchitectureIsLoadable() {
        #expect(ModelSupport.unsupportedReason(configJSON: json(#"{"model_type":"qwen3"}"#)) == nil)
    }

    @Test func affine4BitQuantizationIsLoadable() {
        let config = #"{"model_type":"llama","quantization":{"group_size":64,"bits":4}}"#
        #expect(ModelSupport.unsupportedReason(configJSON: json(config)) == nil)
    }

    @Test func oneBitQuantizationIsRejected() {
        // The actual crash config: prism-ml/Bonsai-27B-mlx-1bit.
        let config = #"{"model_type":"qwen3_5","quantization":{"group_size":128,"bits":1}}"#
        let reason = ModelSupport.unsupportedReason(configJSON: json(config))
        #expect(reason != nil)
        #expect(reason?.localizedCaseInsensitiveContains("bit") == true)
    }

    @Test func sevenBitQuantizationIsRejected() {
        // mlx's affine quantize accepts 2–6 and 8 — 7 is excluded too.
        let config = #"{"model_type":"llama","quantization":{"group_size":64,"bits":7}}"#
        #expect(ModelSupport.unsupportedReason(configJSON: json(config)) != nil)
    }

    @Test func oddGroupSizeIsRejected() {
        let config = #"{"model_type":"llama","quantization":{"group_size":96,"bits":4}}"#
        #expect(ModelSupport.unsupportedReason(configJSON: json(config)) != nil)
    }

    @Test func badQuantizationInsideTextConfigIsRejected() {
        let config = #"{"model_type":"qwen3_5","text_config":{"quantization":{"group_size":128,"bits":1}}}"#
        #expect(ModelSupport.unsupportedReason(configJSON: json(config)) != nil)
    }

    @Test func unknownArchitectureIsRejected() {
        let reason = ModelSupport.unsupportedReason(configJSON: json(#"{"model_type":"futurama9"}"#))
        #expect(reason != nil)
        #expect(reason?.localizedCaseInsensitiveContains("futurama9") == true)
    }

    @Test func missingModelTypeIsRejected() {
        #expect(ModelSupport.unsupportedReason(configJSON: json(#"{}"#)) != nil)
    }

    @Test func malformedJSONIsRejected() {
        #expect(ModelSupport.unsupportedReason(configJSON: json("not json")) != nil)
    }

    @Test(arguments: ["llama", "qwen3", "qwen3_5", "bitnet", "gemma4", "lfm2", "mistral", "qwen2_5_vl"])
    func registryCornerstonesAreListed(_ type: String) {
        #expect(ModelSupport.supportedModelTypes.contains(type))
    }

    @Test func lfm25ConfigIsLoadable() {
        let config = #"{"model_type":"lfm2","rope_parameters":{"rope_theta":10000000},"quantization":{"group_size":64,"bits":4,"mode":"affine"}}"#
        #expect(ModelSupport.unsupportedReason(configJSON: json(config)) == nil)
    }

    /// Drift guard: the static list must stay in sync with the registries of the
    /// mlx-swift-lm version actually linked (LLM ∪ VLM; update after upgrades).
    @Test func staticListMatchesLLMAndVLMRegistries() async {
        for type in ModelSupport.supportedModelTypes {
            let inLLM = await LLMTypeRegistry.shared.contains(type)
            let inVLM = await VLMTypeRegistry.shared.contains(type)
            #expect(inLLM || inVLM, "\(type) is not registered in LLMTypeRegistry or VLMTypeRegistry")
        }
    }

    @Test func visionCapableWhenVisionConfigPresent() {
        let config = #"{"model_type":"gemma3","vision_config":{"hidden_size":1152}}"#
        #expect(ModelSupport.visionCapable(configJSON: json(config)))
    }

    @Test func visionCapableForVLMOnlyModelTypeWithoutVisionConfig() {
        let config = #"{"model_type":"qwen2_5_vl"}"#
        #expect(ModelSupport.visionCapable(configJSON: json(config)))
    }

    @Test func visionCapableFalseForPlainTextQwen3() {
        let config = #"{"model_type":"qwen3"}"#
        #expect(!ModelSupport.visionCapable(configJSON: json(config)))
    }

    // MARK: - directoryHasWeights

    @Test func directoryHasWeightsHappyPath() throws {
        let dir = try makeWeightFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))
        try Data("weights".utf8).write(to: dir.appendingPathComponent("model.safetensors"))
        #expect(ModelStore.directoryHasWeights(dir))
    }

    @Test func directoryHasWeightsRejectsMissingIndexShard() throws {
        let dir = try makeWeightFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))
        let index = #"{"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.weight":"model-00002-of-00002.safetensors"}}"#
        try Data(index.utf8).write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        try Data("shard1".utf8).write(to: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        #expect(!ModelStore.directoryHasWeights(dir))
    }

    @Test func directoryHasWeightsRejectsZeroByteShard() throws {
        let dir = try makeWeightFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))
        let index = #"{"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.weight":"model-00002-of-00002.safetensors"}}"#
        try Data(index.utf8).write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        try Data().write(to: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        try Data("shard2".utf8).write(to: dir.appendingPathComponent("model-00002-of-00002.safetensors"))
        #expect(!ModelStore.directoryHasWeights(dir))
    }

    @Test func directoryHasWeightsRejectsSiblingPartial() throws {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-weights-parent-\(UUID().uuidString)", isDirectory: true)
        let dest = parent.appendingPathComponent("model", isDirectory: true)
        let partial = parent.appendingPathComponent("model.partial", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: parent) }
        try FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: partial, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dest.appendingPathComponent("config.json"))
        try Data("weights".utf8).write(to: dest.appendingPathComponent("model.safetensors"))
        #expect(!ModelStore.directoryHasWeights(dest))
    }

    private func makeWeightFixtureDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-weights-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}
