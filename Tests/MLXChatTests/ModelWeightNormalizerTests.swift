import Foundation
import Testing
@testable import MLXChat

@Suite struct ModelWeightNormalizerTests {
    @Test func removesLanguageModelWrapperFromSafetensorsKeys() throws {
        let header = #"{"__metadata__":{"format":"mlx"},"language_model.model.embed_tokens.weight":{"dtype":"BF16","shape":[8,4],"data_offsets":[0,64]},"language_model.model.layers.0.ffn_norm.weight":{"dtype":"BF16","shape":[4],"data_offsets":[64,72]}}"#

        let result = try ModelWeightNormalizer.normalizedHeaderJSON(Data(header.utf8))
        let normalized = try #require(result)
        let dictionary = try #require(
            JSONSerialization.jsonObject(with: normalized) as? [String: Any])

        #expect(dictionary["model.embed_tokens.weight"] != nil)
        #expect(dictionary["model.layers.0.ffn_norm.weight"] != nil)
        #expect(dictionary["language_model.model.embed_tokens.weight"] == nil)
        #expect(dictionary["__metadata__"] != nil)
    }

    @Test func removesLanguageModelWrapperFromShardIndex() throws {
        let index = #"{"metadata":{"total_size":72},"weight_map":{"language_model.model.embed_tokens.weight":"model-1.safetensors","language_model.model.embedding_norm.weight":"model-2.safetensors"}}"#

        let result = try ModelWeightNormalizer.normalizedIndexJSON(Data(index.utf8))
        let normalized = try #require(result)
        let dictionary = try #require(
            JSONSerialization.jsonObject(with: normalized) as? [String: Any])
        let weightMap = try #require(dictionary["weight_map"] as? [String: Any])

        #expect(weightMap["model.embed_tokens.weight"] as? String == "model-1.safetensors")
        #expect(weightMap["model.embedding_norm.weight"] as? String == "model-2.safetensors")
        #expect(weightMap["language_model.model.embed_tokens.weight"] == nil)
    }

    @Test func matchingHeadersRemainUntouched() throws {
        let header = #"{"model.embed_tokens.weight":{"dtype":"BF16","shape":[8,4],"data_offsets":[0,64]}}"#
        #expect(try ModelWeightNormalizer.normalizedHeaderJSON(Data(header.utf8)) == nil)
    }

    @Test func fillsMissingBlockFeedForwardDimensionFromIntermediateSize() throws {
        let config = #"{"model_type":"lfm2","hidden_size":2048,"intermediate_size":10752,"block_auto_adjust_ff_dim":false}"#

        let result = try ModelWeightNormalizer.normalizedConfigJSON(Data(config.utf8))
        let normalized = try #require(result)
        let dictionary = try #require(
            JSONSerialization.jsonObject(with: normalized) as? [String: Any])

        #expect(dictionary["block_ff_dim"] as? Int == 10_752)
        #expect(dictionary["intermediate_size"] as? Int == 10_752)
    }

    @Test func configWithBlockFeedForwardDimensionRemainsUntouched() throws {
        let config = #"{"model_type":"lfm2","intermediate_size":10752,"block_ff_dim":10752}"#
        #expect(try ModelWeightNormalizer.normalizedConfigJSON(Data(config.utf8)) == nil)
    }
}
