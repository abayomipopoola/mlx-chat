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
}
