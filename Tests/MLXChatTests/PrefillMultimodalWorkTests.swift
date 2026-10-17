import Testing
@testable import TurboFieldfare

@Suite struct PrefillMultimodalWorkTests {
    @Test func imageSpansStayWholeAndTextSplitsAtChunkSize() {
        let work = PrefillChunkPlanner.multimodalWork(
            tokenCount: 300,
            imageRanges: [20..<200],
            chunkTokens: 128)
        #expect(work.filter(\.isImage).map(\.range) == [20..<200])
        #expect(work.filter { !$0.isImage }.allSatisfy { $0.range.count <= 128 })
        #expect(work.first?.range == 0..<20)
        #expect(work.last?.range == 200..<300)
    }

    @Test func textOnlyPromptSplitsWithoutImageItems() {
        let work = PrefillChunkPlanner.multimodalWork(
            tokenCount: 200,
            imageRanges: [],
            chunkTokens: 64)
        #expect(work.allSatisfy { !$0.isImage })
        #expect(work.map(\.range.count) == [64, 64, 64, 8])
    }
}
