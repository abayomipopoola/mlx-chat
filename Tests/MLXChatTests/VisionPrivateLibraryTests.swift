import Metal
import XCTest
@testable import TurboFieldfare

final class VisionPrivateLibraryTests: XCTestCase {
    func testVisionPrivateLibraryContainsRequiredKernels() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No default Metal device")
        }
        let library = try MetalContext.privateLibrary(device: device, module: "vision")
        XCTAssertTrue(library.functionNames.contains("vision_normalize_patches"))
        XCTAssertTrue(library.functionNames.contains("vision_attention_online_72"))
    }
}
