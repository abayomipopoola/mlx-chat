import XCTest

/// Click-outside must resign StudioSearchField focus when ringOnFocusOnly is on.
final class SearchFocusUITests: MLXUITestCase {

    /// Focus the sidebar search, type, click empty detail chrome, type again —
    /// the second keystrokes must not land in the search field.
    func testSidebarSearchClickOutsideClearsFocus() {
        let field = app.textFields["sidebarSearchField"]
        XCTAssertTrue(field.waitForExistence(timeout: 3), "sidebar search TextField")

        field.click()
        field.typeText("abc")
        let valueAfterType = field.value as? String ?? ""
        XCTAssertTrue(valueAfterType.contains("abc"),
                      "typed text should land in the search field, got \(valueAfterType)")

        // Empty-ish detail area (same offset as dropdown outside-click tests).
        app.windows.firstMatch
            .coordinate(withNormalizedOffset: CGVector(dx: 0.55, dy: 0.55))
            .click()

        // No element re-focused; keystrokes must not append to the search field.
        app.typeText("xyz")

        let valueAfterOutside = field.value as? String ?? ""
        XCTAssertEqual(valueAfterOutside, "abc",
                       "click outside should clear focus so later typing never reaches search")
    }
}
