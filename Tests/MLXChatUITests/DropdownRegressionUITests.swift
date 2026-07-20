import XCTest

/// E2E regression suite for the header dropdowns and their chrome.
///
/// These drive the real app because this feature area's worst bugs compiled
/// green and passed unit tests: a dropdown that never opened (its overlay
/// swallowed the anchor's taps), a panel clipped by the header's
/// safeAreaInset bar, an editor sheet that died with its host panel, and a
/// dropdown that ignored outside clicks, navigation, and Esc.
final class DropdownRegressionUITests: MLXUITestCase {

    // MARK: - Helpers

    private var modelPanel: XCUIElement { app.staticTexts["Select model"] }
    private var promptPanel: XCUIElement { app.staticTexts["Prompt"] }

    private func openModelDropdown(file: StaticString = #filePath, line: UInt = #line) {
        app.buttons["modelPickerButton"].click()
        XCTAssertTrue(modelPanel.waitForExistence(timeout: 3),
                      "capsule click should open the model dropdown",
                      file: file, line: line)
    }

    // MARK: - Sidebar chrome

    func testSidebarTopBandShowsSettingsGear() {
        XCTAssertTrue(app.buttons["settingsGearButton"].exists,
                      "settings gear should live in the sidebar's top band")
    }

    // MARK: - Model dropdown

    func testModelDropdownOpensWithMenuContent() {
        openModelDropdown()
        XCTAssertTrue(app.staticTexts["Models"].exists)
        XCTAssertTrue(app.buttons["Download Models…"].exists)
    }

    func testOutsideClickDismissesModelDropdown() {
        openModelDropdown()
        app.windows.firstMatch
            .coordinate(withNormalizedOffset: CGVector(dx: 0.55, dy: 0.55))
            .click()
        XCTAssertTrue(modelPanel.waitForNonExistence(timeout: 3),
                      "outside click should dismiss the dropdown")
    }

    func testEscapeDismissesModelDropdown() {
        openModelDropdown()
        app.typeKey(.escape, modifierFlags: [])
        XCTAssertTrue(modelPanel.waitForNonExistence(timeout: 3),
                      "Esc should dismiss the dropdown")
    }

    func testSelectingAModelClosesDropdown() {
        openModelDropdown()
        app.buttons["modelRow.apple-intelligence"].click()
        XCTAssertTrue(modelPanel.waitForNonExistence(timeout: 3))
    }

    func testSettingsNavigationClosesDropdown() {
        openModelDropdown()
        app.buttons["settingsGearButton"].click()
        XCTAssertTrue(app.staticTexts["Settings"].waitForExistence(timeout: 3),
                      "gear should push the Settings page")
        XCTAssertTrue(modelPanel.waitForNonExistence(timeout: 3),
                      "pushing Settings must close the dropdown")
    }

    // MARK: - Prompt dropdown

    func testPromptDropdownOpensAndChoosingPresetCloses() {
        app.buttons["promptPresetButton"].click()
        XCTAssertTrue(promptPanel.waitForExistence(timeout: 3),
                      "sparkle click should open the prompt dropdown")
        app.buttons["General"].click()
        XCTAssertTrue(promptPanel.waitForNonExistence(timeout: 3))
    }

    func testEditPromptPresentsEditorSheet() {
        app.buttons["promptPresetButton"].click()
        XCTAssertTrue(promptPanel.waitForExistence(timeout: 3))
        app.buttons["Edit General Prompt…"].click()
        let editor = app.staticTexts["Edit General Prompt"]
        XCTAssertTrue(editor.waitForExistence(timeout: 3),
                      "editor sheet must outlive the closed dropdown")
        app.buttons["Cancel"].click()
        XCTAssertTrue(editor.waitForNonExistence(timeout: 3))
    }
}
