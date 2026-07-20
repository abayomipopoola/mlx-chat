import XCTest

/// In-frame navigation: Settings page, Models page, and back to the chat.
final class SettingsFlowUITests: MLXUITestCase {

    func testSettingsPageContent() {
        app.buttons["settingsGearButton"].click()
        XCTAssertTrue(app.staticTexts["Settings"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.descendants(matching: .any)["manageModelsRow"].exists)
        XCTAssertTrue(app.staticTexts["Auto-Unload After"].exists)
        XCTAssertTrue(app.descendants(matching: .any)["aiEngineCard"].exists, "AI engine card")
        XCTAssertTrue(app.staticTexts["Appearance"].exists)
    }

    func testManageModelsNavigatesAndBack() {
        app.buttons["settingsGearButton"].click()
        let manageRow = app.descendants(matching: .any)["manageModelsRow"]
        XCTAssertTrue(manageRow.waitForExistence(timeout: 3))
        manageRow.click()

        XCTAssertTrue(app.descendants(matching: .any)["importModelRow"]
            .waitForExistence(timeout: 3),
            "Models page with the import row above Downloaded")
        app.buttons["backButton"].click()
        XCTAssertTrue(app.staticTexts["Settings"].waitForExistence(timeout: 3))
    }

    func testBackFromSettingsReturnsToChat() {
        app.buttons["settingsGearButton"].click()
        XCTAssertTrue(app.staticTexts["Settings"].waitForExistence(timeout: 3))
        app.buttons["backButton"].click()
        XCTAssertTrue(app.staticTexts["Settings"].waitForNonExistence(timeout: 3))
        XCTAssertTrue(app.buttons["modelPickerButton"].exists, "chat header is back")
    }
}
