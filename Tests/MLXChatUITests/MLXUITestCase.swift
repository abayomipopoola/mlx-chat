import XCTest

/// Shared base for the E2E suite: every test launches the real app fresh with
/// the sidebar expanded and the most recent conversation selected (via the
/// app's UI-verification hooks), and fails fast if the header never appears.
class MLXUITestCase: XCTestCase {
    var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = ["--sidebar-expanded", "--open-chat", "0"]
        app.launch()
        XCTAssertTrue(app.buttons["modelPickerButton"].waitForExistence(timeout: 15),
                      "app should boot to the chat header")
    }

    override func tearDownWithError() throws {
        app.terminate()
    }

    /// The sidebar conversation list, for scoped row queries.
    var conversationList: XCUIElement {
        app.scrollViews["conversationList"]
    }

    /// Selects Apple Intelligence so tests never load an MLX model (and so
    /// the user's engine choice is restored afterwards).
    func selectAppleIntelligence() {
        app.buttons["modelPickerButton"].click()
        XCTAssertTrue(app.staticTexts["Select model"].waitForExistence(timeout: 3))
        app.buttons["modelRow.apple-intelligence"].click()
        XCTAssertTrue(app.staticTexts["Select model"].waitForNonExistence(timeout: 3))
    }
}
