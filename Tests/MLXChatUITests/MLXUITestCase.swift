import XCTest

/// Shared base for the E2E suite: every test launches the real app fresh with
/// the sidebar expanded and the most recent conversation selected (via the
/// app's UI-verification hooks), and fails fast if the header never appears.
class MLXUITestCase: XCTestCase {
    var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = try XCUIApplication(url: targetAppURL())
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

    /// Resolve the app built beside the UI-test runner. Launching by URL avoids
    /// attaching to an installed release that shares the same bundle identifier.
    private func targetAppURL() throws -> URL {
        var directory = Bundle(for: type(of: self)).bundleURL

        for _ in 0..<8 {
            let candidate = directory.appendingPathComponent("MLX Chat.app")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
            directory.deleteLastPathComponent()
        }

        throw TargetAppError.notFound
    }
}

private enum TargetAppError: LocalizedError {
    case notFound

    var errorDescription: String? {
        "Could not find MLX Chat.app beside the UI-test runner."
    }
}
