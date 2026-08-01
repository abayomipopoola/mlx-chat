import XCTest

/// End-to-end regression for the artifact pipeline: fenced HTML extraction,
/// CSS application in WKWebView, and JavaScript event handling.
final class ArtifactPreviewUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = ["--sidebar-expanded", "--seed-artifact-preview"]
        app.launch()
        XCTAssertTrue(
            app.buttons["modelPickerButton"].waitForExistence(timeout: 15),
            "app should boot to the chat header")
    }

    override func tearDownWithError() throws {
        app.terminate()
    }

    func testHTMLCSSAndJavaScriptPreview() {
        let transcript = app.scrollViews["transcriptScrollView"]
        XCTAssertTrue(transcript.waitForExistence(timeout: 4))
        transcript.swipeUp()

        let artifact = app.buttons.matching(NSPredicate(
            format: "label CONTAINS %@", "HTML")).firstMatch
        XCTAssertTrue(artifact.waitForExistence(timeout: 4), "HTML artifact card")
        artifact.click()

        XCTAssertTrue(
            app.staticTexts["HTML CSS JS ready"].waitForExistence(timeout: 5),
            "WebKit should load HTML, apply the grid CSS, and execute JavaScript")

        let firstCell = app.buttons["Cell 1"]
        XCTAssertTrue(firstCell.waitForExistence(timeout: 3))
        firstCell.click()
        XCTAssertTrue(
            app.staticTexts["Player O's Turn"].waitForExistence(timeout: 3),
            "JavaScript click handler should update the DOM")
    }
}
