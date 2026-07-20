import XCTest

/// Import Custom Model sheet: cheap emptiness gate vs full normalize, and
/// that a bare HF root URL is a no-op (enabled button, no dismiss, no error).
final class ImportSheetUITests: MLXUITestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = [
            "--sidebar-expanded",
            "--route", "models",
        ]
        app.launch()
        // Models route may hide the chat header; wait for the import row.
        XCTAssertTrue(
            app.descendants(matching: .any)["importModelRow"]
                .waitForExistence(timeout: 15),
            "app should boot to Manage Models with the import row")
    }

    func testImportButtonCheapGateAndNoOpBareHFURL() {
        let importRow = app.descendants(matching: .any)["importModelRow"]
        XCTAssertTrue(importRow.waitForExistence(timeout: 3))
        importRow.click()

        XCTAssertTrue(app.staticTexts["Import Custom Model"].waitForExistence(timeout: 3),
                      "import sheet should present")

        let repoField = app.descendants(matching: .any)["importRepoField"]
        XCTAssertTrue(repoField.waitForExistence(timeout: 3), "repo text field")
        repoField.click()
        repoField.typeText("https://huggingface.co/")

        let importButton = app.buttons["Import"]
        XCTAssertTrue(importButton.waitForExistence(timeout: 2))
        // Cheap gate: non-whitespace input enables Import even when normalize → "".
        XCTAssertTrue(importButton.isEnabled,
                      "Import should be enabled for non-empty HF root URL (cheap gate)")

        importButton.click()
        // runImport no-ops on empty normalized id — sheet stays, no error.
        RunLoop.current.run(until: Date().addingTimeInterval(0.5))
        XCTAssertTrue(app.staticTexts["Import Custom Model"].exists,
                      "sheet must not dismiss on bare HF root")
        // Avoid apostrophes in the format string (NSPredicate parse failures).
        let errorProbe = app.staticTexts.matching(NSPredicate(
            format: "label CONTAINS[c] %@ OR label CONTAINS[c] %@",
            "error", "fail"))
        XCTAssertFalse(errorProbe.firstMatch.exists,
                       "no-op import must not surface an error banner")

        // Clear and type a well-formed bare repo id.
        repoField.click()
        repoField.typeKey("a", modifierFlags: .command)
        repoField.typeText("mlx-community/Foo-4bit")
        XCTAssertTrue(importButton.isEnabled,
                      "Import should stay enabled for a bare owner/repo id")

        // Do not actually import (no network). Close with Cancel.
        app.buttons["Cancel"].click()
        XCTAssertTrue(app.staticTexts["Import Custom Model"].waitForNonExistence(timeout: 3),
                      "Cancel should dismiss the sheet")
    }
}
