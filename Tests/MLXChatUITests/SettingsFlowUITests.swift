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

        XCTAssertTrue(app.descendants(matching: .any)["addModelRow"]
            .waitForExistence(timeout: 3),
            "Models page with the add-model row")
        XCTAssertTrue(app.staticTexts["MODEL FAMILIES"].exists)
        app.buttons["backButton"].click()
        XCTAssertTrue(app.staticTexts["Settings"].waitForExistence(timeout: 3))
    }

    func testMoEDownloadDefaultsToMLXAndOffersSSDOptIn() {
        assertSSDOption(
            for: "mlx-community/gemma-4-26b-a4b-it-4bit")
    }

    func testQATMoEDownloadDefaultsToMLXAndOffersSSDOptIn() {
        assertSSDOption(
            for: "mlx-community/gemma-4-26B-A4B-it-qat-4bit")
    }

    func testQwen36FamilyAppearsAndSSDModeOffersRuntimeControls() {
        assertSSDOption(
            for: "mlx-community/Qwen3.6-35B-A3B-4bit",
            familyID: "qwen36")
    }

    func testImportedFamilyShowsItsModelsAfterOpeningFromBottomOfOverview() {
        app.terminate()
        app.launchArguments = [
            "--sidebar-expanded",
            "--route", "models",
            "--seed-imported-models",
        ]
        app.launch()

        let scrollView = app.scrollViews["modelsScrollView"]
        XCTAssertTrue(scrollView.waitForExistence(timeout: 15))

        let importedFamily = app.descendants(matching: .any)["familyCard.imported"]
        for _ in 0..<12 where !importedFamily.isHittable {
            scrollView.swipeUp()
        }
        XCTAssertTrue(importedFamily.isHittable, "Imported family should be reachable")
        importedFamily.click()

        let importedModel = app.descendants(matching: .any)[
            "modelRow.mlx-community/Imported-Regression-4bit"
        ]
        XCTAssertTrue(
            importedModel.waitForExistence(timeout: 3) && importedModel.isHittable,
            "Opening Imported should reset the overview scroll position and reveal its models")
    }

    private func assertSSDOption(
        for modelID: String,
        familyID: String = "gemma4"
    ) {
        app.buttons["settingsGearButton"].click()
        let manageRow = app.descendants(matching: .any)["manageModelsRow"]
        XCTAssertTrue(manageRow.waitForExistence(timeout: 3))
        manageRow.click()

        let family = app.descendants(matching: .any)["familyCard.\(familyID)"]
        XCTAssertTrue(family.waitForExistence(timeout: 3))
        family.click()

        let download = app.descendants(matching: .any)["downloadModel.\(modelID)"]
        XCTAssertTrue(download.waitForExistence(timeout: 3))
        download.click()

        let mlx = app.descendants(matching: .any)["downloadMode.mlx"]
        let ssd = app.descendants(matching: .any)["downloadMode.ssd"]
        XCTAssertTrue(mlx.waitForExistence(timeout: 3))
        XCTAssertEqual(mlx.value as? String, "Selected")
        XCTAssertTrue(ssd.exists, "the supported MoE model should offer SSD streaming")
        XCTAssertFalse(app.staticTexts["Copy Setup Commands"].exists)

        ssd.click()
        XCTAssertEqual(ssd.value as? String, "Selected")
        XCTAssertTrue(app.descendants(matching: .any)["ssdContextPicker"].exists)
        XCTAssertTrue(app.descendants(matching: .any)["ssdExpertCachePicker"].exists)
        XCTAssertTrue(app.staticTexts["Context"].exists)
        XCTAssertTrue(app.staticTexts["Expert cache"].exists)
        XCTAssertTrue(
            app.staticTexts["Automatic setup"].exists
                || app.staticTexts["Ready to use"].exists)
        XCTAssertTrue(
            app.buttons["Download SSD Model"].exists
                || app.buttons["Use SSD Model"].exists)
    }

    func testBackFromSettingsReturnsToChat() {
        app.buttons["settingsGearButton"].click()
        XCTAssertTrue(app.staticTexts["Settings"].waitForExistence(timeout: 3))
        app.buttons["backButton"].click()
        XCTAssertTrue(app.staticTexts["Settings"].waitForNonExistence(timeout: 3))
        XCTAssertTrue(app.buttons["modelPickerButton"].exists, "chat header is back")
    }
}
