import XCTest

/// Boot smoke: every surface the rest of the suite (and the user) depends on
/// is present after launch — sidebar chrome, header pickers, composer.
final class AppLaunchUITests: MLXUITestCase {

    func testSidebarChromeExists() {
        XCTAssertTrue(app.buttons["newChatButton"].exists, "New chat row")
        XCTAssertTrue(app.textFields["Search"].exists, "liquid-glass search field")
        XCTAssertTrue(app.buttons["settingsGearButton"].exists, "settings gear in the top band")
        XCTAssertTrue(conversationList.exists, "conversation list")
    }

    func testHeaderChromeExists() {
        XCTAssertTrue(app.buttons["modelPickerButton"].exists, "model capsule")
        XCTAssertTrue(app.buttons["promptPresetButton"].exists, "prompt preset circle")
    }

    func testComposerExists() {
        XCTAssertTrue(app.descendants(matching: .any)["composerField"].exists,
                      "composer text field")
        XCTAssertTrue(app.buttons["composerSendButton"].exists, "send button")
        XCTAssertTrue(app.staticTexts["Runs 100% on-device • Private & offline"].exists)
    }
}
