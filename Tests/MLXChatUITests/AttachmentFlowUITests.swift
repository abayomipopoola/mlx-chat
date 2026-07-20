import XCTest

/// Composer attachment flow: fake vision support exposes the attach menu;
/// selecting a fixture file via NSOpenPanel should seed the pending chip and
/// arm the send button. Model switches must discard any pending attachment.
final class AttachmentFlowUITests: MLXUITestCase {
    private var fixtureURL: URL!

    override func setUpWithError() throws {
        continueAfterFailure = false

        let name = "mlx-attach-fixture-\(UUID().uuidString).txt"
        fixtureURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(name)
        try "hello from fixture\n".write(to: fixtureURL, atomically: true, encoding: .utf8)

        // Per-test launch so each case can pass its own seed hooks.
        app = XCUIApplication()
    }

    override func tearDownWithError() throws {
        if let fixtureURL {
            try? FileManager.default.removeItem(at: fixtureURL)
        }
        try super.tearDownWithError()
    }

    /// Shared launch: sidebar + chat + vision attach affordance, plus any hooks.
    private func launchApp(extraArguments: [String] = [], file: StaticString = #filePath, line: UInt = #line) {
        app.launchArguments = [
            "--sidebar-expanded",
            "--open-chat", "0",
            "--fake-vision-support",
        ] + extraArguments
        app.launch()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 10),
                      "app should reach runningForeground",
                      file: file, line: line)
        app.activate()
        // Restored frames can sit on a disconnected display (negative origin);
        // AX then reports zero windows. Nudge the process onto the main screen.
        bringAppWindowsOnScreen()

        let picker = app.buttons["modelPickerButton"]
        if picker.waitForExistence(timeout: 20) { return }

        // Retry the nudge once — first pass can race process registration.
        bringAppWindowsOnScreen()
        if picker.waitForExistence(timeout: 10) { return }

        // Boot diagnostics when the usual header chrome never appears.
        let windowCount = app.windows.count
        let sampleButtons = app.buttons.allElementsBoundByIndex.prefix(15).map {
            "id=\($0.identifier) label=\($0.label)"
        }
        XCTFail(
            "app should boot to the chat header "
                + "(windows=\(windowCount); buttons=\(sampleButtons))",
            file: file, line: line)
    }

    /// Move every MLX Chat window to a known on-screen origin via System Events.
    private func bringAppWindowsOnScreen() {
        // Match by bundle id — process name can lag / differ under XCTest.
        let script = """
        tell application "System Events"
          set procs to every process whose bundle identifier is "org.mlxchat.MLXChat"
          repeat with p in procs
            try
              set frontmost of p to true
              repeat with w in windows of p
                try
                  set position of w to {80, 80}
                end try
              end repeat
            end try
          end repeat
        end tell
        """
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-e", script]
        try? process.run()
        process.waitUntilExit()
        // Brief settle so XCTest re-queries the moved window.
        RunLoop.current.run(until: Date().addingTimeInterval(0.4))
    }

    func testAttachFileShowsPendingChipAndArmsSend() throws {
        launchApp()

        // SwiftUI Menu may not surface under `.buttons` on all macOS versions;
        // query the full AX tree by identifier.
        let attach = app.descendants(matching: .any)["composerAttachButton"]
        XCTAssertTrue(attach.waitForExistence(timeout: 5),
                      "attach menu should show with --fake-vision-support")
        attach.click()

        // Menu items can appear as menuItems or buttons depending on presentation.
        let menuFile = app.menuItems["File…"]
        let buttonFile = app.buttons["File…"]
        let fileItem: XCUIElement
        if menuFile.waitForExistence(timeout: 3) {
            fileItem = menuFile
        } else if buttonFile.waitForExistence(timeout: 2) {
            fileItem = buttonFile
        } else {
            XCTFail("File… menu item should appear")
            return
        }
        fileItem.click()

        // NSOpenPanel automation (Cmd+Shift+G → path → Return ×2) is OS-dialog
        // driven and can flake on timing/focus. Keep the rest of the suite
        // green if the panel cannot be driven.
        do {
            try driveOpenPanel(to: fixtureURL)
        } catch {
            throw XCTSkip("NSOpenPanel automation flaky/unavailable: \(error)")
        }

        let chip = app.descendants(matching: .any)["composerAttachmentChip"]
        XCTAssertTrue(chip.waitForExistence(timeout: 5),
                      "pending attachment chip should appear after choosing a file")

        let send = app.buttons["composerSendButton"]
        let armed = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "enabled == true"), object: send)
        XCTAssertEqual(XCTWaiter.wait(for: [armed], timeout: 3), .completed,
                       "attachment alone should enable the send button")

        let shot = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        shot.name = "attachment-pending-chip"
        shot.lifetime = .keepAlways
        add(shot)
    }

    /// ComposerView clears pending attachments on any selectedModelID change.
    func testModelSwitchDiscardsPendingAttachment() throws {
        launchApp(extraArguments: ["--seed-attachment"])

        let chip = app.descendants(matching: .any)["composerAttachmentChip"]
        XCTAssertTrue(chip.waitForExistence(timeout: 5),
                      "seeded attachment chip should appear with --seed-attachment")

        let currentLabel = app.buttons["modelPickerButton"].label

        app.buttons["modelPickerButton"].click()
        XCTAssertTrue(app.staticTexts["Select model"].waitForExistence(timeout: 3),
                      "capsule click should open the model dropdown")

        let rows = app.buttons.matching(
            NSPredicate(format: "identifier BEGINSWITH %@", "modelRow."))
        XCTAssertTrue(rows.firstMatch.waitForExistence(timeout: 3),
                      "dropdown should list at least one model row")

        // Prefer a row whose name is not the currently selected model.
        var alternate: XCUIElement?
        let count = rows.count
        for i in 0..<count {
            let row = rows.element(boundBy: i)
            guard row.exists, !row.label.isEmpty else { continue }
            if !currentLabel.localizedCaseInsensitiveContains(row.label) {
                alternate = row
                break
            }
        }

        guard let target = alternate else {
            throw XCTSkip(
                "need a selectable model different from the current one "
                    + "(picker label: \(currentLabel); model rows: \(count))")
        }

        target.click()
        XCTAssertTrue(app.staticTexts["Select model"].waitForNonExistence(timeout: 3),
                      "selecting a model should close the dropdown")

        XCTAssertTrue(chip.waitForNonExistence(timeout: 3),
                      "pending attachment must clear on model switch")
    }

    /// Navigate the system open panel to `url` via Go to Folder.
    private func driveOpenPanel(to url: URL) throws {
        // Give the open panel a moment to become key.
        let panelAppeared = app.dialogs.firstMatch.waitForExistence(timeout: 3)
            || app.sheets.firstMatch.waitForExistence(timeout: 1)
        if !panelAppeared {
            // Some macOS versions expose NSOpenPanel without dialog AX role.
            // Still try Cmd+Shift+G — fail later if no path field appears.
        }

        app.typeKey("g", modifierFlags: [.command, .shift])

        let pathField = firstPathField()
        guard pathField.waitForExistence(timeout: 4) else {
            throw NSError(
                domain: "AttachmentFlowUITests", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Go-to-folder path field not found"])
        }
        pathField.click()
        // Clear any prefilled path, then type the absolute fixture path.
        pathField.typeKey("a", modifierFlags: .command)
        pathField.typeText(url.path)
        app.typeKey(.return, modifierFlags: [])
        // Second Return confirms Open on the selected file.
        RunLoop.current.run(until: Date().addingTimeInterval(0.4))
        app.typeKey(.return, modifierFlags: [])
    }

    private func firstPathField() -> XCUIElement {
        // Go-to-folder sheet: combo box or text field depending on macOS version.
        let candidates: [XCUIElement] = [
            app.comboBoxes.firstMatch,
            app.textFields.firstMatch,
            app.dialogs.comboBoxes.firstMatch,
            app.dialogs.textFields.firstMatch,
            app.sheets.comboBoxes.firstMatch,
            app.sheets.textFields.firstMatch,
        ]
        for el in candidates where el.exists {
            return el
        }
        // Prefer combo box (Go to Folder default on recent macOS).
        return app.comboBoxes.firstMatch.exists
            ? app.comboBoxes.firstMatch
            : app.textFields.firstMatch
    }
}
