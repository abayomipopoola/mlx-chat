import XCTest

/// Camera-paced product-demo walkthrough (~60–90 s). Gated by
/// `DEMO_RECORDING=1` (orchestrator: `TEST_RUNNER_DEMO_RECORDING=1`).
/// Not a regression suite — waits/existence only, to keep the path reliable.
final class DemoRecordingUITests: MLXUITestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
        // Launch is owned by the walkthrough so demo-specific args apply.
        app = XCUIApplication()
    }

    override func tearDownWithError() throws {
        app?.terminate()
        app = nil
    }

    // MARK: - Walkthrough

    func testRecordDemoWalkthrough() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["DEMO_RECORDING"] == "1",
            "Set DEMO_RECORDING=1 (or TEST_RUNNER_DEMO_RECORDING=1) to run the camera walkthrough")

        // --seed-demo inserts + selects the showcase conversation.
        // --seed-attachment shows the file chip without driving NSOpenPanel.
        app.launchArguments = [
            "--sidebar-expanded",
            "--seed-demo",
            "--seed-attachment",
        ]
        app.launch()
        XCTAssertTrue(
            app.wait(for: .runningForeground, timeout: 10),
            "app should reach runningForeground")
        app.activate()
        bringAppWindowsOnScreen()

        let picker = app.buttons["modelPickerButton"]
        if !picker.waitForExistence(timeout: 20) {
            bringAppWindowsOnScreen()
            XCTAssertTrue(
                picker.waitForExistence(timeout: 10),
                "app should boot to the chat header")
        }

        // 1. Seeded conversation on screen.
        hold(3.5)

        // 2. Slowly scroll the transcript so markdown/math render on camera.
        let transcript = app.scrollViews["transcriptScrollView"]
        _ = transcript.waitForExistence(timeout: 5)
        for _ in 0..<3 {
            if transcript.exists {
                transcript.swipeUp()
            }
            hold(2.0)
        }

        // 3. New chat (button — same as ConversationFlowUITests; visible on camera).
        let newChat = app.buttons["newChatButton"]
        if newChat.waitForExistence(timeout: 3) {
            newChat.click()
        } else {
            // Fallback: File-menu New Chat is ⌃N.
            conversationList.click()
            app.typeKey("n", modifierFlags: .control)
        }
        _ = app.staticTexts["What can I help with?"].waitForExistence(timeout: 4)
        hold(1.5)

        // 4. Human-paced typing into the composer; attachment chip already seeded.
        let composer = app.descendants(matching: .any)["composerField"]
        XCTAssertTrue(composer.waitForExistence(timeout: 5), "composer field")
        composer.click()
        typeHuman(
            "Summarize the attached notes in three bullet points.",
            into: composer,
            charDelay: 0.09)
        // Chip from --seed-attachment (ComposerView onAppear seam).
        _ = app.descendants(matching: .any)["composerAttachmentChip"]
            .waitForExistence(timeout: 3)
        // Do not send — generation is nondeterministic on camera.
        hold(3.5)

        // 5. Settings → Manage Models → Import sheet → search focus ring.
        app.buttons["settingsGearButton"].click()
        _ = app.staticTexts["Settings"].waitForExistence(timeout: 4)
        hold(1.2)

        let manageRow = app.descendants(matching: .any)["manageModelsRow"]
        XCTAssertTrue(manageRow.waitForExistence(timeout: 4), "Manage Models row")
        manageRow.click()
        _ = app.descendants(matching: .any)["addModelRow"].waitForExistence(timeout: 4)
        hold(1.2)

        let addRow = app.descendants(matching: .any)["addModelRow"]
        addRow.click()
        _ = app.staticTexts["Import Custom Model"].waitForExistence(timeout: 4)
        hold(2.5)

        app.buttons["Cancel"].click()
        _ = app.staticTexts["Import Custom Model"].waitForNonExistence(timeout: 3)
        hold(1.0)

        // Models search has no accessibility id; placeholder is "Search models".
        let modelsSearch = app.textFields["Search models"]
        if modelsSearch.waitForExistence(timeout: 3) {
            modelsSearch.click()
            hold(2.5)
        } else if let fallback = firstModelsSearchField() {
            fallback.click()
            hold(2.5)
        }

        // 6. Back to chat for a closing frame.
        app.buttons["backButton"].click()
        _ = app.staticTexts["Settings"].waitForExistence(timeout: 3)
        hold(0.8)
        app.buttons["backButton"].click()
        _ = app.buttons["modelPickerButton"].waitForExistence(timeout: 4)
        hold(3.5)
    }

    // MARK: - Pacing helpers

    private func hold(_ seconds: TimeInterval) {
        RunLoop.current.run(until: Date().addingTimeInterval(seconds))
    }

    /// Types `text` one character at a time so the composer looks human on camera.
    private func typeHuman(
        _ text: String,
        into element: XCUIElement,
        charDelay: TimeInterval
    ) {
        for character in text {
            element.typeText(String(character))
            if charDelay > 0 {
                hold(charDelay)
            }
        }
    }

    /// Move every MLX Chat window onto the main screen (same approach as
    /// AttachmentFlowUITests — restored frames can sit on a disconnected display).
    private func bringAppWindowsOnScreen() {
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
        RunLoop.current.run(until: Date().addingTimeInterval(0.4))
    }

    /// Fallback when the models search TextField is not labeled by placeholder.
    private func firstModelsSearchField() -> XCUIElement? {
        let candidates = app.textFields.allElementsBoundByIndex
        for field in candidates where field.exists {
            let label = field.label
            let value = field.value as? String ?? ""
            let placeholder = field.placeholderValue ?? ""
            if label.localizedCaseInsensitiveContains("search")
                || placeholder.localizedCaseInsensitiveContains("search")
                || value.localizedCaseInsensitiveContains("search") {
                return field
            }
        }
        // Last resort: any text field on the models page after import row exists.
        return app.textFields.firstMatch.exists ? app.textFields.firstMatch : nil
    }
}
