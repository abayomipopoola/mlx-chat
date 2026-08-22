import XCTest

/// Camera-paced product-demo walkthrough (~60–90 s). Gated by
/// `DEMO_RECORDING=1` (orchestrator: `TEST_RUNNER_DEMO_RECORDING=1`).
/// Not a regression suite — waits/existence only, to keep the path reliable.
final class DemoRecordingUITests: MLXUITestCase {

    private let qatSourceModelID =
        "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
    private let qatStreamedModelID =
        "streamed-moe/gemma-4-26b-a4b-it-qat-4bit"

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

    func testCaptureDeterministicReleaseVideoFrames() throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(
            environment["RELEASE_VIDEO_CAPTURE"] == "1"
                || environment["TEST_RUNNER_RELEASE_VIDEO_CAPTURE"] == "1",
            "Set TEST_RUNNER_RELEASE_VIDEO_CAPTURE=1 to capture release frames")

        launchCapture(extraArguments: selectedSSDDemoArguments)
        XCTAssertTrue(app.staticTexts["What can I help with?"].waitForExistence(timeout: 8))
        captureWindow(named: "01-new-chat")

        launchCapture(extraArguments: selectedSSDDemoArguments + ["--open-model-dropdown"])
        XCTAssertTrue(app.staticTexts["Select model"].waitForExistence(timeout: 8))
        captureWindow(named: "02-model-picker")

        launchCapture(extraArguments: ["--route", "models", "--seed-streamed-demo-models"])
        XCTAssertTrue(app.staticTexts["Manage Models"].waitForExistence(timeout: 8))
        captureWindow(named: "03-manage-models")

        launchCapture(extraArguments: ["--route", "models"])
        openGemmaFamily()
        captureWindow(named: "04-gemma-family")

        launchCapture(extraArguments: ["--route", "models"])
        openGemmaFamily()
        openQATDownloadSheet()
        XCTAssertEqual(
            app.descendants(matching: .any)["downloadMode.mlx"].value as? String,
            "Selected")
        captureWindow(named: "05-download-choices")

        launchCapture(extraArguments: ["--route", "models"])
        openGemmaFamily()
        openQATDownloadSheet()
        let ssdMode = app.descendants(matching: .any)["downloadMode.ssd"]
        XCTAssertTrue(ssdMode.waitForExistence(timeout: 5))
        ssdMode.click()
        XCTAssertTrue(app.staticTexts["Runtime balance"].waitForExistence(timeout: 5))
        captureWindow(named: "06-ssd-runtime-balance")

        launchCapture(extraArguments: selectedSSDDemoArguments + ["--seed-moe-demo-prompt"])
        XCTAssertTrue(
            app.staticTexts["Does a larger expert cache make the SSD model smarter?"]
                .waitForExistence(timeout: 8))
        captureWindow(named: "07-cache-question")

        launchCapture(extraArguments: selectedSSDDemoArguments + ["--seed-moe-demo-answer"])
        XCTAssertTrue(app.staticTexts["Thought for 7s"].waitForExistence(timeout: 8))
        XCTAssertTrue(
            app.staticTexts.matching(
                NSPredicate(format: "label BEGINSWITH %@", "No."))
                .firstMatch.exists)
        captureWindow(named: "08-cache-answer")
    }

    func testCaptureChromeAndOrnithFrames() throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(
            environment["RELEASE_VIDEO_CAPTURE"] == "1"
                || environment["TEST_RUNNER_RELEASE_VIDEO_CAPTURE"] == "1",
            "Set TEST_RUNNER_RELEASE_VIDEO_CAPTURE=1 to capture release frames")

        launchCapture(extraArguments: ornithChromeArguments)
        XCTAssertTrue(app.staticTexts["What can I help with?"].waitForExistence(timeout: 8))
        captureWindow(named: "01-new-chat")

        launchCapture(extraArguments: ornithChromeArguments + ["--open-model-dropdown"])
        XCTAssertTrue(app.staticTexts["Select model"].waitForExistence(timeout: 8))
        captureWindow(named: "02-model-picker")

        launchCapture(extraArguments: ornithChromeArguments + [
            "--seed-ornith-html-preview",
            "--expand-artifacts",
        ])
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "HTML"))
            .firstMatch.waitForExistence(timeout: 8))
        hold(2.0)
        captureWindow(named: "03-html-preview")

        launchCapture(extraArguments: ornithChromeLightArguments + [
            "--seed-ornith-html-preview",
            "--expand-artifacts",
        ])
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "HTML"))
            .firstMatch.waitForExistence(timeout: 8))
        hold(2.0)
        captureWindow(named: "04-html-preview-light")
    }

    /// Square set v2: one coherent session. Apple Intelligence is selected, the
    /// SSD Ornith 35B is picked from the menu, the switch is held, and every
    /// answer after that is attributed to the model that was chosen.
    /// One square-set frame: the seeded gradient-descent conversation.
    func testCaptureGradientDescentChat() throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(
            environment["RELEASE_VIDEO_CAPTURE"] == "1"
                || environment["TEST_RUNNER_RELEASE_VIDEO_CAPTURE"] == "1",
            "Set TEST_RUNNER_RELEASE_VIDEO_CAPTURE=1 to capture release frames")

        launchCapture(extraArguments: ornithChromeArguments + ["--seed-demo"])
        XCTAssertTrue(
            app.staticTexts["Explain how gradient descent works, with the key math."]
                .waitForExistence(timeout: 10))
        hold(2.5)
        captureWindow(named: "05-gradient-descent")
    }

    func testCaptureSSDSessionFrames() throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(
            environment["RELEASE_VIDEO_CAPTURE"] == "1"
                || environment["TEST_RUNNER_RELEASE_VIDEO_CAPTURE"] == "1",
            "Set TEST_RUNNER_RELEASE_VIDEO_CAPTURE=1 to capture release frames")

        // 01 — the picker, with Apple Intelligence still selected.
        //
        // Apple Intelligence cannot be the *launch* selection: the window is then
        // never ordered onscreen (reproducible with `open -n` too). So launch on a
        // local model, switch to Apple Intelligence through the menu, and reopen.
        launchCapture(extraArguments: ssdOrnithArguments)
        XCTAssertTrue(app.staticTexts["What can I help with?"].waitForExistence(timeout: 10))
        let picker = app.descendants(matching: .any)["modelPickerButton"]
        XCTAssertTrue(picker.waitForExistence(timeout: 6))
        picker.click()
        let appleRow = app.descendants(matching: .any)
            .matching(NSPredicate(format: "label CONTAINS %@", "Apple Intelligence"))
            .firstMatch
        XCTAssertTrue(appleRow.waitForExistence(timeout: 6))
        appleRow.click()
        hold(1.0)
        picker.click()
        XCTAssertTrue(app.staticTexts["Select model"].waitForExistence(timeout: 6))
        hold(0.8)
        captureWindow(named: "01-picker-apple")

        // 02 — the SSD model chosen, still loading into RAM.
        launchCapture(extraArguments: ssdOrnithArguments + ["--fake-model-loading"])
        XCTAssertTrue(app.staticTexts["What can I help with?"].waitForExistence(timeout: 8))
        hold(1.0)
        captureWindow(named: "02-loading")

        // 03 — the answer, from the model that was selected.
        launchCapture(extraArguments: ssdOrnithArguments + ["--seed-demo-ssd"])
        XCTAssertTrue(
            app.staticTexts["Explain how gradient descent works, with the key math."]
                .waitForExistence(timeout: 10))
        hold(2.5)
        captureWindow(named: "03-gradient-descent")

        // 04/05 — the live HTML artifact, same model, dark then light.
        launchCapture(extraArguments: ssdOrnithArguments + [
            "--seed-html-preview-ssd",
            "--expand-artifacts",
        ])
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "HTML"))
            .firstMatch.waitForExistence(timeout: 8))
        hold(2.0)
        captureWindow(named: "04-html-preview")

        launchCapture(extraArguments: ssdOrnithLightArguments + [
            "--seed-html-preview-ssd",
            "--expand-artifacts",
        ])
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "HTML"))
            .firstMatch.waitForExistence(timeout: 8))
        hold(2.0)
        captureWindow(named: "05-html-preview-light")
    }

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
            // Fallback: File-menu New Chat is ⌘N.
            conversationList.click()
            app.typeKey("n", modifierFlags: .command)
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

    /// Streamed-MoE models available, Apple Intelligence still selected.
    private var appleIntelligenceArguments: [String] {
        [
            // Without a seeded model list the window is never ordered onscreen
            // when Apple Intelligence is the selection.
            "--seed-streamed-demo-models",
            "-selectedModelID", "apple-intelligence",
            "-appearance", "dark",
        ]
    }

    private var ssdOrnithArguments: [String] {
        [
            "--seed-streamed-demo-models",
            "-selectedModelID", "streamed-moe/ornith-1.5-35b-a3b-4bit",
            "-appearance", "dark",
        ]
    }

    private var ssdOrnithLightArguments: [String] {
        [
            "--seed-streamed-demo-models",
            "-selectedModelID", "streamed-moe/ornith-1.5-35b-a3b-4bit",
            "-appearance", "light",
        ]
    }

    private var ornithChromeArguments: [String] {
        [
            "--seed-ornith-demo-model",
            "-selectedModelID", "ornith-ai/Ornith-1.5-9B-MLX-4bit",
            "-appearance", "dark",
        ]
    }

    private var ornithChromeLightArguments: [String] {
        [
            "--seed-ornith-demo-model",
            "-selectedModelID", "ornith-ai/Ornith-1.5-9B-MLX-4bit",
            "-appearance", "light",
        ]
    }

    private var selectedSSDDemoArguments: [String] {
        [
            "--seed-streamed-demo-models",
            "-selectedModelID", qatStreamedModelID,
        ]
    }

    private func launchCapture(extraArguments: [String]) {
        app?.terminate()
        app = XCUIApplication()
        app.launchArguments = [
            "--capture-release-video",
            "--capture-hide-history",
            "--sidebar-expanded",
        ] + extraArguments
        app.launch()
        XCTAssertTrue(app.wait(for: .runningForeground, timeout: 12))
        app.activate()
        resizeCaptureWindow()
        XCTAssertTrue(app.windows.firstMatch.waitForExistence(timeout: 8))
        hold(1.5)
    }

    private func openGemmaFamily() {
        let family = app.descendants(matching: .any)["familyCard.gemma4"]
        XCTAssertTrue(family.waitForExistence(timeout: 6))
        family.click()
        XCTAssertTrue(app.staticTexts["Gemma 4"].waitForExistence(timeout: 5))
    }

    private func openQATDownloadSheet() {
        let download = app.descendants(matching: .any)[
            "downloadModel.\(qatSourceModelID)"
        ]
        if !download.isHittable {
            app.scrollViews["modelsScrollView"].swipeUp()
        }
        XCTAssertTrue(download.waitForExistence(timeout: 5))
        XCTAssertTrue(download.isHittable)
        download.click()
        XCTAssertTrue(app.staticTexts["Gemma 4 26B A4B QAT"].waitForExistence(timeout: 5))
    }

    private func captureWindow(named name: String) {
        let window = app.windows.firstMatch
        XCTAssertTrue(window.exists)
        window.coordinate(withNormalizedOffset: CGVector(dx: 0.97, dy: 0.97)).hover()
        hold(0.8)
        let attachment = XCTAttachment(screenshot: window.screenshot())
        attachment.name = "\(name).png"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func resizeCaptureWindow() {
        let script = """
        tell application "System Events"
          set procs to every process whose bundle identifier is "org.mlxchat.MLXChat"
          repeat with p in procs
            try
              set frontmost of p to true
              repeat with w in windows of p
                try
                  set position of w to {80, 80}
                  set size of w to {900, 776}
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
        hold(0.5)
    }

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
