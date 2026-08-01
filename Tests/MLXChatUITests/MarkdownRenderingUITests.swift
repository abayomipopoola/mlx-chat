import XCTest

/// Markdown/math rendering, exercised against a freshly seeded demo
/// conversation (`--seed-demo` inserts and selects it — deterministic content:
/// code, bullets, and the \boxed{} line; one extra dupe per suite run).
final class MarkdownRenderingUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = ["--sidebar-expanded", "--seed-demo"]
        app.launch()
        XCTAssertTrue(app.buttons["modelPickerButton"].waitForExistence(timeout: 15),
                      "app should boot to the chat header")
    }

    override func tearDownWithError() throws {
        app.terminate()
    }

    func testSeededConversationRendersAllBlockTypes() {
        // Transcript is a LazyVStack in ScrollView — offscreen rows may not
        // materialize in the AX tree. Scroll to the bottom so the code block
        // and boxed lead-in line exist before asserting.
        let transcript = app.scrollViews["transcriptScrollView"]
        XCTAssertTrue(transcript.waitForExistence(timeout: 4), "transcript scroll view")
        transcript.swipeUp()
        transcript.swipeUp()

        // Heading + prose
        XCTAssertTrue(app.staticTexts["In code"].waitForExistence(timeout: 4))
        // Code block with language label and its copy affordance
        XCTAssertTrue(app.staticTexts["swift"].exists, "code language label")
        XCTAssertTrue(app.buttons["Copy"].exists, "code block copy button")

        // The \boxed line: lead-in text + inline math share one text block.
        // On macOS, SwiftUI/MarkdownUI often exposes copy via `value`, not
        // `label` (label can be empty while value holds the string).
        let leadIn = "So the whole update rule in one line:"
        let probe = transcript.staticTexts.matching(NSPredicate(
            format: "label CONTAINS %@ OR value CONTAINS %@", leadIn, leadIn))
        if !probe.firstMatch.waitForExistence(timeout: 5) {
            // Fallback: whole-app query in case scoping misses nested text.
            let appProbe = app.staticTexts.matching(NSPredicate(
                format: "label CONTAINS %@ OR value CONTAINS %@", leadIn, leadIn))
            if appProbe.firstMatch.waitForExistence(timeout: 2) {
                return
            }
            let dump = app.staticTexts.allElementsBoundByIndex.map { el in
                "label=\(el.label) value=\(el.value as? String ?? String(describing: el.value ?? ""))"
            }
            print("BOXED-PROBE dump:", dump)
            XCTFail("boxed formula should render inline with its lead-in text")
        }
    }

    func testMalformedIndentedMathRendersWithoutRawProviderLinksOrLatex() {
        app.terminate()
        app = XCUIApplication()
        app.launchArguments = ["--sidebar-expanded", "--seed-markdown-regression"]
        app.launch()

        let transcript = app.scrollViews["transcriptScrollView"]
        XCTAssertTrue(transcript.waitForExistence(timeout: 15), "transcript scroll view")

        let repairedProse = transcript.staticTexts.matching(NSPredicate(
            format: "label CONTAINS %@ OR value CONTAINS %@",
            "The first term converges to", "The first term converges to"))
        XCTAssertTrue(repairedProse.firstMatch.waitForExistence(timeout: 5),
                      "indented math prose should render as prose")

        let leakedProviderURL = transcript.staticTexts.matching(NSPredicate(
            format: "label CONTAINS %@ OR value CONTAINS %@", "swiftmath://", "swiftmath://"))
        XCTAssertFalse(leakedProviderURL.firstMatch.waitForExistence(timeout: 1),
                       "internal math-provider URLs must never be visible")

        let leakedLatex = transcript.staticTexts.matching(NSPredicate(
            format: "label CONTAINS %@ OR value CONTAINS %@", "\\frac", "\\frac"))
        XCTAssertFalse(leakedLatex.firstMatch.waitForExistence(timeout: 1),
                       "a bare indented equation should render as math, not source")
    }
}
