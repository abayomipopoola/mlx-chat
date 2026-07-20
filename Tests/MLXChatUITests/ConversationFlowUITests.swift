import XCTest

/// Conversation lifecycle: sidebar list, search filtering, new-chat welcome,
/// and a real end-to-end send through Apple Intelligence.
final class ConversationFlowUITests: MLXUITestCase {

    func testSidebarSearchFiltersConversations() {
        let field = app.textFields["Search"]
        XCTAssertTrue(field.waitForExistence(timeout: 3))

        field.click()
        field.typeText("gradient")
        XCTAssertTrue(conversationList.buttons
            .matching(NSPredicate(format: "label CONTAINS[c] 'gradient'"))
            .firstMatch.waitForExistence(timeout: 3),
            "gradient conversations should match the search")

        field.typeKey("a", modifierFlags: .command)
        field.typeText("zzz-no-such-chat")
        XCTAssertFalse(conversationList.buttons
            .matching(NSPredicate(format: "label CONTAINS[c] 'gradient'"))
            .firstMatch.waitForExistence(timeout: 2),
            "nonsense query should filter everything out")
    }

    func testNewChatShowsWelcome() {
        app.buttons["newChatButton"].click()
        XCTAssertTrue(app.staticTexts["What can I help with?"].waitForExistence(timeout: 3),
                      "new chat should land on the welcome state")
    }

    /// ⌃N clears the selection to the welcome state without creating a
    /// conversation (ChatScreen creates one lazily on first send). Pressing
    /// again must stay on welcome and leave the sidebar list unchanged.
    /// The File-menu New Chat command keeps ⌃N alive even with the composer
    /// focused (menu key equivalents beat the TextField's emacs Ctrl+N binding).
    /// Click the sidebar list first so focus is deterministic for the synthetic key.
    func testCtrlNOpensNewChatAndStaysIdempotent() {
        XCTAssertTrue(conversationList.waitForExistence(timeout: 3))
        conversationList.click()

        let countBefore = conversationList.buttons.count

        app.typeKey("n", modifierFlags: .control)
        XCTAssertTrue(app.staticTexts["What can I help with?"].waitForExistence(timeout: 3),
                      "⌃N should land on the welcome state")

        app.typeKey("n", modifierFlags: .control)
        XCTAssertTrue(app.staticTexts["What can I help with?"].waitForExistence(timeout: 3),
                      "second ⌃N should stay on welcome (idempotent)")
        XCTAssertEqual(conversationList.buttons.count, countBefore,
                       "⌃N must not create sidebar conversations")
    }

    /// The one test that creates data: a throwaway "ping" conversation in the
    /// user's store (delete it any time). Skipped when Apple Intelligence is
    /// still preparing its model.
    func testSendMessageStreamsReply() throws {
        if app.staticTexts.matching(NSPredicate(format: "label CONTAINS[c] 'preparing'"))
            .firstMatch.exists {
            throw XCTSkip("Apple Intelligence is still preparing; try again later")
        }

        selectAppleIntelligence()

        app.buttons["newChatButton"].click()
        let composer = app.descendants(matching: .any)["composerField"]
        XCTAssertTrue(composer.waitForExistence(timeout: 3))
        // macOS: click() focuses; tap() can set value without key focus, and
        // SwiftUI .onKeyPress(.return) only fires when the field is first responder.
        composer.click()
        composer.typeText("Reply with the single word: pong")

        // The draft must arm the send button; if typing never landed, the
        // submission below would go nowhere.
        let send = app.buttons["composerSendButton"]
        let armed = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "enabled == true"), object: send)
        XCTAssertEqual(XCTWaiter.wait(for: [armed], timeout: 3), .completed,
                       "draft should enable the send button")
        // Submit via Return on the focused field (the composer's own key handler).
        // The glass send button's AX frame is unreliable for synthetic clicks;
        // app.typeKey alone often misses when the TextField lost key focus.
        composer.click()
        composer.typeKey(.return, modifierFlags: [])

        XCTAssertTrue(app.staticTexts
            .matching(NSPredicate(format: "label CONTAINS 'pong' OR value CONTAINS 'pong'"))
            .firstMatch.waitForExistence(timeout: 10),
            "user bubble should appear immediately")

        // Apple Intelligence never reports tok/s (.completed(tokensPerSecond: nil)
        // and the footer only renders when tokensPerSecond is set), so assert the
        // reply itself rendered. The user bubble starts with "Reply", so
        // BEGINSWITH 'pong' matches only the assistant's answer. Assistant blocks
        // render via MarkdownUI, which exposes copy via value, not label.
        let reply = app.staticTexts.matching(NSPredicate(
            format: "label BEGINSWITH[c] 'pong' OR value BEGINSWITH[c] 'pong'"))
        if !reply.firstMatch.waitForExistence(timeout: 120) {
            let dump = app.staticTexts.allElementsBoundByIndex.map {
                "label=\($0.label) value=\($0.value as? String ?? "")"
            }
            print("REPLY-PROBE dump:", dump)
            XCTFail("assistant reply should render")
        }
    }
}
