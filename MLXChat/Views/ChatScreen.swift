import AppKit
import SwiftData
import SwiftUI

/// Detail pane: welcome state or transcript, plus the composer and artifact inspector.
struct ChatScreen: View {
    @Environment(ChatController.self) private var chat
    let conversation: Conversation?
    /// Creates and selects a conversation lazily on first send.
    let onCreateConversation: () -> Conversation?

    @State private var draft = ""
    @FocusState private var composerFocused: Bool
    @State private var atBottom = true
    @State private var userPausedAutoScroll = false
    @State private var userIsScrolling = false

    var body: some View {
        let messages = conversation?.orderedMessages ?? []
        VStack(spacing: 0) {
            if messages.isEmpty {
                WelcomeView(isEnabled: !chat.isSelectedModelPreparing) { template in
                    draft = template
                    composerFocused = true
                    moveComposerCursorToEnd()
                }
            } else {
                transcript(messages: messages)
            }
            EngineReadinessNotice(selectedModelID: chat.selectedModelID)
            ComposerView(draft: $draft, focused: $composerFocused) { text, attachments in
                send(text, attachments: attachments)
            }
        }
        .background(Color.detailBackground)
        .safeAreaInset(edge: .top, spacing: 0) {
            // Fixed header: opaque, so the transcript scrolls under it
            // instead of sliding past the model picker. Window chrome lives
            // on RootView, so this row stays put when the sidebar closes.
            HStack(spacing: 8) {
                Spacer(minLength: 0)
                ModelPickerMenu()
                if chat.selectedModelSupportsThinking {
                    ThinkingToggle(canDisable: chat.selectedModelSupportsThinkingToggle)
                }
                PromptPresetMenu()
            }
            .padding(.leading, 16)
            .padding(.trailing, 20)
            .padding(.top, 16)
            .padding(.bottom, 4)
            .frame(maxWidth: .infinity)
            .background(Color.detailBackground)
        }
        .overlay(alignment: .topLeading) {
            if let toast = chat.effectiveSwitchToast {
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(Color.brandGreen)
                    Text(toast)
                        .font(Studio.sans(14, weight: .semibold))
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Capsule().fill(Color.cardFill))
                .overlay(Capsule().strokeBorder(Color.cardStroke, lineWidth: 1))
                .shadow(color: .black.opacity(0.12), radius: 8, y: 2)
                .padding(.leading, 24)
                .padding(.top, 8)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.easeOut(duration: 0.25), value: chat.effectiveSwitchToast)
        .onChange(of: conversation?.id) {
            atBottom = true
            userPausedAutoScroll = false
            userIsScrolling = false
            draft = ""
        }
        .onChange(of: chat.streamingMessageID) {
            // Preserve an explicit user pause. A newly appended assistant row
            // can make geometry report "not at bottom" for one layout pass;
            // treating that transient state as intent is what caused bouncing.
            if atBottom { userPausedAutoScroll = false }
        }
    }

    private func transcriptFingerprint(
        for messages: [ChatMessage]
    ) -> TranscriptFingerprint {
        TranscriptFingerprint(
            messageCount: messages.count,
            lastMessageID: messages.last?.id)
    }

    private func transcript(messages: [ChatMessage]) -> some View {
        scrollBody(messages: messages)
            // Fresh scroll state per conversation: a reused ScrollView keeps the
            // previous chat's offset, which can land past the new (shorter)
            // content and show an apparently empty transcript until scrolled.
            .id(conversation?.id)
    }

    private func scrollBody(messages: [ChatMessage]) -> some View {
        let lastMessageID = messages.last?.id
        let streamingMessageID = chat.streamingMessageID
        let isAnyStreaming = chat.isStreaming
        let isInThinkPhase = chat.isInThinkPhase
        return ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    ForEach(messages) { message in
                        let isStreamingThis = streamingMessageID == message.id
                        MessageView(
                            message: message,
                            isStreamingThis: isStreamingThis,
                            isInThinkPhase: isStreamingThis && isInThinkPhase,
                            isAnyStreaming: isAnyStreaming,
                            modelDisplayName: message.modelID.map { chat.modelDisplayName(for: $0) },
                            onRegenerate: regenerateAction(
                                for: message, lastMessageID: lastMessageID))
                            .id(message.id)
                    }
                    Color.clear.frame(height: 1).id("chat-bottom")
                }
                .frame(maxWidth: 720)
                .frame(maxWidth: .infinity)
                .padding(.horizontal, 32)
                .padding(.top, 24)
                .animation(.easeOut(duration: 0.25), value: messages.count)
            }
            .accessibilityIdentifier("transcriptScrollView")
            .defaultScrollAnchor(.bottom)
            .onScrollGeometryChange(for: ScrollPinState.self) { geometry in
                ScrollPinState(
                    atBottom: Self.isNearBottom(geometry),
                    contentHeight: geometry.contentSize.height)
            } action: { old, new in
                atBottom = new.atBottom
                if new.atBottom {
                    userPausedAutoScroll = false
                } else if userIsScrolling {
                    userPausedAutoScroll = true
                }
                if old.contentHeight != new.contentHeight,
                   old.atBottom || new.atBottom
                    || (isAnyStreaming && !userPausedAutoScroll) {
                    proxy.scrollTo("chat-bottom", anchor: .bottom)
                }
            }
            .onScrollPhaseChange { _, newPhase, context in
                let isUserPhase = newPhase == .tracking
                    || newPhase == .interacting
                    || newPhase == .decelerating
                if isUserPhase {
                    userIsScrolling = true
                    if !Self.isNearBottom(context.geometry) {
                        userPausedAutoScroll = true
                    }
                } else if newPhase == .idle {
                    if userIsScrolling {
                        userPausedAutoScroll = !Self.isNearBottom(context.geometry)
                    }
                    userIsScrolling = false
                }
            }
            .onAppear {
                // History opens must show the first row. Anchoring to
                // chat-bottom after a taller previous chat can leave the
                // viewport past this transcript (empty pane until scroll).
                if let firstID = messages.first?.id {
                    proxy.scrollTo(firstID, anchor: .top)
                } else {
                    proxy.scrollTo("chat-bottom", anchor: .bottom)
                }
            }
            .onChange(of: transcriptFingerprint(for: messages)) {
                if atBottom || (chat.isStreaming && !userPausedAutoScroll) {
                    proxy.scrollTo("chat-bottom", anchor: .bottom)
                }
            }
            .background {
                ScrollPinner(
                    proxy: proxy,
                    atBottom: atBottom,
                    userPausedAutoScroll: userPausedAutoScroll)
            }
            .overlay(alignment: .bottom) {
                if !atBottom {
                    Button {
                        atBottom = true
                        userPausedAutoScroll = false
                        withAnimation { proxy.scrollTo("chat-bottom", anchor: .bottom) }
                    } label: {
                        Image(systemName: "arrow.down")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 28, height: 28)
                            .studioQuietFill(in: Circle())
                    }
                    .buttonStyle(.plain)
                    .padding(.bottom, 10)
                }
            }
        }
    }

    private static func isNearBottom(_ geometry: ScrollGeometry) -> Bool {
        geometry.contentOffset.y + geometry.containerSize.height
            >= geometry.contentSize.height - 60
    }

    /// Programmatic focus selects the whole TextField content; collapse the
    /// selection to a caret at the end so the template isn't highlighted.
    private func moveComposerCursorToEnd() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) {
            if let editor = NSApp.keyWindow?.firstResponder as? NSTextView {
                editor.selectedRange = NSRange(location: (editor.string as NSString).length, length: 0)
            }
        }
    }

    private func send(_ text: String, attachments: ComposerAttachments = ComposerAttachments()) {
        guard !chat.isStreaming else { return }
        guard let target = conversation ?? onCreateConversation() else { return }
        chat.send(
            text,
            in: target,
            image: attachments.imageData,
            attachments: attachments.files)
    }

    private func regenerateAction(
        for message: ChatMessage,
        lastMessageID: UUID?
    ) -> (() -> Void)? {
        guard let conversation,
              message.id == lastMessageID,
              !message.isUser
        else { return nil }
        return { chat.regenerate(message, in: conversation) }
    }
}

private struct TranscriptFingerprint: Equatable {
    let messageCount: Int
    let lastMessageID: UUID?
}

private struct ScrollPinState: Equatable {
    let atBottom: Bool
    let contentHeight: CGFloat
}

/// Owns `transcriptRevision` so ChatScreen's body does not depend on it.
/// Flushes still pin to the bottom under the existing pause rules.
private struct ScrollPinner: View {
    @Environment(ChatController.self) private var chat
    let proxy: ScrollViewProxy
    let atBottom: Bool
    let userPausedAutoScroll: Bool

    var body: some View {
        Color.clear
            .accessibilityHidden(true)
            .onChange(of: chat.transcriptRevision) {
                if atBottom || (chat.isStreaming && !userPausedAutoScroll) {
                    proxy.scrollTo("chat-bottom", anchor: .bottom)
                }
            }
    }
}

/// Warns BEFORE sending when the selected engine cannot serve yet — on a
/// fresh Mac, macOS may still be downloading the Apple Intelligence model.
/// Re-checks periodically so the notice clears on its own once ready.
private struct EngineReadinessNotice: View {
    let selectedModelID: String

    var body: some View {
        TimelineView(.periodic(from: .now, by: 10)) { _ in
            if let message = notice {
                HStack(spacing: 8) {
                    Image(systemName: "hourglass")
                        .foregroundStyle(Color.iconOrange)
                    Text(message)
                        .font(Studio.sans(12.5))
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Capsule().fill(Color.cardFill))
                .overlay(Capsule().strokeBorder(Color.cardStroke, lineWidth: 1))
                .padding(.bottom, 6)
            }
        }
    }

    private var notice: String? {
        // UI-verification hook: preview the notice on Macs where Apple
        // Intelligence is already ready.
        if ProcessInfo.processInfo.arguments.contains("--fake-ai-preparing") {
            return "Apple Intelligence is still preparing. Try again shortly."
        }
        guard selectedModelID == appleIntelligenceEngineID else { return nil }
        return AppleIntelligenceEngine.availabilityMessage()
    }
}

/// Bulb toggle for thinking-capable models: lit purple when the model may
/// think, slashed gray when thinking is switched off.
private struct ThinkingToggle: View {
    let canDisable: Bool
    @AppStorage(Keys.thinkingEnabled) private var enabled = Keys.Defaults.thinkingEnabled

    private var isEnabled: Bool { canDisable ? enabled : true }

    @ViewBuilder
    var body: some View {
        if canDisable {
            Button {
                enabled.toggle()
            } label: {
                bulbIcon()
            }
            .buttonStyle(.plain)
            .help(helpText)
            .accessibilityLabel("Thinking")
        } else {
            bulbIcon()
                .help(helpText)
                .accessibilityLabel("Thinking is always on")
        }
    }

    private func bulbIcon() -> some View {
        Image(systemName: isEnabled ? "lightbulb.fill" : "lightbulb.slash")
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(isEnabled ? Color.iconPurple : .secondary)
            .studioHeaderIconChip()
    }

    private var helpText: String {
        guard canDisable else { return "Thinking is always on for this model" }
        return enabled
            ? "Thinking is on — click to turn off"
            : "Thinking is off — click to turn on"
    }
}
