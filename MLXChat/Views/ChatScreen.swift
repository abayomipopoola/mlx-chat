import AppKit
import SwiftData
import SwiftUI

/// Detail pane: welcome state or transcript, plus the composer and artifact inspector.
struct ChatScreen: View {
    @Environment(ChatController.self) private var chat
    let conversation: Conversation?
    /// Creates and selects a conversation lazily on first send.
    let onCreateConversation: () -> Conversation?
    /// Pushes the in-frame Models page (picker lives in-content, not in a toolbar).
    let onManageModels: () -> Void

    @State private var draft = ""
    @FocusState private var composerFocused: Bool
    @State private var atBottom = true

    private var messages: [ChatMessage] {
        conversation?.orderedMessages ?? []
    }

    var body: some View {
        VStack(spacing: 0) {
            if messages.isEmpty {
                WelcomeView { template in
                    draft = template
                    composerFocused = true
                    moveComposerCursorToEnd()
                }
            } else {
                transcript
            }
            EngineReadinessNotice(selectedModelID: chat.selectedModelID)
            ComposerView(draft: $draft, focused: $composerFocused) { text in
                send(text)
            }
        }
        .background(Color.detailBackground)
        .safeAreaInset(edge: .top, spacing: 0) {
            // Fixed header: opaque, so the transcript scrolls under it
            // instead of sliding past the model picker.
            HStack(spacing: 8) {
                Spacer()
                ModelPickerMenu(onManageModels: onManageModels)
                if chat.selectedModelSupportsThinking {
                    ThinkingToggle()
                }
                PromptPresetMenu()
            }
            .padding(.horizontal, 20)
            .padding(.top, 16)
            .padding(.bottom, 4)
            .frame(maxWidth: .infinity)
            .background(Color.detailBackground)
        }
        .overlay(alignment: .topLeading) {
            if let toast = chat.switchToast {
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(Color.brandGreen)
                    Text(toast)
                        .font(.system(size: 14, weight: .semibold))
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
        .animation(.easeOut(duration: 0.25), value: chat.switchToast)
        .onChange(of: conversation?.id) {
            atBottom = true
            draft = ""
        }
    }

    private var transcriptFingerprint: String {
        let last = messages.last
        return "\(messages.count)|\(last?.content.count ?? 0)|\(last?.thinking?.count ?? 0)"
    }

    private var transcript: some View {
        scrollBody
            // Fresh scroll state per conversation: a reused ScrollView keeps the
            // previous chat's offset, which can land past the new (shorter)
            // content and show an apparently empty transcript until scrolled.
            .id(conversation?.id)
    }

    private var scrollBody: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 24) {
                    ForEach(messages) { message in
                        MessageView(message: message, conversation: conversation)
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
            .defaultScrollAnchor(.bottom)
            .onScrollGeometryChange(for: Bool.self) { geometry in
                geometry.contentOffset.y + geometry.containerSize.height
                    >= geometry.contentSize.height - 60
            } action: { _, nowAtBottom in
                atBottom = nowAtBottom
            }
            .onChange(of: transcriptFingerprint) {
                if atBottom {
                    proxy.scrollTo("chat-bottom", anchor: .bottom)
                }
            }
            .overlay(alignment: .bottom) {
                if !atBottom {
                    Button {
                        atBottom = true
                        withAnimation { proxy.scrollTo("chat-bottom", anchor: .bottom) }
                    } label: {
                        Image(systemName: "arrow.down")
                            .padding(8)
                            .background(.regularMaterial, in: Circle())
                            .shadow(radius: 3)
                    }
                    .buttonStyle(.plain)
                    .padding(.bottom, 10)
                }
            }
        }
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

    private func send(_ text: String) {
        guard !chat.isStreaming else { return }
        guard let target = conversation ?? onCreateConversation() else { return }
        chat.send(text, in: target)
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
                        .font(.system(size: 12.5))
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
    @AppStorage(Keys.thinkingEnabled) private var enabled = Keys.Defaults.thinkingEnabled

    var body: some View {
        Button {
            enabled.toggle()
        } label: {
            Image(systemName: enabled ? "lightbulb.fill" : "lightbulb.slash")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(enabled ? Color.iconPurple : .secondary)
                .frame(width: 34, height: 34)
                .background(Circle().fill(Color.sidebarBackground))
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .help(enabled ? "Thinking is on — click to turn off" : "Thinking is off — click to turn on")
    }
}
