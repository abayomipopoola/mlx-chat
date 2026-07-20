import AppKit
import SwiftUI

/// Payload handed from the composer to ChatScreen on send.
struct ComposerAttachments {
    var imageData: Data? = nil
    var attachment: (name: String, text: String)? = nil

    var isEmpty: Bool { imageData == nil && attachment == nil }
}

/// Bottom input area: engine status rows, error banner, growing text field, send/stop.
struct ComposerView: View {
    @Environment(ChatController.self) private var chat
    @Binding var draft: String
    @FocusState.Binding var focused: Bool
    let onSend: (String, ComposerAttachments) -> Void

    @State private var pendingImageData: Data?
    @State private var pendingImagePreview: NSImage?
    @State private var pendingFileName: String?
    @State private var pendingFileText: String?

    private var hasPendingAttachment: Bool {
        pendingImageData != nil || pendingFileName != nil
    }

    /// Non-allocating draft check (no intermediate trimmed String).
    private var draftHasContent: Bool {
        draft.contains { !$0.isWhitespace }
    }

    private var canSend: Bool {
        (draftHasContent || hasPendingAttachment) && !chat.isStreaming
    }

    /// UI-verification hook: `--fake-vision-support` forces the attach menu on.
    private var showsAttachMenu: Bool {
        ProcessInfo.processInfo.arguments.contains("--fake-vision-support")
            || chat.selectedModelSupportsVision
    }

    var body: some View {
        // Evaluate once per body pass (send button reads this three times).
        let canSendNow = canSend

        VStack(spacing: 10) {
            // Group status glass strips with the input so Liquid Glass can blend them.
            GlassEffectContainer {
                VStack(spacing: 10) {
                    ComposerStatusRows()

                    VStack(alignment: .leading, spacing: 8) {
                        if hasPendingAttachment {
                            PendingAttachmentStrip(
                                pendingImagePreview: pendingImagePreview,
                                pendingFileName: pendingFileName,
                                onClearImage: clearPendingImage,
                                onClearFile: clearPendingFile)
                                .padding(.horizontal, 18)
                                .padding(.top, 10)
                        }

                        HStack(alignment: .bottom, spacing: 10) {
                            if showsAttachMenu {
                                attachMenu
                            }

                            TextField("What can I help you with?", text: $draft, axis: .vertical)
                                .textFieldStyle(.plain)
                                .lineLimit(1...8)
                                .font(Studio.composerText)
                                .padding(.vertical, 11)
                                .focused($focused)
                                .accessibilityIdentifier("composerField")
                                .onKeyPress(.return, phases: .down) { press in
                                    if press.modifiers.contains(.shift) {
                                        draft += "\n"
                                        return .handled
                                    }
                                    submit()
                                    return .handled
                                }

                            Button(action: sendOrStop) {
                                Image(systemName: chat.isStreaming ? "stop.fill" : "arrow.up")
                                    .font(.system(size: 15, weight: .semibold))
                                    .foregroundStyle(chat.isStreaming || canSendNow ? .white : Color.secondary)
                                    .frame(width: Studio.sendButtonSize, height: Studio.sendButtonSize)
                                    .background(
                                        Circle().fill(
                                            chat.isStreaming || canSendNow
                                                ? Color.sendGreen
                                                : Color.cardStroke.opacity(0.6)))
                            }
                            .buttonStyle(.plain)
                            .disabled(!chat.isStreaming && !canSendNow)
                            .accessibilityIdentifier("composerSendButton")
                        }
                        .padding(.leading, showsAttachMenu ? 10 : 24)
                        .padding(.trailing, 6)
                        .padding(.vertical, 6)
                    }
                    .glassEffect(.regular, in: RoundedRectangle(cornerRadius: Studio.radiusComposer, style: .continuous))
                }
            }

            Text("Runs 100% on-device • Private & offline")
                .font(.system(size: 11.5))
                .foregroundStyle(Color.subtitleGray)
        }
        // Same centered column as the transcript (720 + 32pt gutters), so the
        // composer keeps the content width when the sidebar is collapsed.
        .frame(maxWidth: 720)
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 32)
        .padding(.bottom, 12)
        .padding(.top, 6)
        .onAppear {
            // UI-verification hook: seed a pending file chip without NSOpenPanel.
            if ProcessInfo.processInfo.arguments.contains("--seed-attachment") {
                pendingFileName = "notes.txt"
                pendingFileText = "hello"
            }
            // UI-verification hook: force ErrorBannerRow to render.
            if ProcessInfo.processInfo.arguments.contains("--seed-error-banner") {
                chat.errorBanner = "Seeded test banner: model failed to load."
            }
        }
        .onChange(of: chat.selectedModelID) {
            clearPending()
        }
    }

    // MARK: - Attach

    private var attachMenu: some View {
        Menu {
            Button("Photo or Image…") { pickImage() }
            Button("File…") { pickFile() }
        } label: {
            Image(systemName: "plus")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 32, height: 32)
                .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .frame(width: 32, height: Studio.sendButtonSize)
        .help("Attach")
        .accessibilityLabel("Attach")
        .accessibilityIdentifier("composerAttachButton")
        .accessibilityAddTraits(.isButton)
    }

    private func sendOrStop() {
        if chat.isStreaming {
            chat.stop()
        } else {
            submit()
        }
    }

    private func pickImage() {
        switch AttachmentPicker.pickImage() {
        case .cancelled:
            return
        case .loadFailed:
            chat.errorBanner = "Couldn't load that image."
        case .picked(let data, let preview):
            pendingImageData = data
            pendingImagePreview = preview
        }
    }

    private func pickFile() {
        switch AttachmentPicker.pickFile() {
        case .cancelled:
            return
        case .failed(let message):
            chat.errorBanner = message
        case .picked(let name, let text):
            pendingFileName = name
            pendingFileText = text
        }
    }

    private func submit() {
        guard canSend else { return }
        let text = draft
        draft = ""
        let attachments = ComposerAttachments(
            imageData: pendingImageData,
            attachment: pendingFileName.flatMap { name in
                guard let body = pendingFileText else { return nil }
                return (name: name, text: body)
            })
        clearPending()
        onSend(text, attachments)
    }

    private func clearPending() {
        clearPendingImage()
        clearPendingFile()
    }

    private func clearPendingImage() {
        pendingImageData = nil
        pendingImagePreview = nil
    }

    private func clearPendingFile() {
        pendingFileName = nil
        pendingFileText = nil
    }
}
