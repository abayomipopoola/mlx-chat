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
    @Environment(HeaderDropdown.self) private var dropdown
    @Binding var draft: String
    @FocusState.Binding var focused: Bool
    let onSend: (String, ComposerAttachments) -> Void

    @State private var pendingImageData: Data?
    @State private var pendingImagePreview: NSImage?
    @State private var pendingFileName: String?
    @State private var pendingFileText: String?
    @State private var attachHovered = false

    private var hasPendingAttachment: Bool {
        pendingImageData != nil || pendingFileName != nil
    }

    /// Non-allocating draft check (no intermediate trimmed String).
    private var draftHasContent: Bool {
        draft.contains { !$0.isWhitespace }
    }

    private var isPreparing: Bool { chat.isSelectedModelPreparing }

    private var canSend: Bool {
        (draftHasContent || hasPendingAttachment)
            && !chat.isStreaming
            && !isPreparing
    }

    private var sendHelp: String {
        if isPreparing { return "Wait until the model finishes loading" }
        if chat.isStreaming { return "Stop generating" }
        return "Send"
    }

    /// Photo and file attach both need a vision model.
    private var showsAttachMenu: Bool {
        ProcessInfo.processInfo.arguments.contains("--fake-vision-support")
            || chat.selectedModelSupportsVision
    }

    var body: some View {
        // Evaluate once per body pass (send button reads this three times).
        let canSendNow = canSend
        let composerShape = RoundedRectangle(
            cornerRadius: Studio.radiusComposer, style: .continuous)

        VStack(spacing: 8) {
            ComposerStatusRows()

            VStack(alignment: .leading, spacing: 0) {
                if hasPendingAttachment {
                    PendingAttachmentStrip(
                        pendingImagePreview: pendingImagePreview,
                        pendingFileName: pendingFileName,
                        onClearImage: clearPendingImage,
                        onClearFile: clearPendingFile)
                        .padding(.horizontal, 14)
                        .padding(.top, 8)
                }

                ComposerDraftField(
                    draft: $draft,
                    focused: $focused,
                    isEnabled: !isPreparing
                ) {
                    submit()
                }
                .padding(.horizontal, 14)
                .padding(.top, 12)

                HStack(alignment: .center, spacing: 8) {
                    if showsAttachMenu {
                        attachMenu
                    }
                    Spacer(minLength: 0)
                    Button(action: sendOrStop) {
                        composerCircleLabel(
                            systemName: chat.isStreaming ? "stop.fill" : "arrow.up",
                            fill: chat.isStreaming || canSendNow
                                ? Color.sendGreen : Color.composerButtonFill,
                            foreground: chat.isStreaming || canSendNow ? .white : .secondary)
                    }
                    .buttonStyle(.plain)
                    .pointingHandCursor(enabled: chat.isStreaming || canSendNow)
                    .disabled(!chat.isStreaming && !canSendNow)
                    .accessibilityIdentifier("composerSendButton")
                    .help(sendHelp)
                }
                .padding(.leading, 10)
                .padding(.trailing, 10)
                .padding(.bottom, 8)
            }
            .clipShape(composerShape)
            .studioQuietFill(in: composerShape)

            Text("Runs 100% on-device • Private & offline")
                .font(Studio.sans(11))
                .foregroundStyle(Color.subtitleGray)
                .accessibilityIdentifier("composerPrivacyCaption")
        }
        // Same centered column as the transcript (720 + 32pt gutters), so the
        // composer keeps the content width when the sidebar is collapsed.
        .frame(maxWidth: 720)
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 32)
        .padding(.bottom, 12)
        .padding(.top, 6)
        .onAppear {
            dropdown.attachPhoto = { pickImage() }
            dropdown.attachFile = { pickFile() }
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
        .onChange(of: isPreparing) { _, preparing in
            if preparing { focused = false }
        }
        .onDisappear {
            dropdown.attachPhoto = nil
            dropdown.attachFile = nil
            if dropdown.open == .attach { dropdown.open = nil }
        }
    }

    // MARK: - Attach

    private var attachMenu: some View {
        Button {
            dropdown.toggle(.attach)
        } label: {
            Image(systemName: "plus")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(width: Studio.sendButtonSize, height: Studio.sendButtonSize)
                .background(
                    Circle().fill(Color.composerButtonFill)
                        .opacity(attachHovered || dropdown.open == .attach ? 1 : 0))
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .frame(width: Studio.sendButtonSize, height: Studio.sendButtonSize)
        .onHover { attachHovered = $0 }
        .pointingHandCursor(enabled: !isPreparing)
        .help("Attach")
        .accessibilityLabel("Attach")
        .accessibilityIdentifier("composerAttachButton")
        .dropdownAnchor(.attach, in: dropdown)
        .disabled(isPreparing)
    }

    private func composerCircleLabel(
        systemName: String,
        fill: Color,
        foreground: Color
    ) -> some View {
        Image(systemName: systemName)
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(foreground)
            .frame(width: Studio.sendButtonSize, height: Studio.sendButtonSize)
            .background(Circle().fill(fill))
            .contentShape(Circle())
    }

    private func sendOrStop() {
        if dropdown.open == .attach { dropdown.open = nil }
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

/// In-window attach popup: title plus icon rows, matching the header dropdowns.
struct AttachPanel: View {
    @Environment(HeaderDropdown.self) private var dropdown

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Attach")
                .font(Studio.sans(13, weight: .medium))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 8)
                .padding(.bottom, 4)

            attachRow(title: "Photo or Image…", systemImage: "photo") {
                dropdown.open = nil
                dropdown.attachPhoto?()
            }
            attachRow(title: "File…", systemImage: "doc") {
                dropdown.open = nil
                dropdown.attachFile?()
            }
        }
        .padding(8)
    }

    private func attachRow(
        title: String,
        systemImage: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: systemImage)
                    .font(.system(size: 13, weight: .regular))
                    .frame(width: 16)
                Text(title)
                    .font(Studio.sans(13))
                Spacer(minLength: 0)
            }
            .foregroundStyle(.secondary)
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(QuietHoverFillButtonStyle(cornerRadius: 8))
        .pointingHandCursor()
    }
}

/// Grows from one line to eight, then scrolls so a long paste stays reachable.
private struct ComposerDraftField: View {
    @Binding var draft: String
    @FocusState.Binding var focused: Bool
    var isEnabled: Bool = true
    let onSubmit: () -> Void

    private static let minHeight: CGFloat = 22
    private static let maxHeight: CGFloat = 176
    @State private var textHeight: CGFloat = Self.minHeight

    private var fieldHeight: CGFloat {
        min(max(textHeight, Self.minHeight), Self.maxHeight)
    }

    private var needsScroll: Bool { textHeight > Self.maxHeight }

    var body: some View {
        ZStack(alignment: .topLeading) {
            TextEditor(text: $draft)
                .font(Studio.composerText)
                .scrollContentBackground(.hidden)
                .textEditorStyle(.plain)
                .scrollIndicators(needsScroll ? .visible : .hidden)
                .focused($focused)
                .disabled(!isEnabled)
                .opacity(isEnabled ? 1 : 0.55)
                .writingToolsBehavior(.disabled)
                .writingToolsAffordanceVisibility(.hidden)
                .accessibilityIdentifier("composerField")
                .accessibilityLabel("What can I help you with?")
                .onKeyPress(.return, phases: .down) { press in
                    if !isEnabled { return .handled }
                    if press.modifiers.contains(.shift) {
                        return .ignored
                    }
                    onSubmit()
                    return .handled
                }
                .background(alignment: .top) {
                    Text(draft.isEmpty ? " " : draft)
                        .font(Studio.composerText)
                        .padding(.horizontal, 5)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)
                        .hidden()
                        .accessibilityHidden(true)
                        .onGeometryChange(for: CGFloat.self) { proxy in
                            proxy.size.height
                        } action: { textHeight = $0 }
                }
                .background {
                    ComposerTextScrollChrome(showScroller: needsScroll)
                }

            if draft.isEmpty {
                Text(isEnabled ? "What can I help you with?" : "Loading model…")
                    .font(Studio.composerText)
                    .foregroundStyle(.tertiary)
                    .padding(.leading, 5)
                    .allowsHitTesting(false)
            }
        }
        .frame(height: fieldHeight, alignment: .top)
    }
}

/// Pins the AppKit text scroll view so overflow actually scrolls and the
/// overlay scroller stays on while the draft is taller than the composer.
private struct ComposerTextScrollChrome: NSViewRepresentable {
    var showScroller: Bool

    func makeNSView(context: Context) -> NSView {
        NSView()
    }

    func updateNSView(_ view: NSView, context: Context) {
        DispatchQueue.main.async { apply(from: view) }
    }

    private func apply(from view: NSView) {
        guard let scrollView = enclosingTextScrollView(from: view) else { return }
        if let textView = scrollView.documentView as? NSTextView {
            textView.isVerticallyResizable = true
            textView.isHorizontallyResizable = false
            textView.isRichText = false
            textView.importsGraphics = false
            textView.writingToolsBehavior = .none
            textView.textContainer?.widthTracksTextView = true
            textView.textContainer?.containerSize = NSSize(
                width: max(scrollView.contentSize.width, 1),
                height: CGFloat.greatestFiniteMagnitude)
        }
        scrollView.hasHorizontalScroller = false
        scrollView.hasVerticalScroller = showScroller
        scrollView.autohidesScrollers = false
        scrollView.scrollerStyle = .overlay
        OverlayScrollers.applyThinVerticalScroller(to: scrollView)
        scrollView.drawsBackground = false
        if showScroller {
            scrollView.flashScrollers()
        }
    }

    private func enclosingTextScrollView(from view: NSView) -> NSScrollView? {
        var node: NSView? = view.superview
        while let current = node {
            if let found = textScrollView(near: current) { return found }
            node = current.superview
        }
        return nil
    }

    private func textScrollView(near view: NSView) -> NSScrollView? {
        if let scrollView = view as? NSScrollView, scrollView.documentView is NSTextView {
            return scrollView
        }
        for sub in view.subviews {
            if let scrollView = sub as? NSScrollView, scrollView.documentView is NSTextView {
                return scrollView
            }
            for inner in sub.subviews {
                if let scrollView = inner as? NSScrollView, scrollView.documentView is NSTextView {
                    return scrollView
                }
            }
        }
        return nil
    }
}
