import AppKit
import MarkdownUI
import SwiftUI

struct MessageView: View {
    @Environment(ChatController.self) private var chat
    @Environment(\.colorScheme) private var colorScheme

    let message: ChatMessage
    let conversation: Conversation?

    var body: some View {
        if message.isUser {
            userBubble
        } else {
            assistantBody
        }
    }

    // MARK: - User

    private var userBubble: some View {
        VStack(alignment: .trailing, spacing: 6) {
            HStack {
                Spacer(minLength: 80)
                VStack(alignment: .trailing, spacing: 8) {
                    if let data = message.imageData, let nsImage = NSImage(data: data) {
                        Image(nsImage: nsImage)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: 240)
                            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    }

                    if let name = message.attachmentName {
                        HStack(spacing: 6) {
                            Image(systemName: "doc.fill")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(Color.iconBlue)
                            Text(name)
                                .font(.system(size: 12.5))
                                .lineLimit(1)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(
                            Capsule().fill(Color.fieldFill))
                        .overlay(
                            Capsule().strokeBorder(Color.cardStroke, lineWidth: 1))
                    }

                    if !message.content.isEmpty {
                        Text(message.content)
                            .font(.system(size: 15))
                            .textSelection(.enabled)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(Color.bubbleFill)
                            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    }
                }
            }
            if !message.content.isEmpty {
                CopyButton(text: message.content, size: 11)
                    .padding(.top, 4)
                    .padding(.trailing, 6)
            }
        }
    }

    // MARK: - Assistant

    private var isStreamingThis: Bool { chat.streamingMessageID == message.id }

    private var assistantBody: some View {
        VStack(alignment: .leading, spacing: 8) {
            thinkingSection

            if message.content.isEmpty, isStreamingThis, !chat.isInThinkPhase {
                WaitingDots()
            } else {
                contentBlocks
            }

            footer
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var thinkingSection: some View {
        if isStreamingThis, chat.isInThinkPhase {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Thinking…")
                    .foregroundStyle(.secondary)
                    .font(.callout)
            }
        } else if let thinking = message.thinking, !thinking.isEmpty {
            DisclosureGroup {
                Markdown(thinking)
                    .markdownTheme(.chat)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .padding(.top, 4)
            } label: {
                Text(thinkingLabel)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var thinkingLabel: String {
        if let seconds = message.thinkingSeconds, seconds >= 1 {
            return "Thought for \(Int(seconds))s"
        }
        return "Thinking"
    }

    private var mathColor: NSColor {
        colorScheme == .dark ? NSColor(white: 0.92, alpha: 1) : NSColor(white: 0.12, alpha: 1)
    }

    private var contentBlocks: some View {
        ForEach(MessageSegmenter.segment(message.content)) { block in
            switch block {
            case .markdown(_, let text):
                Markdown(MarkdownContent(MathPreprocessor.preprocess(text)))
                    .markdownTheme(.chat)
                    .markdownCodeSyntaxHighlighter(HighlightrHighlighter.forScheme(colorScheme))
                    .markdownInlineImageProvider(MathInlineProvider(fontSize: 15, color: mathColor))
                    .markdownImageProvider(MathBlockProvider(fontSize: 17, color: mathColor))
                    .textSelection(.enabled)
            case .artifact(let artifact):
                ExpandableArtifactCard(artifact: artifact)
            }
        }
    }

    // MARK: - Footer (tok/s + hover actions)

    private var isLastMessage: Bool {
        conversation?.orderedMessages.last?.id == message.id
    }

    /// Always-visible icon actions under assistant messages (MLX Studio pattern) —
    /// stable layout, no hover-driven insertion, so nothing flickers.
    @ViewBuilder
    private var footer: some View {
        if !isStreamingThis {
            HStack(spacing: 16) {
                CopyButton(text: message.content, size: 11)

                if isLastMessage, conversation != nil {
                    Button {
                        if let conversation, !chat.isStreaming {
                            chat.regenerate(message, in: conversation)
                        }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                            .font(.system(size: 11))
                            .frame(width: 16, height: 16)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .disabled(chat.isStreaming)
                    .help("Regenerate")
                }

                if let tps = message.tokensPerSecond {
                    Text(String(format: "%.1f tok/s", tps))
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.top, 2)
        }
    }
}

/// NSPasteboard copy with a 1.5s checkmark flip — shared by the assistant
/// footer and the user bubble (which uses the smaller `size`).
private struct CopyButton: View {
    let text: String
    var size: CGFloat = 14
    @State private var copied = false

    var body: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            copied = true
            Task {
                try? await Task.sleep(for: .seconds(1.5))
                copied = false
            }
        } label: {
            Image(systemName: copied ? "checkmark" : "doc.on.doc")
                .font(.system(size: size))
                .frame(width: 16, height: 16)
        }
        .buttonStyle(.plain)
        .foregroundStyle(copied ? Color.brandGreen : .secondary)
        .help("Copy")
    }
}

/// Three pulsing dots while waiting for the first token.
struct WaitingDots: View {
    @State private var phase = 0

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(.secondary)
                    .frame(width: 6, height: 6)
                    .opacity(phase == index ? 1 : 0.35)
            }
        }
        .task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(300))
                phase = (phase + 1) % 3
            }
        }
    }
}
