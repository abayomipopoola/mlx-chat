import AppKit
import MarkdownUI
import SwiftUI

struct MessageView: View {
    @Environment(\.colorScheme) private var colorScheme

    let message: ChatMessage
    let isStreamingThis: Bool
    let isInThinkPhase: Bool
    let isAnyStreaming: Bool
    let modelDisplayName: String?
    let onRegenerate: (() -> Void)?

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

                    ForEach(message.attachments, id: \.name) { file in
                        HStack(spacing: 6) {
                            Image(systemName: "doc.fill")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(Color.iconBlue)
                            Text(file.name)
                                .font(Studio.sans(12.5))
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
                        Markdown(MarkdownContent(MathPreprocessor.preprocess(message.content)))
                            .markdownTheme(.chatUser)
                            .markdownInlineImageProvider(
                                MathInlineProvider(fontSize: Studio.chatFontSize + Studio.monaPointBump, color: mathColor))
                            .markdownImageProvider(
                                MathBlockProvider(fontSize: 16 + Studio.monaPointBump, color: mathColor))
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

    private var assistantBody: some View {
        VStack(alignment: .leading, spacing: 8) {
            thinkingSection

            if message.content.isEmpty, isStreamingThis, !isInThinkPhase {
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
        if isStreamingThis, isInThinkPhase {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Thinking…")
                    .foregroundStyle(.secondary)
                    .font(Studio.sans(12))
            }
        } else if let thinking = message.thinking, !thinking.isEmpty {
            DisclosureGroup {
                Markdown(thinking)
                    .markdownTheme(.chat)
                    .font(Studio.sans(12))
                    .foregroundStyle(.secondary)
                    .padding(.top, 4)
            } label: {
                Text(thinkingLabel)
                    .font(Studio.sans(12))
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
        ForEach(SegmentCache.shared.segments(for: message.id, content: message.content)) { block in
            switch block {
            case .markdown(_, let text):
                streamingAssistantMarkdown(text)
            case .artifact(let artifact):
                ExpandableArtifactCard(artifact: artifact)
            }
        }
    }

    /// Preprocess every streaming flush. WelcomeView's anti-ghost pair is
    /// applied only to this markdown subview so LazyVStack insert animation
    /// on `messages.count` is left intact.
    @ViewBuilder
    private func streamingAssistantMarkdown(_ text: String) -> some View {
        let renderedText = isStreamingThis
            ? Self.preprocessStreamingFlush(text)
            : MathPreprocessor.preprocess(text)
        let markdown = Markdown(MarkdownContent(renderedText))
            .markdownTheme(.chat)
            .markdownCodeSyntaxHighlighter(HighlightrHighlighter.forScheme(colorScheme))
            .markdownInlineImageProvider(MathInlineProvider(fontSize: Studio.chatFontSize + Studio.monaPointBump, color: mathColor))
            .markdownImageProvider(MathBlockProvider(fontSize: 16 + Studio.monaPointBump, color: mathColor))
            .textSelection(.enabled)
        if isStreamingThis {
            markdown
                .geometryGroup()
                .transaction { $0.animation = nil }
        } else {
            markdown
        }
    }

    /// Hold back a trailing odd `$` so a half-open `$…` span is not fed to
    /// the preprocessor as if it were closed. The raw suffix stays visible
    /// until the closer arrives on a later flush.
    private static func preprocessStreamingFlush(_ text: String) -> String {
        let dollarCount = text.reduce(into: 0) { count, character in
            if character == "$" { count += 1 }
        }
        guard !dollarCount.isMultiple(of: 2),
              let lastDollar = text.lastIndex(of: "$")
        else {
            return MathPreprocessor.preprocess(text)
        }
        let stable = String(text[..<lastDollar])
        let held = String(text[lastDollar...])
        return MathPreprocessor.preprocess(stable) + held
    }

    // MARK: - Footer (tok/s + hover actions)

    /// Always-visible icon actions under assistant messages (MLX Studio pattern) —
    /// stable layout, no hover-driven insertion, so nothing flickers.
    @ViewBuilder
    private var footer: some View {
        if !isStreamingThis {
            HStack(spacing: 16) {
                CopyButton(text: message.content, size: 11)

                if let onRegenerate {
                    Button {
                        if !isAnyStreaming {
                            onRegenerate()
                        }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                            .font(.system(size: 11))
                            .frame(width: 16, height: 16)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .disabled(isAnyStreaming)
                    .help("Regenerate")
                }

                if let tps = message.tokensPerSecond {
                    Text(String(format: "%.1f tok/s", tps))
                        .font(Studio.sans(10))
                        .foregroundStyle(.tertiary)
                }

                if let modelDisplayName {
                    Text(modelDisplayName)
                        .font(Studio.sans(10))
                        .foregroundStyle(.tertiary)
                        .accessibilityIdentifier("responseModelName")
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
