import AppKit
import MarkdownUI
import SwiftUI

private struct CascadedFontStyle: TextStyle {
    let font: Font
    let nsFont: NSFont

    init(nsFont: NSFont) {
        self.nsFont = nsFont
        self.font = Font(nsFont as CTFont)
    }

    init(size: CGFloat, weight: NSFont.Weight = .regular, italic: Bool = false) {
        let base = Studio.monaNSFont(size: size, weight: weight, italic: italic)
        self.init(nsFont: Studio.cascadedNSFont(base: base, size: base.pointSize))
    }

    static func mono(size: CGFloat) -> CascadedFontStyle {
        CascadedFontStyle(nsFont: Studio.cascadedNSFont(
            base: NSFont.monospacedSystemFont(ofSize: size, weight: .regular),
            size: size
        ))
    }

    func _collectAttributes(in attributes: inout AttributeContainer) {
        // MarkdownUI's fontProperties key is module-internal, so `= nil` does
        // not type-check here. Replace the container to drop it; otherwise
        // resolvingFonts overwrites `font` with Font.system / .monospaced().
        // Set appKit.font so the cascade list survives SwiftUI Font(CTFont).
        var next = AttributeContainer()
        next.foregroundColor = attributes.foregroundColor
        next.backgroundColor = attributes.backgroundColor
        next.underlineStyle = attributes.underlineStyle
        next.strikethroughStyle = attributes.strikethroughStyle
        next.kern = attributes.kern
        next.tracking = attributes.tracking
        next.link = attributes.link
        next.baselineOffset = attributes.baselineOffset
        next.font = font
        next.appKit.font = nsFont
        attributes = next
    }
}

extension Theme {
    /// Chat theme: Mona Sans VF 14pt body (text opsz); headings stay close to body and use weight 700.
    static let chat = Theme.gitHub
        .text {
            CascadedFontStyle(size: Studio.chatBodySize)
        }
        .strong {
            CascadedFontStyle(size: Studio.chatBodySize, weight: .bold)
        }
        .emphasis {
            CascadedFontStyle(size: Studio.chatBodySize, italic: true)
        }
        .heading1 { chatHeading($0, size: 17 + Studio.monaPointBump) }
        .heading2 { chatHeading($0, size: 16 + Studio.monaPointBump) }
        .heading3 { chatHeading($0, size: 15 + Studio.monaPointBump) }
        .heading4 { chatHeading($0, size: 14 + Studio.monaPointBump) }
        .heading5 { chatHeading($0, size: 13 + Studio.monaPointBump) }
        .heading6 { chatHeading($0, size: 12 + Studio.monaPointBump) }
        .paragraph { configuration in
            configuration.label
                .fixedSize(horizontal: false, vertical: true)
                .relativeLineSpacing(.em(0.45))
                .markdownMargin(top: .zero, bottom: .em(0.8))
        }
        .codeBlock { configuration in
            ChatCodeBlockView(configuration: configuration)
        }
        .code {
            CascadedFontStyle.mono(size: 12 + Studio.monaPointBump)
            BackgroundColor(Color.fieldFill)
        }

    /// User prompts reuse chat styles, but `#` titles stay at body size.
    /// Display headings made the first line of a pasted question look like a page title.
    static let chatUser = Theme.chat
        .heading1 { chatUserHeading($0) }
        .heading2 { chatUserHeading($0) }
        .heading3 { chatUserHeading($0) }
        .heading4 { chatUserHeading($0) }
        .heading5 { chatUserHeading($0) }
        .heading6 { chatUserHeading($0) }
}

@ViewBuilder
private func chatHeading(_ configuration: BlockConfiguration, size: CGFloat) -> some View {
    configuration.label
        .fixedSize(horizontal: false, vertical: true)
        .relativeLineSpacing(.em(0.125))
        .markdownMargin(top: .em(1.0), bottom: .em(0.5))
        .markdownTextStyle {
            CascadedFontStyle(size: size, weight: .bold)
        }
}

@ViewBuilder
private func chatUserHeading(_ configuration: BlockConfiguration) -> some View {
    configuration.label
        .fixedSize(horizontal: false, vertical: true)
        .relativeLineSpacing(.em(0.45))
        .markdownMargin(top: .zero, bottom: .em(0.8))
        .markdownTextStyle {
            CascadedFontStyle(size: Studio.chatBodySize, weight: .bold)
        }
}

struct ChatCodeBlockView: View {
    let configuration: CodeBlockConfiguration
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(configuration.language?.lowercased() ?? "code")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(configuration.content, forType: .string)
                    copied = true
                    Task {
                        try? await Task.sleep(for: .seconds(1.5))
                        copied = false
                    }
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                        .font(Studio.sans(10))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)

            Divider()

            ScrollView(.horizontal, showsIndicators: false) {
                configuration.label
                    .fixedSize(horizontal: false, vertical: true)
                    .relativeLineSpacing(.em(0.2))
                    .markdownTextStyle {
                        CascadedFontStyle.mono(size: 12 + Studio.monaPointBump)
                    }
                    .padding(12)
            }
        }
        .background(Color.fieldFill.opacity(0.5))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.cardStroke, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .markdownMargin(top: .em(0.8), bottom: .em(0.8))
    }
}
