import AppKit
import MarkdownUI
import SwiftMath
import SwiftUI

/// Renders LaTeX to NSImage via SwiftMath, with caching and a never-blank fallback.
enum MathRenderer {
    private static let cache = NSCache<NSString, NSImage>()

    static func render(base64URL encoded: String, display: Bool, fontSize: CGFloat, color: NSColor) -> NSImage {
        let key = "\(display ? "d" : "i")|\(fontSize)|\(color.description)|\(encoded)" as NSString
        if let cached = cache.object(forKey: key) { return cached }

        let latex = MathPreprocessor.decodeBase64URL(encoded) ?? ""
        // Text style even for display blocks: operators like Σ stay in scale
        // with their operands instead of towering over them. (SwiftMath only
        // honors \limits in display style, so side-set limits come with this.)
        var renderer = MathImage(
            latex: latex,
            fontSize: fontSize,
            textColor: color,
            labelMode: .text,
            textAlignment: .center)
        let (error, image, _) = renderer.asImage()

        let final: NSImage
        if error == nil, let image {
            final = image
        } else {
            // Invalid LaTeX: show the literal source rather than nothing.
            final = literalImage(text: latex.isEmpty ? "math" : latex, color: color)
        }
        cache.setObject(final, forKey: key)
        return final
    }

    private static func literalImage(text: String, color: NSColor) -> NSImage {
        let attributed = NSAttributedString(string: text, attributes: [
            .font: NSFont.monospacedSystemFont(ofSize: 12, weight: .regular),
            .foregroundColor: color,
        ])
        var size = attributed.size()
        size.width = max(size.width, 4)
        size.height = max(size.height, 4)
        return NSImage(size: size, flipped: false) { rect in
            attributed.draw(in: rect)
            return true
        }
    }
}

/// Handles `swiftmath://i/<b64url>` inside lines of text.
struct MathInlineProvider: InlineImageProvider {
    let fontSize: CGFloat
    let color: NSColor

    struct UnsupportedURL: Error {}

    func image(with url: URL, label: String) async throws -> Image {
        guard url.scheme == "swiftmath" else { throw UnsupportedURL() }
        let encoded = String(url.path.dropFirst())  // strip leading "/"
        let display = url.host == "d"
        let nsImage = MathRenderer.render(
            base64URL: encoded, display: display, fontSize: fontSize, color: color)
        return Image(nsImage: nsImage)
    }
}

/// Handles `swiftmath://d/<b64url>` block images; renders other URLs as plain links
/// (never fetches the network — privacy).
struct MathBlockProvider: ImageProvider {
    let fontSize: CGFloat
    let color: NSColor

    @ViewBuilder
    func makeImage(url: URL?) -> some View {
        if let url, url.scheme == "swiftmath" {
            let nsImage = MathRenderer.render(
                base64URL: String(url.path.dropFirst()),
                display: url.host == "d",
                fontSize: fontSize,
                color: color)
            if nsImage.size.width > 720 {
                ScrollView(.horizontal, showsIndicators: true) {
                    Image(nsImage: nsImage).padding(.vertical, 4)
                }
            } else {
                Image(nsImage: nsImage)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 4)
            }
        } else if let url {
            Link(url.absoluteString, destination: url)
        } else {
            EmptyView()
        }
    }
}
