import AppKit
import MarkdownUI
import SwaTex
import SwaTexRender
import SwiftUI

/// Renders LaTeX to NSImage via SwaTex, with caching and a never-blank fallback.
enum MathRenderer {
    static let cacheCountLimit = 512

    private static let cache: NSCache<NSString, NSImage> = {
        let cache = NSCache<NSString, NSImage>()
        cache.countLimit = cacheCountLimit
        return cache
    }()
    private static let retinaScale: CGFloat = 2

    static func cacheKey(
        base64URL encoded: String,
        display: Bool,
        fontSize: CGFloat,
        color: NSColor
    ) -> String {
        let ink = swatexColor(from: color)
        let r = Int((ink.r * 255).rounded())
        let g = Int((ink.g * 255).rounded())
        let b = Int((ink.b * 255).rounded())
        let a = Int((ink.a * 255).rounded())
        let sizeTenths = Int((fontSize * 10).rounded())
        return "\(display ? "d" : "i")|\(sizeTenths)|\(r),\(g),\(b),\(a)|\(encoded)"
    }

    static func render(base64URL encoded: String, display: Bool, fontSize: CGFloat, color: NSColor) -> NSImage {
        let key = cacheKey(
            base64URL: encoded,
            display: display,
            fontSize: fontSize,
            color: color) as NSString
        if let cached = cache.object(forKey: key) { return cached }

        let ink = swatexColor(from: color)
        let latex = MathPreprocessor.decodeBase64URL(encoded) ?? ""
        let options = RenderOptions(fontSize: fontSize, padding: display ? 6 : 1)
        let style: MathStyle = display ? .display : .text

        let final: NSImage
        if let cgImage = try? ImageRenderer.image(
            latex: latex,
            style: style,
            color: ink,
            options: options,
            displayScale: retinaScale
        ) {
            let pointSize = NSSize(
                width: CGFloat(cgImage.width) / retinaScale,
                height: CGFloat(cgImage.height) / retinaScale)
            let image = NSImage(cgImage: cgImage, size: pointSize)
            final = image
        } else {
            final = literalImage(text: latex.isEmpty ? "math" : latex, color: color)
        }
        cache.setObject(final, forKey: key)
        return final
    }

    static func swatexColor(from color: NSColor) -> SwaTex.Color {
        let rgb = color.usingColorSpace(.sRGB)
            ?? NSColor(srgbRed: 0, green: 0, blue: 0, alpha: 1)
        return SwaTex.Color(
            r: Float(rgb.redComponent),
            g: Float(rgb.greenComponent),
            b: Float(rgb.blueComponent),
            a: Float(rgb.alphaComponent))
    }

    static func literalImage(text: String, color: NSColor) -> NSImage {
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

    func image(with url: URL, label: String) async throws -> Image {
        guard url.scheme == "swiftmath" else {
            let fallbackText = label.isEmpty ? url.absoluteString : label
            return Image(nsImage: MathRenderer.literalImage(text: fallbackText, color: color))
        }
        let encoded = String(url.path.dropFirst())
        let nsImage = MathRenderer.render(
            base64URL: encoded,
            display: (url.host ?? "") == "d",
            fontSize: fontSize,
            color: color)
        return Image(nsImage: nsImage)
    }
}

/// Handles `swiftmath://d/<b64url>` block images; other URLs stay as links.
struct MathBlockProvider: ImageProvider {
    let fontSize: CGFloat
    let color: NSColor

    @ViewBuilder
    func makeImage(url: URL?) -> some View {
        if let url, url.scheme == "swiftmath" {
            let nsImage = MathRenderer.render(
                base64URL: String(url.path.dropFirst()),
                display: (url.host ?? "") == "d",
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
