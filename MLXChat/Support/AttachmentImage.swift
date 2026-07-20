import AppKit
import Foundation

/// Prepares user-selected images as downscaled JPEG payloads for VLM turns.
enum AttachmentImage {
    /// Loads an image from `url`, downscales so the longest side is at most
    /// `maxDimension` (never upscales), and returns JPEG data at `quality`.
    /// Returns nil on any load or encode failure.
    static func jpegData(
        from url: URL,
        maxDimension: CGFloat = 1024,
        quality: Double = 0.8
    ) -> Data? {
        let accessed = url.startAccessingSecurityScopedResource()
        defer {
            if accessed { url.stopAccessingSecurityScopedResource() }
        }

        guard let image = NSImage(contentsOf: url) else { return nil }
        return jpegData(from: image, maxDimension: maxDimension, quality: quality)
    }

    /// Same pipeline for an already-loaded `NSImage` (used by tests).
    static func jpegData(
        from image: NSImage,
        maxDimension: CGFloat = 1024,
        quality: Double = 0.8
    ) -> Data? {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }

        let sourceWidth = CGFloat(cgImage.width)
        let sourceHeight = CGFloat(cgImage.height)
        guard sourceWidth > 0, sourceHeight > 0 else { return nil }

        let longest = max(sourceWidth, sourceHeight)
        let scale = longest > maxDimension ? maxDimension / longest : 1
        let targetWidth = max(1, Int((sourceWidth * scale).rounded(.down)))
        let targetHeight = max(1, Int((sourceHeight * scale).rounded(.down)))

        let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: targetWidth,
            pixelsHigh: targetHeight,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0)
        guard let rep else { return nil }

        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSImage(cgImage: cgImage, size: NSSize(width: sourceWidth, height: sourceHeight))
            .draw(
                in: NSRect(x: 0, y: 0, width: targetWidth, height: targetHeight),
                from: .zero,
                operation: .copy,
                fraction: 1.0)
        NSGraphicsContext.restoreGraphicsState()

        return rep.representation(
            using: .jpeg,
            properties: [.compressionFactor: NSNumber(value: quality)])
    }
}
