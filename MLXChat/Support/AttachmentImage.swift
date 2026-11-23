import AppKit
import Foundation
import ImageIO

/// Prepares user-selected images as downscaled JPEG payloads for VLM turns.
enum AttachmentImage {
    /// Loads an image from `url`, applies EXIF orientation so portrait photos
    /// stay upright, downscales so the longest side is at most `maxDimension`
    /// (never upscales), and returns JPEG data at `quality`.
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

        if let orientedCGImage = loadOrientedCGImage(from: url, maxDimension: maxDimension) {
            return jpegData(from: orientedCGImage, maxDimension: maxDimension, quality: quality)
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
        return jpegData(from: cgImage, maxDimension: maxDimension, quality: quality)
    }

    private static func loadOrientedCGImage(from url: URL, maxDimension: CGFloat) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let pixelWidth = (properties?[kCGImagePropertyPixelWidth] as? NSNumber)?.doubleValue ?? 0
        let pixelHeight = (properties?[kCGImagePropertyPixelHeight] as? NSNumber)?.doubleValue ?? 0
        let sourceLongest = max(pixelWidth, pixelHeight)
        let targetMax = sourceLongest > 0 ? min(sourceLongest, Double(maxDimension)) : Double(maxDimension)
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: max(1, Int(targetMax.rounded(.down))),
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    private static func jpegData(
        from cgImage: CGImage,
        maxDimension: CGFloat,
        quality: Double
    ) -> Data? {
        let sourceWidth = CGFloat(cgImage.width)
        let sourceHeight = CGFloat(cgImage.height)
        guard sourceWidth > 0, sourceHeight > 0 else { return nil }

        let longest = max(sourceWidth, sourceHeight)
        let scale = longest > maxDimension ? maxDimension / longest : 1
        let targetWidth = max(1, Int((sourceWidth * scale).rounded(.down)))
        let targetHeight = max(1, Int((sourceHeight * scale).rounded(.down)))

        guard let rawRep = NSBitmapImageRep(
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
        else { return nil }
        let rep = rawRep.retagging(with: .sRGB) ?? rawRep

        guard let context = NSGraphicsContext(bitmapImageRep: rep) else { return nil }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = context
        let targetRect = NSRect(x: 0, y: 0, width: targetWidth, height: targetHeight)
        NSColor(srgbRed: 1, green: 1, blue: 1, alpha: 1).setFill()
        targetRect.fill()
        NSImage(cgImage: cgImage, size: NSSize(width: sourceWidth, height: sourceHeight))
            .draw(
                in: targetRect,
                from: .zero,
                operation: .sourceOver,
                fraction: 1.0)
        NSGraphicsContext.restoreGraphicsState()

        return rep.representation(
            using: .jpeg,
            properties: [.compressionFactor: NSNumber(value: quality)])
    }
}
