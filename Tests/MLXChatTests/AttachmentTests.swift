import AppKit
import Foundation
import PDFKit
import Testing
@testable import MLXChat

/// Contract: file extraction, image downscale, and file-block prompt composition.
@Suite struct AttachmentTests {

    // MARK: - FileTextExtractor

    @Test func textFileRoundTrips() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-extract-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: url) }

        let body = "Hello attachments\nsecond line"
        try body.write(to: url, atomically: true, encoding: .utf8)

        let extracted = try FileTextExtractor.extract(url: url)
        #expect(extracted == body)
    }

    @Test func overlongTextIsTruncatedWithMarker() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-extract-\(UUID().uuidString).md")
        defer { try? FileManager.default.removeItem(at: url) }

        let body = String(repeating: "a", count: FileTextExtractor.maxCharacters + 500)
        try body.write(to: url, atomically: true, encoding: .utf8)

        let extracted = try FileTextExtractor.extract(url: url)
        #expect(extracted.hasSuffix("\n…[truncated]"))
        #expect(extracted.count == FileTextExtractor.maxCharacters + "\n…[truncated]".count)
        #expect(extracted.dropLast("\n…[truncated]".count).allSatisfy { $0 == "a" })
    }

    @Test func unsupportedExtensionThrows() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-extract-\(UUID().uuidString).zip")
        defer { try? FileManager.default.removeItem(at: url) }

        try Data([0x50, 0x4B, 0x03, 0x04]).write(to: url)

        #expect(throws: FileTextExtractor.ExtractError.unsupportedType) {
            try FileTextExtractor.extract(url: url)
        }
    }

    @Test func onePagePDFExtractsText() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-extract-\(UUID().uuidString).pdf")
        defer { try? FileManager.default.removeItem(at: url) }

        let payload = "PDF attachment body"
        try writeOnePagePDF(text: payload, to: url)

        let extracted = try FileTextExtractor.extract(url: url)
        #expect(extracted.contains(payload))
    }

    // MARK: - AttachmentImage

    @Test func largeImageIsDownscaledToJPEG() throws {
        let size = NSSize(width: 4000, height: 2000)
        let image = NSImage(size: size)
        image.lockFocus()
        NSColor.systemBlue.setFill()
        NSRect(origin: .zero, size: size).fill()
        image.unlockFocus()

        let data = try #require(AttachmentImage.jpegData(from: image, maxDimension: 1024, quality: 0.8))
        #expect(!data.isEmpty)

        let decoded = try #require(NSImage(data: data))
        let rep = try #require(decoded.representations.compactMap { $0 as? NSBitmapImageRep }.first
            ?? decoded.cgImage(forProposedRect: nil, context: nil, hints: nil).map { cg in
                NSBitmapImageRep(cgImage: cg)
            })
        let longest = max(rep.pixelsWide, rep.pixelsHigh)
        #expect(longest <= 1024)
        #expect(longest > 0)
    }

    // MARK: - PromptBuilder.userText

    @Test func plainContentUnchangedWithoutAttachment() {
        #expect(PromptBuilder.userText(content: "Hello", attachmentName: nil, attachmentText: nil)
            == "Hello")
        #expect(PromptBuilder.userText(content: "Hello", attachmentName: "a.txt", attachmentText: nil)
            == "Hello")
        #expect(PromptBuilder.userText(content: "Hello", attachmentName: nil, attachmentText: "body")
            == "Hello")
    }

    @Test func attachmentComposesFileBlock() {
        let result = PromptBuilder.userText(
            content: "Please summarize",
            attachmentName: "notes.md",
            attachmentText: "line one\nline two")
        #expect(result == "Please summarize\n\n[File: notes.md]\nline one\nline two")
    }

    @Test func multipleAttachmentsComposeOneBlockEach() {
        let result = PromptBuilder.userText(
            content: "Compare these",
            attachments: [
                (name: "a.txt", text: "alpha"),
                (name: "b.txt", text: "beta"),
            ])
        #expect(result == "Compare these\n\n[File: a.txt]\nalpha\n\n[File: b.txt]\nbeta")
    }

    @Test func noAttachmentsLeavesContentUnchanged() {
        #expect(PromptBuilder.userText(content: "Hello", attachments: []) == "Hello")
    }

    // MARK: - PromptBuilder.attachmentsExceedBudget

    @Test func smallAttachmentsFitBudget() {
        let exceeds = PromptBuilder.attachmentsExceedBudget(
            draft: "Summarize this",
            attachments: [(name: "short.txt", text: "just a few words")],
            ctxTokens: 32_768)
        #expect(!exceeds)
    }

    @Test func oversizedAttachmentExceedsBudget() {
        // ~4 chars/token estimate; budget for a 4096-token model is ~2867 tokens
        // (~11,468 chars). A single file well past that should trip the check
        // even though PromptBuilder.trimmed() would never drop this pending turn.
        let hugeText = String(repeating: "a", count: 50_000)
        let exceeds = PromptBuilder.attachmentsExceedBudget(
            draft: "",
            attachments: [(name: "huge.txt", text: hugeText)],
            ctxTokens: 4_096)
        #expect(exceeds)
    }

    @Test func multipleSmallAttachmentsCanCombineToExceedBudget() {
        let chunk = String(repeating: "a", count: 4_000)  // ~1000 tokens each
        let attachments = (0..<5).map { (name: "file\($0).txt", text: chunk) }  // ~5000 tokens
        let exceeds = PromptBuilder.attachmentsExceedBudget(
            draft: "",
            attachments: attachments,
            ctxTokens: 4_096)  // budget ~2867 tokens
        #expect(exceeds)
    }

    // MARK: - Helpers

    private func writeOnePagePDF(text: String, to url: URL) throws {
        let pageBounds = CGRect(x: 0, y: 0, width: 612, height: 792)
        let document = PDFDocument()
        // Use a simple Core Graphics PDF context so the string is extractable.
        var mediaBox = pageBounds
        let consumer = CGDataConsumer(url: url as CFURL)!
        guard let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
            throw NSError(domain: "AttachmentTests", code: 1)
        }
        context.beginPDFPage(nil)
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 14),
            .foregroundColor: NSColor.black,
        ]
        let ns = NSAttributedString(string: text, attributes: attrs)
        let framesetter = CTFramesetterCreateWithAttributedString(ns)
        let path = CGPath(rect: pageBounds.insetBy(dx: 72, dy: 72), transform: nil)
        let frame = CTFramesetterCreateFrame(
            framesetter, CFRange(location: 0, length: ns.length), path, nil)
        context.textMatrix = .identity
        // PDFKit / Core Text uses a flipped coordinate system for drawing.
        context.translateBy(x: 0, y: pageBounds.height)
        context.scaleBy(x: 1, y: -1)
        CTFrameDraw(frame, context)
        context.endPDFPage()
        context.closePDF()
    }
}
