import AppKit
import UniformTypeIdentifiers

/// Owns NSOpenPanel configuration and attachment loading for the composer.
@MainActor
enum AttachmentPicker {
    enum ImageResult {
        case cancelled
        case loadFailed
        case picked(data: Data, preview: NSImage?)
    }

    enum FileResult {
        case cancelled
        case failed(String)
        case picked(name: String, text: String)
    }

    /// Content types accepted by the file attachment panel.
    static var fileContentTypes: [UTType] {
        var types: [UTType] = [.pdf, .plainText, .utf8PlainText, .sourceCode, .json, .xml, .yaml]
        for ext in FileTextExtractor.textExtensions {
            if let type = UTType(filenameExtension: ext) {
                types.append(type)
            }
        }
        return types
    }

    /// Presents an image open panel. Returns `.cancelled` if the user dismisses.
    static func pickImage() -> ImageResult {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [
            .png, .jpeg, .heic, .webP, .gif,
        ]
        panel.message = "Choose a photo or image"
        guard panel.runModal() == .OK, let url = panel.url else { return .cancelled }

        guard let data = AttachmentImage.jpegData(from: url) else {
            return .loadFailed
        }
        let preview = NSImage(data: data) ?? NSImage(contentsOf: url)
        return .picked(data: data, preview: preview)
    }

    /// Presents a file open panel and extracts text. Returns `.cancelled` if dismissed.
    static func pickFile() -> FileResult {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = fileContentTypes
        panel.message = "Choose a PDF or text file"
        guard panel.runModal() == .OK, let url = panel.url else { return .cancelled }

        do {
            let text = try FileTextExtractor.extract(url: url)
            return .picked(name: url.lastPathComponent, text: text)
        } catch {
            return .failed(error.localizedDescription)
        }
    }
}
