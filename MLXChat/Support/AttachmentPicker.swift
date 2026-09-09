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
        /// Every selected file failed to read (e.g. a single unsupported file).
        case failed(String)
        /// `files` succeeded; `failures` lists any selected files that didn't
        /// (per-file name + reason), so a partial success still surfaces what
        /// went wrong instead of failing the whole pick silently.
        case picked(files: [(name: String, text: String)], failures: [(name: String, reason: String)])
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

    /// Presents a file open panel allowing multiple selections and extracts text
    /// from each. Returns `.cancelled` if dismissed without picking anything.
    /// Files that fail to extract don't block the ones that succeed — they're
    /// reported back in `failures` instead.
    static func pickFile() -> FileResult {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = fileContentTypes
        panel.message = "Choose PDF or text files"
        guard panel.runModal() == .OK, !panel.urls.isEmpty else { return .cancelled }

        var files: [(name: String, text: String)] = []
        var failures: [(name: String, reason: String)] = []
        for url in panel.urls {
            do {
                let text = try FileTextExtractor.extract(url: url)
                files.append((name: url.lastPathComponent, text: text))
            } catch {
                failures.append((name: url.lastPathComponent, reason: error.localizedDescription))
            }
        }

        if files.isEmpty, let first = failures.first {
            return .failed(first.reason)
        }
        return .picked(files: files, failures: failures)
    }
}
