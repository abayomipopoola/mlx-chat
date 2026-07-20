import Foundation
import PDFKit

/// Extracts plain text from user-selected files for prompt injection.
enum FileTextExtractor {
    static let maxCharacters = 100_000

    /// File extensions treated as UTF-8 text (lossy fallback).
    static let textExtensions: Set<String> = [
        "txt", "md", "markdown", "swift", "py", "js", "ts", "json", "jsonl",
        "csv", "xml", "yaml", "yml", "log", "html", "css", "sh", "rb", "go",
        "rs", "java", "c", "h", "cpp", "toml", "ini", "cfg",
    ]

    enum ExtractError: LocalizedError {
        case unsupportedType
        case unreadable
        case emptyPDF

        var errorDescription: String? {
            switch self {
            case .unsupportedType:
                return "That file type isn't supported."
            case .unreadable:
                return "Couldn't read that file."
            case .emptyPDF:
                return "That PDF has no extractable text."
            }
        }
    }

    /// Reads text from `url`. Caps at `maxCharacters` and appends a truncation marker.
    /// Callers that received a security-scoped URL from `NSOpenPanel` get scoped access
    /// for the duration of the call.
    static func extract(url: URL) throws -> String {
        let accessed = url.startAccessingSecurityScopedResource()
        defer {
            if accessed { url.stopAccessingSecurityScopedResource() }
        }

        let ext = url.pathExtension.lowercased()
        let raw: String
        if ext == "pdf" {
            raw = try extractPDF(url: url)
        } else if textExtensions.contains(ext) {
            raw = try extractTextFile(url: url)
        } else {
            throw ExtractError.unsupportedType
        }

        return capped(raw)
    }

    // MARK: - Private

    private static func extractPDF(url: URL) throws -> String {
        guard let document = PDFDocument(url: url) else {
            throw ExtractError.unreadable
        }
        guard let string = document.string, !string.isEmpty else {
            throw ExtractError.emptyPDF
        }
        return string
    }

    private static func extractTextFile(url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        if let utf8 = String(data: data, encoding: .utf8) {
            return utf8
        }
        // Lossy fallback for non-UTF-8 text-like files.
        return String(decoding: data, as: UTF8.self)
    }

    private static func capped(_ text: String) -> String {
        guard text.count > maxCharacters else { return text }
        return String(text.prefix(maxCharacters)) + "\n…[truncated]"
    }
}
