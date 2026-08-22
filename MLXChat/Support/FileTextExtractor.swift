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
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        let chunkSize = 16 * 1024
        let characterLimit = maxCharacters + 1
        var leftover = Data()
        var result = ""

        while result.count < characterLimit {
            let chunk = (try? handle.read(upToCount: chunkSize)) ?? Data()
            if chunk.isEmpty && leftover.isEmpty { break }

            let buffer = leftover + chunk
            leftover = Data()
            let atEnd = chunk.isEmpty
            guard let (text, rest) = decodeUTF8(buffer, allowIncompleteTail: !atEnd) else {
                return try lossyBoundedPrefix(at: url)
            }
            leftover = rest
            if !text.isEmpty {
                result.append(text)
                if result.count >= characterLimit {
                    return String(result.prefix(characterLimit))
                }
            }
            if atEnd { break }
        }
        return result.count > characterLimit ? String(result.prefix(characterLimit)) : result
    }

    /// Decodes a UTF-8 buffer, holding back a trailing incomplete sequence when
    /// more bytes may follow. Returns nil when the complete prefix is invalid.
    private static func decodeUTF8(_ data: Data, allowIncompleteTail: Bool) -> (String, Data)? {
        if let text = String(data: data, encoding: .utf8) {
            return (text, Data())
        }
        guard allowIncompleteTail else { return nil }
        let tail = incompleteUTF8TailCount(in: data)
        guard tail > 0 else { return nil }
        if tail == data.count { return ("", data) }
        let head = data.dropLast(tail)
        guard let text = String(data: Data(head), encoding: .utf8) else { return nil }
        return (text, Data(data.suffix(tail)))
    }

    private static func incompleteUTF8TailCount(in data: Data) -> Int {
        guard !data.isEmpty else { return 0 }
        let start = max(0, data.count - 4)
        var i = data.count - 1
        while i >= start && data[i] & 0xC0 == 0x80 {
            i -= 1
        }
        if i < start { return 0 }
        let lead = data[i]
        let needed: Int
        switch lead {
        case 0xC0..<0xE0: needed = 2
        case 0xE0..<0xF0: needed = 3
        case 0xF0..<0xF8: needed = 4
        default: return 0
        }
        let have = data.count - i
        return have < needed ? have : 0
    }

    /// Lossy UTF-8 decode of a prefix large enough for `maxCharacters`, never the whole file.
    private static func lossyBoundedPrefix(at url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let maxBytes = (maxCharacters + 1) * 4
        let data = (try? handle.read(upToCount: maxBytes)) ?? Data()
        return String(decoding: data, as: UTF8.self)
    }

    private static func capped(_ text: String) -> String {
        guard text.count > maxCharacters else { return text }
        return String(text.prefix(maxCharacters)) + "\n…[truncated]"
    }
}
