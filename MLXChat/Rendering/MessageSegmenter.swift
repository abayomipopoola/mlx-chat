import Foundation

/// A renderable slice of an assistant message.
enum RenderBlock: Identifiable, Hashable {
    case markdown(id: Int, text: String)
    case artifact(Artifact)

    var id: Int {
        switch self {
        case .markdown(let id, _): return id
        case .artifact(let artifact): return artifact.id
        }
    }
}

struct Artifact: Identifiable, Hashable {
    enum Kind: Hashable { case code, html, svg, markdown }

    let id: Int
    let kind: Kind
    let language: String?
    let title: String
    let content: String
    /// false while the closing fence hasn't streamed in yet.
    let isComplete: Bool

    var lineCount: Int {
        content.isEmpty ? 0 : content.components(separatedBy: "\n").count
    }

    var suggestedFileExtension: String {
        switch (kind, language?.lowercased()) {
        case (.html, _): return "html"
        case (.svg, _): return "svg"
        case (.markdown, _): return "md"
        case (_, "python"), (_, "py"): return "py"
        case (_, "swift"): return "swift"
        case (_, "javascript"), (_, "js"): return "js"
        case (_, "typescript"), (_, "ts"): return "ts"
        case (_, "json"): return "json"
        case (_, "bash"), (_, "sh"), (_, "shell"), (_, "zsh"): return "sh"
        case (_, "rust"), (_, "rs"): return "rs"
        case (_, "go"): return "go"
        case (_, "c"): return "c"
        case (_, "cpp"), (_, "c++"): return "cpp"
        case (_, "java"): return "java"
        case (_, "kotlin"), (_, "kt"): return "kt"
        case (_, "ruby"), (_, "rb"): return "rb"
        default: return "txt"
        }
    }
}

/// Splits raw assistant markdown into markdown runs and artifact blocks.
///
/// Rules: a fenced block becomes an artifact when its language is html/svg (any size),
/// or it has >= 16 lines, or >= 1200 characters. Smaller fences stay embedded in the
/// surrounding markdown. An unterminated trailing fence yields `isComplete: false`.
enum MessageSegmenter {
    private static let artifactMinLines = 16
    private static let artifactMinChars = 1200

    static func segment(_ text: String) -> [RenderBlock] {
        var blocks: [RenderBlock] = []
        var markdownBuffer: [String] = []
        var nextID = 0

        var fenceMarker: Character?
        var fenceLength = 0
        var fenceIndent = ""
        var fenceLanguage: String?
        var fenceOpenLine = ""
        var fenceLines: [String] = []

        func flushMarkdown() {
            let text = markdownBuffer.joined(separator: "\n")
            if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                blocks.append(.markdown(id: nextID, text: text))
                nextID += 1
            }
            markdownBuffer = []
        }

        func closeFence(complete: Bool) {
            let content = fenceLines.joined(separator: "\n")
            if isArtifact(language: fenceLanguage, content: content) {
                flushMarkdown()
                blocks.append(.artifact(makeArtifact(
                    id: nextID, language: fenceLanguage, content: content, complete: complete)))
                nextID += 1
            } else {
                markdownBuffer.append(fenceOpenLine)
                markdownBuffer.append(contentsOf: fenceLines)
                if complete { markdownBuffer.append(fenceIndent + String(repeating: String(fenceMarker!), count: fenceLength)) }
            }
            fenceMarker = nil
            fenceLength = 0
            fenceIndent = ""
            fenceLanguage = nil
            fenceOpenLine = ""
            fenceLines = []
        }

        let lines = text.components(separatedBy: "\n")
        for line in lines {
            if let marker = fenceMarker {
                if isClosingFence(line, marker: marker, minLength: fenceLength) {
                    closeFence(complete: true)
                } else {
                    fenceLines.append(line)
                }
            } else if let opener = parseOpeningFence(line) {
                fenceMarker = opener.marker
                fenceLength = opener.length
                fenceIndent = opener.indent
                fenceLanguage = opener.language
                fenceOpenLine = line
            } else {
                markdownBuffer.append(line)
            }
        }

        if fenceMarker != nil {
            closeFence(complete: false)
        }
        flushMarkdown()
        return blocks
    }

    // MARK: - Fence parsing

    private static func parseOpeningFence(
        _ line: String
    ) -> (marker: Character, length: Int, indent: String, language: String?)? {
        var index = line.startIndex
        var indent = ""
        while index < line.endIndex, line[index] == " ", indent.count < 3 {
            indent.append(" ")
            index = line.index(after: index)
        }
        guard index < line.endIndex else { return nil }
        let marker = line[index]
        guard marker == "`" || marker == "~" else { return nil }
        var length = 0
        while index < line.endIndex, line[index] == marker {
            length += 1
            index = line.index(after: index)
        }
        guard length >= 3 else { return nil }
        let info = line[index...].trimmingCharacters(in: .whitespaces)
        // Backtick fences may not contain backticks in the info string (CommonMark).
        if marker == "`", info.contains("`") { return nil }
        let language = info.split(separator: " ").first.map(String.init)
        return (marker, length, indent, (language?.isEmpty ?? true) ? nil : language)
    }

    private static func isClosingFence(_ line: String, marker: Character, minLength: Int) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, trimmed.allSatisfy({ $0 == marker }) else { return false }
        return trimmed.count >= minLength
    }

    // MARK: - Artifact classification

    private static func isArtifact(language: String?, content: String) -> Bool {
        let lang = language?.lowercased()
        if lang == "html" || lang == "svg" { return true }
        if content.count >= artifactMinChars { return true }
        return content.components(separatedBy: "\n").count >= artifactMinLines
    }

    private static func makeArtifact(id: Int, language: String?, content: String, complete: Bool) -> Artifact {
        let lang = language?.lowercased()
        let kind: Artifact.Kind
        switch lang {
        case "html": kind = .html
        case "svg": kind = .svg
        case "md", "markdown": kind = .markdown
        default: kind = .code
        }
        return Artifact(
            id: id, kind: kind, language: language,
            title: title(for: kind, language: language, content: content),
            content: content, isComplete: complete)
    }

    private static let languageDisplayNames: [String: String] = [
        "python": "Python", "py": "Python",
        "javascript": "JavaScript", "js": "JavaScript",
        "typescript": "TypeScript", "ts": "TypeScript",
        "swift": "Swift", "rust": "Rust", "go": "Go",
        "bash": "Shell", "sh": "Shell", "shell": "Shell", "zsh": "Shell",
        "json": "JSON", "c": "C", "cpp": "C++", "java": "Java",
        "kotlin": "Kotlin", "ruby": "Ruby", "sql": "SQL",
    ]

    private static func title(for kind: Artifact.Kind, language: String?, content: String) -> String {
        // A leading comment line names the artifact.
        if let firstLine = content
            .components(separatedBy: "\n")
            .first(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty }) {
            let trimmed = firstLine.trimmingCharacters(in: .whitespaces)
            for prefix in ["//", "#!", "#", "--", "/*", "<!--"] where trimmed.hasPrefix(prefix) {
                var name = String(trimmed.dropFirst(prefix.count))
                for suffix in ["*/", "-->"] where name.hasSuffix(suffix) {
                    name = String(name.dropLast(suffix.count))
                }
                name = name.trimmingCharacters(in: .whitespaces)
                if !name.isEmpty { return String(name.prefix(40)) }
            }
        }
        switch kind {
        case .html: return "HTML"
        case .svg: return "SVG"
        case .markdown: return "Markdown"
        case .code:
            let display = language.flatMap { languageDisplayNames[$0.lowercased()] }
                ?? language?.capitalized
            return display.map { "\($0) code" } ?? "Code"
        }
    }
}
