import Foundation

/// Rewrites LaTeX math spans in markdown into `swiftmath://` image syntax that the
/// math image providers render natively. Applied ONLY to markdown blocks (never to
/// fenced code); inline code spans are protected here.
///
/// - Display math (`$$…$$`, `\[…\]`) → own-paragraph `![math](swiftmath://d/<b64url>)`
/// - Inline math (`\(…\)`, and `$…$` that looks like LaTeX) → `![math](swiftmath://i/<b64url>)`
/// - Bare `\boxed{X}` in prose → inline image. SwaTex renders `\boxed` itself.
enum MathPreprocessor {
    static func preprocess(_ markdown: String) -> String {
        // Math dressed up as a code fence renders as an equation instead.
        let markdown = repairAccidentalIndentedMath(
            convertMathFences(markdown))
        // Protect fenced and indented code blocks before inline-code handling.
        // Otherwise the generated swiftmath:// image syntax becomes visible
        // source inside a Markdown code block.
        return transformOutsideCodeBlocks(markdown)
    }

    private static func transformOutsideCodeBlocks(_ markdown: String) -> String {
        guard hasPotentialCodeBlock(markdown) else {
            return transformOutsideInlineCode(markdown)
        }
        var result = ""
        var cursor = markdown.startIndex
        for range in protectedCodeBlockRanges(in: markdown) {
            result += transformOutsideInlineCode(String(markdown[cursor..<range.lowerBound]))
            result += markdown[range]
            cursor = range.upperBound
        }
        result += transformOutsideInlineCode(String(markdown[cursor...]))
        return result
    }

    private static func hasPotentialCodeBlock(_ text: String) -> Bool {
        text.contains("```")
            || text.contains("~~~")
            || hasPotentialIndentedLine(text)
    }

    private static func hasPotentialIndentedLine(_ text: String) -> Bool {
        text.hasPrefix("    ")
            || text.hasPrefix("\t")
            || text.contains("\n    ")
            || text.contains("\n\t")
    }

    private static func transformOutsideInlineCode(_ markdown: String) -> String {
        // Transform only the gaps between inline code spans.
        let codeSpan = /(`{1,3})[^`]*?\1/
        var result = ""
        var cursor = markdown.startIndex
        for match in markdown.matches(of: codeSpan) {
            result += transform(String(markdown[cursor..<match.range.lowerBound]))
            result += markdown[match.range]
            cursor = match.range.upperBound
        }
        result += transform(String(markdown[cursor...]))
        return result
    }

    // MARK: - Code-block protection

    private struct LineSlice {
        let contentRange: Range<String.Index>
        let fullRange: Range<String.Index>
    }

    private struct OpenFence {
        let marker: Character
        let minimumLength: Int
        let start: String.Index
    }

    /// Finds CommonMark fenced and indented code regions. Math fences have
    /// already been converted, so every range returned here must stay literal.
    private static func protectedCodeBlockRanges(in text: String) -> [Range<String.Index>] {
        var ranges: [Range<String.Index>] = []
        var fence: OpenFence?
        var indentedStart: String.Index?
        var indentedEnd: String.Index?

        for line in lineSlices(in: text) {
            let content = text[line.contentRange]

            if let currentFence = fence {
                if isClosingFence(
                    content,
                    marker: currentFence.marker,
                    minimumLength: currentFence.minimumLength)
                {
                    ranges.append(currentFence.start..<line.fullRange.upperBound)
                    fence = nil
                }
                continue
            }

            if let start = indentedStart {
                if content.trimmingCharacters(in: .whitespaces).isEmpty
                    || isIndentedCodeLine(content)
                {
                    indentedEnd = line.fullRange.upperBound
                    continue
                }
                ranges.append(start..<(indentedEnd ?? line.fullRange.lowerBound))
                indentedStart = nil
                indentedEnd = nil
            }

            if let opening = openingFence(in: content) {
                fence = OpenFence(
                    marker: opening.marker,
                    minimumLength: opening.length,
                    start: line.fullRange.lowerBound)
            } else if !content.trimmingCharacters(in: .whitespaces).isEmpty,
                      isIndentedCodeLine(content) {
                indentedStart = line.fullRange.lowerBound
                indentedEnd = line.fullRange.upperBound
            }
        }

        if let fence {
            ranges.append(fence.start..<text.endIndex)
        } else if let indentedStart {
            ranges.append(indentedStart..<(indentedEnd ?? text.endIndex))
        }
        return ranges
    }

    private static func lineSlices(in text: String) -> [LineSlice] {
        var lines: [LineSlice] = []
        var start = text.startIndex
        while start < text.endIndex {
            let newline = text[start...].firstIndex(of: "\n")
            let contentEnd = newline ?? text.endIndex
            let fullEnd = newline.map { text.index(after: $0) } ?? text.endIndex
            lines.append(LineSlice(
                contentRange: start..<contentEnd,
                fullRange: start..<fullEnd))
            start = fullEnd
        }
        return lines
    }

    private static func openingFence(
        in line: Substring
    ) -> (marker: Character, length: Int)? {
        var index = line.startIndex
        var spaces = 0
        while index < line.endIndex, line[index] == " ", spaces < 4 {
            spaces += 1
            index = line.index(after: index)
        }
        guard spaces <= 3, index < line.endIndex else { return nil }
        let marker = line[index]
        guard marker == "`" || marker == "~" else { return nil }
        var length = 0
        while index < line.endIndex, line[index] == marker {
            length += 1
            index = line.index(after: index)
        }
        guard length >= 3 else { return nil }
        if marker == "`", line[index...].contains("`") { return nil }
        return (marker, length)
    }

    private static func isClosingFence(
        _ line: Substring,
        marker: Character,
        minimumLength: Int
    ) -> Bool {
        var index = line.startIndex
        var spaces = 0
        while index < line.endIndex, line[index] == " ", spaces < 4 {
            spaces += 1
            index = line.index(after: index)
        }
        guard spaces <= 3 else { return false }
        var length = 0
        while index < line.endIndex, line[index] == marker {
            length += 1
            index = line.index(after: index)
        }
        guard length >= minimumLength else { return false }
        return line[index...].allSatisfy { $0 == " " || $0 == "\t" }
    }

    private static func isIndentedCodeLine(_ line: Substring) -> Bool {
        let remainder: Substring
        if line.first == "\t" {
            remainder = line.dropFirst()
        } else if line.hasPrefix("    ") {
            remainder = line.dropFirst(4)
        } else {
            return false
        }
        // Nested markdown lists use one indent unit plus a marker; they are
        // not CommonMark indented code, so `$…$` inside them must transform.
        if remainder.prefixMatch(of: /^[ \t]*(?:[-+*]|[0-9]+[.)])[ \t]+/) != nil {
            return false
        }
        return true
    }

    // MARK: - Malformed-model repair

    /// Models sometimes prefix prose or a bare LaTeX equation with four spaces,
    /// which CommonMark treats as code. Repair only one indentation level and
    /// only when the line is clearly math, leaving real code and list bodies
    /// untouched.
    private static func repairAccidentalIndentedMath(_ markdown: String) -> String {
        guard hasPotentialIndentedLine(markdown) else { return markdown }
        var lines = markdown.components(separatedBy: "\n")
        var previousNonblank: String?
        var fence: (marker: Character, minimumLength: Int)?

        for index in lines.indices {
            let original = lines[index]
            let line = Substring(original)
            if let currentFence = fence {
                if isClosingFence(
                    line,
                    marker: currentFence.marker,
                    minimumLength: currentFence.minimumLength)
                {
                    fence = nil
                }
                continue
            }
            if let opening = openingFence(in: line) {
                fence = (opening.marker, opening.length)
                continue
            }

            let originalTrimmed = original.trimmingCharacters(in: .whitespaces)
            defer {
                if !originalTrimmed.isEmpty { previousNonblank = originalTrimmed }
            }

            guard let dedented = removeSingleCodeIndent(from: original),
                  previousNonblank?.wholeMatch(
                    of: /(?:[-+*]|[0-9]+[.)])[ \t]+.*/
                  ) == nil
            else { continue }

            let candidate = dedented.trimmingCharacters(in: .whitespaces)
            guard !candidate.isEmpty, !looksLikeSourceCode(candidate) else { continue }

            if transform(candidate) != candidate {
                lines[index] = dedented
            } else if looksLikeBareLatexEquation(candidate) {
                lines[index] = "$$\(candidate)$$"
            }
        }
        return lines.joined(separator: "\n")
    }

    private static func removeSingleCodeIndent(from line: String) -> String? {
        if line.hasPrefix("\t") {
            let result = String(line.dropFirst())
            return result.first == "\t" || result.hasPrefix("    ") ? nil : result
        }
        guard line.hasPrefix("    ") else { return nil }
        let result = String(line.dropFirst(4))
        return result.first == " " || result.first == "\t" ? nil : result
    }

    private static func looksLikeSourceCode(_ line: String) -> Bool {
        let prefixes = [
            "let ", "var ", "func ", "class ", "struct ", "enum ",
            "import ", "return ", "if ", "for ", "while ", "switch ",
            "case ", "print(", "#", "//", "/*", "<",
        ]
        return prefixes.contains { line.hasPrefix($0) }
            || line.contains("\"")
            || line.contains("`")
            || line.hasSuffix(";")
    }

    private static func looksLikeBareLatexEquation(_ line: String) -> Bool {
        let commands = [
            "\\frac", "\\sum", "\\prod", "\\int", "\\lim", "\\left",
            "\\right", "\\sqrt", "\\infty", "\\to", "\\cdot", "\\times",
            "\\nabla", "\\epsilon", "\\theta", "\\mathbb", "\\mathrm",
            "\\begin", "\\end", "\\tfrac", "\\dfrac", "\\text",
        ]
        guard commands.contains(where: { line.contains($0) }) else { return false }
        return line.contains("=") || line.first == "\\"
    }

    /// Models sometimes emit equations inside fences. Two shapes are rescued:
    /// - ```math / ```latex / ```tex — the whole body is LaTeX
    /// - ```text (or bare ```) whose entire body is a single `$…$` span
    private static func convertMathFences(_ text: String) -> String {
        var output = text
        output = output.replacing(
            /(?m)^```[ \t]*(?:math|latex|tex)[ \t]*\n([\s\S]{1,2000}?)\n```[ \t]*$/
        ) { match in
            var latex = String(match.1).trimmingCharacters(in: .whitespacesAndNewlines)
            latex = latex.trimmingCharacters(in: CharacterSet(charactersIn: "$"))
            return displayImage(latex)
        }
        output = output.replacing(
            /(?m)^```[ \t]*(?:text)?[ \t]*\n[ \t]*\$\$?([^$\n]{1,300}?)\$\$?[ \t]*\n```[ \t]*$/
        ) { match in
            displayImage(String(match.1))
        }
        return output
    }

    private static func transform(_ text: String) -> String {
        var output = text

        // Order matters: $$…$$ before $…$; brackets/parens are independent.
        output = output.replacing(/(?s)\$\$(.{1,2000}?)\$\$/) { match in
            math(String(match.1), display: isStandalone(output, match.range))
        }
        output = output.replacing(/(?s)\\\[(.{1,2000}?)\\\]/) { match in
            math(String(match.1), display: isStandalone(output, match.range))
        }
        output = output.replacing(/(?s)\\\((.{1,300}?)\\\)/) { match in
            inlineMath(String(match.1))
        }
        output = output.replacing(/\$[ \t]*([^\s$][^$\n]{0,118}?[^\s$]|[^\s$])[ \t]*\$/) { match in
            let latex = String(match.1)
            // Simple expressions ($i$, $dp[i]$, $dp[0] = 1$) become math-italic
            // Unicode TEXT — it sits on the text baseline perfectly, unlike an
            // image. Must contain a letter so "$100$" stays money-like.
            // Trailing `[ \t]*` before the closing `$` can pair `$5 and $10`
            // or `$x and $y` as one span; keep those English phrases literal
            // unless the capture already looks like an expression.
            if latex.wholeMatch(of: /[A-Za-z0-9'\[\](),=+\-*\/×·. ]{1,60}/) != nil,
               latex.contains(/[A-Za-z]/),
               !(latex.contains(" ") && latex.wholeMatch(of: /.*[=+\-*\/×·].*/) == nil) {
                return mathItalic(latex)
            }
            // Paired abs bars (`$|Pos(B) - Pos(D)| = 1$`) are math, but a lone
            // prose pipe is not. Do not add `|` to the italic charset — that
            // path is text, and this span needs a math image.
            if looksLikePairedPipeAbs(latex) {
                return inlineImage(latex)
            }
            // `$P ⇔ Q$` has no TeX commands; SwaTex still typesets the Unicode.
            if containsMathUnicode(latex) {
                return inlineImage(latex)
            }
            // Anything with real LaTeX syntax renders as an image.
            guard latex.contains(/[\\^_{}<>]/) else { return String(match.0) }
            return inlineImage(latex)
        }
        output = wrapUnicodeBigOperators(output)
        // `transform` only receives slices outside code blocks, so bare boxed
        // expressions can be converted directly without scanning for fences again.
        output = convertBareBoxed(output)
        return output
    }

    /// Unicode math in a `$…$` span with no TeX commands. Prose operators
    /// stay as text and draw through the font cascade.
    private static func containsMathUnicode(_ text: String) -> Bool {
        text.unicodeScalars.contains { $0.properties.isMath && $0.value > 0x7F }
    }

    /// Unicode Σ/∑/Π/∏/∫ plus a subscript and/or superscript become inline
    /// math images. Lone letters with no script stay literal.
    private static func wrapUnicodeBigOperators(_ text: String) -> String {
        text.replacing(
            /([Σ∑Π∏∫])(_\{[^}\n]{1,80}\}(?:\^\{[^}\n]{1,40}\})?|\^\{[^}\n]{1,40}\}(?:_\{[^}\n]{1,80}\})?)/
        ) { match in
            let command: String
            switch String(match.1) {
            case "Σ", "∑": command = "\\sum"
            case "Π", "∏": command = "\\prod"
            default: command = "\\int"
            }
            return inlineImage(command + String(match.2))
        }
    }

    /// True for a dollar span whose pipes are paired abs/norm bars and the
    /// rest of the span looks like an expression. A single prose pipe
    /// (`$a | b$`) and table-like `$| a | b | c |$` stay literal; currency
    /// spans never reach here because they have no `|`.
    private static func looksLikePairedPipeAbs(_ latex: String) -> Bool {
        let pipeCount = latex.reduce(into: 0) { count, character in
            if character == "|" { count += 1 }
        }
        guard pipeCount >= 2, pipeCount.isMultiple(of: 2) else { return false }
        if latex.contains(/[=+\-*\/^_{}\\<>]/) || latex.contains(/[A-Za-z]\(/) {
            return true
        }
        return pipeCount == 2 && latex.contains(/\|[A-Za-z0-9'\[\](),. ]+\|/)
    }

    /// Display or inline rendering chosen by position. Models often emit
    /// `\[…\]`/`$$…$$` right after a lead-in ("…in one line: \[ X \]"); a span
    /// embedded in a text line must stay inline on that line, not break into a
    /// centered block. Only a span alone on its own line(s) becomes display.
    private static func math(_ latex: String, display: Bool) -> String {
        display ? displayMath(latex) : inlineMath(latex)
    }

    /// True when the span is the only non-whitespace content on its line(s) —
    /// i.e. it forms its own block rather than trailing a colon in a sentence.
    private static func isStandalone(_ text: String, _ range: Range<String.Index>) -> Bool {
        let lineStart = text[..<range.lowerBound].lastIndex(of: "\n")
            .map { text.index(after: $0) } ?? text.startIndex
        let lineEnd = text[range.upperBound...].firstIndex(of: "\n") ?? text.endIndex
        return text[lineStart..<range.lowerBound].allSatisfy(" \t".contains)
            && text[range.upperBound..<lineEnd].allSatisfy(" \t".contains)
    }

    private static func displayMath(_ latex: String) -> String {
        displayImage(latex)
    }

    private static func inlineMath(_ latex: String) -> String {
        inlineImage(latex)
    }

    private static func convertBareBoxed(_ text: String) -> String {
        var rest = text
        var result = ""
        while let (_, range) = firstBoxed(in: rest) {
            result += rest[..<range.lowerBound] + inlineImage(String(rest[range]))
            rest = String(rest[range.upperBound...])
        }
        return result + rest
    }

    /// Locates the first `\boxed{X}` span, walking nested braces so X may itself
    /// contain `{…}`. Returns X and the full span's range, or nil when absent
    /// or unbalanced.
    private static func firstBoxed(in text: String) -> (inner: String, range: Range<String.Index>)? {
        guard let start = text.range(of: #"\boxed{"#) else { return nil }
        var depth = 1
        var index = start.upperBound
        while index < text.endIndex, depth > 0 {
            switch text[index] {
            case "{": depth += 1
            case "}": depth -= 1
            default: break
            }
            if depth == 0 { break }
            index = text.index(after: index)
        }
        guard depth == 0 else { return nil }
        return (String(text[start.upperBound..<index]), start.lowerBound..<text.index(after: index))
    }

    /// Maps ASCII letters to their MATHEMATICAL ITALIC code points, leaving
    /// digits and symbols untouched.
    private static func mathItalic(_ text: String) -> String {
        String(text.flatMap { char -> [Character] in
            guard let scalar = char.unicodeScalars.first, char.unicodeScalars.count == 1 else {
                return [char]
            }
            let value: UInt32
            switch scalar.value {
            case 0x27:  // "'" → PRIME, so f'(x) typesets as f′(x)
                value = 0x2032
            case 0x68:  // "h" has no slot in the italic block (Planck constant).
                value = 0x210E
            case 0x41...0x5A:  // A–Z
                value = 0x1D434 + (scalar.value - 0x41)
            case 0x61...0x7A:  // a–z
                value = 0x1D44E + (scalar.value - 0x61)
            default:
                return [char]
            }
            return [Character(UnicodeScalar(value)!)]
        })
    }

    private static func displayImage(_ latex: String) -> String {
        "\n\n![math](swiftmath://d/\(base64URL(latex)))\n\n"
    }

    private static func inlineImage(_ latex: String) -> String {
        "![math](swiftmath://i/\(base64URL(latex)))"
    }

    static func base64URL(_ text: String) -> String {
        Data(text.utf8).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    static func decodeBase64URL(_ encoded: String) -> String? {
        var base64 = encoded
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        while base64.count % 4 != 0 { base64 += "=" }
        guard let data = Data(base64Encoded: base64) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}
