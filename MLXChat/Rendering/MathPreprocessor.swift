import Foundation

/// Rewrites LaTeX math spans in markdown into `swiftmath://` image syntax that the
/// math image providers render natively. Applied ONLY to markdown blocks (never to
/// fenced code); inline code spans are protected here.
///
/// - Display math (`$$…$$`, `\[…\]`) → own-paragraph `![math](swiftmath://d/<b64url>)`
/// - Inline math (`\(…\)`, and `$…$` that looks like LaTeX) → `![math](swiftmath://i/<b64url>)`
/// - `\boxed{X}` (delimited or bare) → boxed variants `bd`/`bi`: SwiftMath has no
///   \boxed support, so the image provider draws the box around the rendered X.
enum MathPreprocessor {
    static func preprocess(_ markdown: String) -> String {
        // Math dressed up as a code fence renders as an equation instead.
        let markdown = convertMathFences(markdown)
        // Protect inline code spans: transform only the gaps between them.
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
        output = output.replacing(/\$([^\s$][^$\n]{0,118}?[^\s$]|[^\s$])\$/) { match in
            let latex = String(match.1)
            if let boxed = boxedContent(latex) { return boxedInlineImage(boxed) }
            // Simple expressions ($i$, $dp[i]$, $dp[0] = 1$) become math-italic
            // Unicode TEXT — it sits on the text baseline perfectly, unlike an
            // image. Must contain a letter so "$100$" stays money-like.
            if latex.wholeMatch(of: /[A-Za-z0-9'\[\](),=+\-*\/×·. ]{1,60}/) != nil,
               latex.contains(/[A-Za-z]/) {
                return mathItalic(latex)
            }
            // Anything with real LaTeX syntax renders as an image.
            guard latex.contains(/[\\^_{}<>]/) else { return String(match.0) }
            return inlineImage(stripBoxed(latex))
        }
        // Bare \boxed{X} with no math delimiters — models often end the final
        // answer with it, and cmark would otherwise print the command literally.
        output = replaceBareBoxed(output)
        return output
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

    /// \boxed-aware display rendering: a span that IS `\boxed{X}` gets the box
    /// drawn around X's image; \boxed embedded in a larger expression degrades
    /// to its content so the surrounding math still renders.
    private static func displayMath(_ latex: String) -> String {
        if let boxed = boxedContent(latex) { return boxedDisplayImage(boxed) }
        return displayImage(stripBoxed(latex))
    }

    /// \boxed-aware inline rendering; same policy as displayMath.
    private static func inlineMath(_ latex: String) -> String {
        if let boxed = boxedContent(latex) { return boxedInlineImage(boxed) }
        return inlineImage(stripBoxed(latex))
    }

    /// Replaces bare `\boxed{X}` occurrences with boxed inline images, skipping
    /// fenced code blocks (inside code it's content, not math). Runs after the
    /// delimiter rules, so only delimiter-less occurrences remain.
    private static func replaceBareBoxed(_ text: String) -> String {
        var result = ""
        var cursor = text.startIndex
        let fence = /(?m)^(`{3,})[^\n]*\n[\s\S]*?^\1[ \t]*$/
        for match in text.matches(of: fence) {
            result += convertBareBoxed(String(text[cursor..<match.range.lowerBound]))
            result += text[match.range]
            cursor = match.range.upperBound
        }
        result += convertBareBoxed(String(text[cursor...]))
        return result
    }

    private static func convertBareBoxed(_ text: String) -> String {
        var rest = text
        var result = ""
        while let (inner, range) = firstBoxed(in: rest) {
            result += rest[..<range.lowerBound] + boxedInlineImage(inner)
            rest = String(rest[range.upperBound...])
        }
        return result + rest
    }

    /// If `latex` is exactly `\boxed{X}` (surrounding whitespace tolerated),
    /// returns X — else nil.
    private static func boxedContent(_ latex: String) -> String? {
        let trimmed = latex.trimmingCharacters(in: .whitespaces)
        guard let (inner, range) = firstBoxed(in: trimmed),
              range.lowerBound == trimmed.startIndex, range.upperBound == trimmed.endIndex
        else { return nil }
        return inner
    }

    /// Replaces every `\boxed{X}` with X (nested braces survive inside X).
    private static func stripBoxed(_ latex: String) -> String {
        var rest = latex
        var result = ""
        while let (inner, range) = firstBoxed(in: rest) {
            result += rest[..<range.lowerBound] + inner
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

    /// Maps ASCII letters to their MATHEMATICAL ITALIC code points (the same
    /// glyph style SwiftMath uses), leaving digits and symbols untouched.
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

    private static func boxedDisplayImage(_ latex: String) -> String {
        "\n\n![math](swiftmath://bd/\(base64URL(latex)))\n\n"
    }

    private static func boxedInlineImage(_ latex: String) -> String {
        "![math](swiftmath://bi/\(base64URL(latex)))"
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
