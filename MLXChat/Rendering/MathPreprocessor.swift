import Foundation

/// Rewrites LaTeX math spans in markdown into `swiftmath://` image syntax that the
/// math image providers render natively. Applied ONLY to markdown blocks (never to
/// fenced code); inline code spans are protected here.
///
/// - Display math (`$$…$$`, `\[…\]`) → own-paragraph `![math](swiftmath://d/<b64url>)`
/// - Inline math (`\(…\)`, and `$…$` that looks like LaTeX) → `![math](swiftmath://i/<b64url>)`
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
            displayImage(String(match.1))
        }
        output = output.replacing(/(?s)\\\[(.{1,2000}?)\\\]/) { match in
            displayImage(String(match.1))
        }
        output = output.replacing(/(?s)\\\((.{1,300}?)\\\)/) { match in
            inlineImage(String(match.1))
        }
        output = output.replacing(/\$([^\s$][^$\n]{0,118}?[^\s$]|[^\s$])\$/) { match in
            let latex = String(match.1)
            // Simple expressions ($i$, $dp[i]$, $dp[0] = 1$) become math-italic
            // Unicode TEXT — it sits on the text baseline perfectly, unlike an
            // image. Must contain a letter so "$100$" stays money-like.
            if latex.wholeMatch(of: /[A-Za-z0-9'\[\](),=+\-*\/×·. ]{1,60}/) != nil,
               latex.contains(/[A-Za-z]/) {
                return mathItalic(latex)
            }
            // Anything with real LaTeX syntax renders as an image.
            guard latex.contains(/[\\^_{}<>]/) else { return String(match.0) }
            return inlineImage(latex)
        }
        return output
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
