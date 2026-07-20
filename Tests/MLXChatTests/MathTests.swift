import Testing
@testable import MLXChat

/// Contract: MathPreprocessor rewrites LaTeX spans to swiftmath:// image links
/// (inline `i`, display `d` on its own paragraph, boxed variants `bi`/`bd`),
/// leaves money amounts and inline code untouched, and base64url survives a
/// round trip.
@Suite struct MathTests {
    /// Extracts the base64url payload of the first `swiftmath://<host>/…)` link.
    private func payload(host: String, in output: String) -> String? {
        guard let start = output.range(of: "swiftmath://\(host)/") else { return nil }
        let rest = output[start.upperBound...]
        guard let close = rest.firstIndex(of: ")") else { return nil }
        return String(rest[..<close])
    }

    @Test func dollarLatexBecomesInlineImage() {
        let output = MathPreprocessor.preprocess("Euler: $e^{i\\pi}+1=0$.")
        #expect(output.contains("swiftmath://i/"))
        #expect(payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "e^{i\\pi}+1=0")
    }

    @Test func moneyAmountsStayUntouched() {
        let input = "$5 and $10 please"
        #expect(MathPreprocessor.preprocess(input) == input)
        // "$5-$10" spans two dollars with non-space edges — only the LaTeX-ish
        // content guard keeps it from becoming a math image.
        let range = "from $5-$10 total"
        #expect(MathPreprocessor.preprocess(range) == range)
    }

    @Test func displayMathGetsOwnParagraph() {
        let output = MathPreprocessor.preprocess("Before\n$$\\int_0^1 x^2\\,dx$$\nAfter")
        #expect(output.contains("\n\n![math](swiftmath://d/"))
        #expect(output.contains(")\n\n"))
        #expect(payload(host: "d", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "\\int_0^1 x^2\\,dx")
    }

    @Test func mathInsideInlineCodeIsUntouched() {
        let input = "use `$x_1$` here"
        #expect(MathPreprocessor.preprocess(input) == input)
    }

    @Test func escapedParensBecomeInlineImage() {
        let output = MathPreprocessor.preprocess("Solve \\(x_1\\) now")
        #expect(output.contains("swiftmath://i/"))
        #expect(payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "x_1")
    }

    @Test func bareBoxedBecomesBoxedInlineImage() {
        // The reported bug: models end answers with `\boxed{…}` and no math
        // delimiters — it must render boxed, not leak raw LaTeX.
        let output = MathPreprocessor.preprocess("The answer is \\boxed{73682}.")
        #expect(output.contains("The answer is "))
        #expect(output.contains("swiftmath://bi/"))
        #expect(payload(host: "bi", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "73682")
    }

    @Test func dollarWrappedBoxedBecomesBoxedInlineImage() {
        let output = MathPreprocessor.preprocess("Answer: $\\boxed{73682}$")
        #expect(output.contains("swiftmath://bi/"))
        #expect(payload(host: "bi", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "73682")
    }

    @Test func displayBoxedBecomesBoxedDisplayImage() {
        let output = MathPreprocessor.preprocess("$$\\boxed{x}$$")
        #expect(output.contains("swiftmath://bd/"))
        #expect(payload(host: "bd", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "x")
    }

    @Test func bracketMathEmbeddedInTextLineRendersInline() {
        // The reported bug: "…in one line: \[ \boxed{…} \]" broke into its own
        // centered block; an embedded display span must stay on the text line.
        let output = MathPreprocessor.preprocess(
            "So the whole update rule in one line: \\[\\boxed{\\theta_{t+1} = \\theta_t - \\eta\\,\\nabla L(\\theta_t)}\\]")
        #expect(output.contains("So the whole update rule in one line: "))
        #expect(output.contains("swiftmath://bi/"))
        #expect(!output.contains("swiftmath://bd/"))
    }

    @Test func dollarDisplayEmbeddedInTextLineRendersInline() {
        let output = MathPreprocessor.preprocess("Inline here: $$x_1$$ tail")
        #expect(output.contains("swiftmath://i/"))
        #expect(!output.contains("swiftmath://d/"))
    }

    @Test func bracketMathAloneOnItsLineStaysDisplayBlock() {
        let output = MathPreprocessor.preprocess("Result:\n\\[\n\\int_0^1 x\\,dx\n\\]\n")
        #expect(output.contains("swiftmath://d/"))
        #expect(payload(host: "d", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "\n\\int_0^1 x\\,dx\n")
    }

    @Test func boxedEmbeddedInDisplayMathIsStripped() {
        // SwiftMath has no \boxed: inside a larger expression keep the content
        // so the surrounding math still renders.
        let output = MathPreprocessor.preprocess("$$x = \\boxed{5}$$")
        #expect(!output.contains("swiftmath://bd/"))
        #expect(payload(host: "d", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "x = 5")
    }

    @Test func nestedBracesInsideBoxedSurvive() {
        let output = MathPreprocessor.preprocess("\\boxed{\\frac{1}{2}}")
        #expect(payload(host: "bi", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "\\frac{1}{2}")
    }

    @Test func boxedInsideInlineCodeIsUntouched() {
        let input = "use `\\boxed{x}` here"
        #expect(MathPreprocessor.preprocess(input) == input)
    }

    @Test(arguments: [
        ">>>???",            // plain base64 would contain '+' and '/'
        "pi π int ∫ 🚀",     // multi-byte UTF-8
        "a",                 // two padding chars in plain base64
        "ab",                // one padding char
        "abc",               // no padding
        "sign=+/=",
    ])
    func base64URLRoundTrips(_ text: String) {
        let encoded = MathPreprocessor.base64URL(text)
        // URL-safe alphabet: the payload is embedded in a markdown link URL.
        #expect(!encoded.contains("+"))
        #expect(!encoded.contains("/"))
        #expect(!encoded.contains("="))
        #expect(MathPreprocessor.decodeBase64URL(encoded) == text)
    }
}
