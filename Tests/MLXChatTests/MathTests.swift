import AppKit
import Testing
@testable import MLXChat

/// Contract: MathPreprocessor rewrites LaTeX spans to swiftmath:// image links
/// (inline `i`, display `d` on its own paragraph), leaves money amounts and
/// inline code untouched, and base64url survives a round trip. SwaTex keeps
/// KaTeX commands (`\tfrac`, `\implies`, `\boxed`) as written.
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

    @Test func dollarPipeAbsBecomesInlineImage() {
        let output = MathPreprocessor.preprocess("$|Pos(B) - Pos(D)| = 1$")
        #expect(output.contains("swiftmath://i/"))
        #expect(payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "|Pos(B) - Pos(D)| = 1")
    }

    @Test func dollarPipeAbsInProseBecomesInlineImage() {
        let output = MathPreprocessor.preprocess(
            "So $|Pos(B) - Pos(D)| = 1$ and we are done.")
        #expect(output.contains("So "))
        #expect(output.contains("swiftmath://i/"))
        #expect(payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "|Pos(B) - Pos(D)| = 1")
    }

    @Test func unpairedProsePipeDollarSpanStaysLiteral() {
        let input = "choose $a | b$ here"
        #expect(MathPreprocessor.preprocess(input) == input)
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

    @Test func mathInsideFencedCodeIsUntouched() {
        let input = """
            ```swift
            let formula = "$x_1$"
            ```
            """
        #expect(MathPreprocessor.preprocess(input) == input)
    }

    @Test func mathInsideIntentionalIndentedCodeIsUntouched() {
        let input = "    print(\"$x_1$\")"
        #expect(MathPreprocessor.preprocess(input) == input)
    }

    @Test func nestedListImpliesBecomesInlineImage() {
        let input = """
            1. Premise
                - Check P: $Q=R \\implies T=T$.
            """
        let output = MathPreprocessor.preprocess(input)
        #expect(output.contains("swiftmath://i/"))
        #expect(payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "Q=R \\implies T=T")
        #expect(!output.contains("$Q=R"))
    }

    @Test func q1ProseLogicUnicodeStaysText() {
        let input = "P ⇔ (Q = R) and Q ⇔ (¬P ∨ S) so S ⇔ ¬R."
        let output = MathPreprocessor.preprocess(input)
        #expect(output.contains("⇔"))
        #expect(output.contains("∨"))
        #expect(!output.contains("swiftmath://"))
    }

    @Test func dollarLogicUnicodeBecomesMathImage() {
        let output = MathPreprocessor.preprocess("$P ⇔ Q$")
        #expect(output.contains("swiftmath://i/"))
        let decoded = payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL) ?? ""
        #expect(decoded.contains("⇔"))
        #expect(output.components(separatedBy: "swiftmath://i/").count == 2)
    }

    @Test func impliesStaysImplies() {
        let output = MathPreprocessor.preprocess("$Q=R \\implies T=T$")
        let decoded = payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL) ?? ""
        #expect(decoded.contains("\\implies"))
    }

    @Test func replacementCharacterIsNotMapped() {
        let input = "P \u{FFFD} (Q = R)"
        #expect(MathPreprocessor.preprocess(input) == input)
    }

    @Test func nestedListNeqBecomesInlineImage() {
        let input = """
            1. Premise
                - $Q \\neq R$
            """
        let output = MathPreprocessor.preprocess(input)
        #expect(output.contains("swiftmath://i/"))
        let decoded = payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL) ?? ""
        #expect(decoded.contains("\\neq"))
    }

    @Test func dollarSpanAllowsSpaceAfterOpeningDollar() {
        let output = MathPreprocessor.preprocess(
            "$ \\neg P \\lor S \\implies F \\lor F = \\text{False}$")
        #expect(output.contains("swiftmath://i/"))
        let decoded = payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL) ?? ""
        #expect(decoded.contains("\\neg"))
        #expect(decoded.contains("\\text{False}"))
        #expect(!decoded.hasPrefix(" "))
    }

    @Test func nestedListSimpleEqualsIsMathItalic() {
        let input = """
            1. Premise
                - $Q=R$
            """
        let output = MathPreprocessor.preprocess(input)
        #expect(!output.contains("$"))
        #expect(output.unicodeScalars.contains { $0.value >= 0x1D434 && $0.value <= 0x1D467 })
    }

    @Test func twoSpaceListItemMathStillTransforms() {
        let input = """
            - parent
              - $Q \\neq R$
            """
        let output = MathPreprocessor.preprocess(input)
        #expect(output.contains("swiftmath://i/"))
        #expect(!output.contains("$Q"))
    }

    @Test func nestedNumberedListMathStillTransforms() {
        let input = """
            1. Premise
                1. $x=1$
            """
        let output = MathPreprocessor.preprocess(input)
        #expect(!output.contains("$"))
        #expect(output.unicodeScalars.contains { $0.value >= 0x1D434 && $0.value <= 0x1D467 })
    }

    @Test func englishAndBetweenTwoDollarSpansStaysLiteral() {
        let input = "pick $x and $y"
        #expect(MathPreprocessor.preprocess(input) == input)
    }

    @Test func accidentallyIndentedMathProseDoesNotExposeProviderURLsAsCode() {
        let input = """
                The first term converges to $F(n)=\\frac{3}{4}+\\epsilon_n$.
                The second term converges to $0$ because $\\epsilon_n\\to0$.
            """
        let output = MathPreprocessor.preprocess(input)

        #expect(output.contains("The first term converges to "))
        #expect(output.contains("swiftmath://i/"))
        #expect(!output.split(separator: "\n").contains(where: {
            $0.hasPrefix("    ") && $0.contains("swiftmath://")
        }))
    }

    @Test func accidentallyIndentedBareLatexBecomesDisplayMath() {
        let latex = "F(n) = \\frac{1}{2} \\left(1 + \\frac{1}{n}\\right)"
        let output = MathPreprocessor.preprocess("    \(latex)")

        #expect(output.contains("swiftmath://d/"))
        #expect(payload(host: "d", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == latex)
        #expect(!output.contains("    F(n)"))
    }

    @Test func escapedParensBecomeInlineImage() {
        let output = MathPreprocessor.preprocess("Solve \\(x_1\\) now")
        #expect(output.contains("swiftmath://i/"))
        #expect(payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "x_1")
    }

    @Test func bareBoxedBecomesInlineImage() {
        let output = MathPreprocessor.preprocess("The answer is \\boxed{73682}.")
        #expect(output.contains("The answer is "))
        #expect(output.contains("swiftmath://i/"))
        #expect(payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "\\boxed{73682}")
    }

    @Test func dollarWrappedBoxedBecomesInlineImage() {
        let output = MathPreprocessor.preprocess("Answer: $\\boxed{73682}$")
        #expect(output.contains("swiftmath://i/"))
        #expect(payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "\\boxed{73682}")
    }

    @Test func displayBoxedKeepsBoxedCommand() {
        let output = MathPreprocessor.preprocess("$$\\boxed{x}$$")
        #expect(output.contains("swiftmath://d/"))
        #expect(payload(host: "d", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "\\boxed{x}")
    }

    @Test func bracketMathEmbeddedInTextLineRendersInline() {
        // The reported bug: "…in one line: \[ \boxed{…} \]" broke into its own
        // centered block; an embedded display span must stay on the text line.
        let output = MathPreprocessor.preprocess(
            "So the whole update rule in one line: \\[\\boxed{\\theta_{t+1} = \\theta_t - \\eta\\,\\nabla L(\\theta_t)}\\]")
        #expect(output.contains("So the whole update rule in one line: "))
        #expect(output.contains("swiftmath://i/"))
        #expect(!output.contains("swiftmath://d/"))
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

    @Test func boxedEmbeddedInDisplayMathIsKept() {
        let output = MathPreprocessor.preprocess("$$x = \\boxed{5}$$")
        #expect(payload(host: "d", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "x = \\boxed{5}")
    }

    @Test func nestedBracesInsideBoxedSurvive() {
        let output = MathPreprocessor.preprocess("\\boxed{\\frac{1}{2}}")
        #expect(payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL)
            == "\\boxed{\\frac{1}{2}}")
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

    @Test func boxedDisplayKeepsTfrac() {
        let output = MathPreprocessor.preprocess(
            "$$\\boxed{\\text{Batch } n \\text{ fee} = \\tfrac{3}{4} - \\frac{2n+3}{2(n+1)(n+2)}}$$")
        #expect(output.contains("swiftmath://d/"))
        let decoded = payload(host: "d", in: output).flatMap(MathPreprocessor.decodeBase64URL) ?? ""
        #expect(decoded.contains("\\tfrac{3}{4}"))
        #expect(decoded.contains("\\text{Batch }"))
        #expect(decoded.contains("\\boxed{"))
    }

    @Test func displayKeepsDfracAndLim() {
        let output = MathPreprocessor.preprocess(
            "$$\\lim_{N\\to\\infty} a(N) = \\frac34 \\cdot \\frac{6}{\\pi^2} = \\boxed{\\dfrac{9}{2\\pi^2}}$$")
        #expect(output.contains("swiftmath://d/"))
        let decoded = payload(host: "d", in: output).flatMap(MathPreprocessor.decodeBase64URL) ?? ""
        #expect(decoded.contains("\\dfrac{9}{2\\pi^2}"))
        #expect(decoded.contains("\\lim"))
        #expect(!output.contains("$"))
    }

    @Test func unicodeSigmaWithScriptsBecomesSumImage() {
        let output = MathPreprocessor.preprocess(
            "Fee of batch n = Σ_{k=1}^{n} 1/(k(k+2)).")
        #expect(output.contains("swiftmath://i/"))
        let decoded = payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL) ?? ""
        #expect(decoded.contains("\\sum_{k=1}^{n}"))
        #expect(!output.contains("Σ_{k=1}^{n}"))
    }

    @Test func unicodeSigmaNestedPredicateBecomesSumImage() {
        let output = MathPreprocessor.preprocess("Nested Σ_{n≤N, n not sf}")
        #expect(output.contains("swiftmath://i/"))
        let decoded = payload(host: "i", in: output).flatMap(MathPreprocessor.decodeBase64URL) ?? ""
        #expect(decoded.contains("\\sum_{n≤N, n not sf}"))
    }

    @Test func loneUnicodeSigmaStaysLiteral() {
        let input = "variance Σ is"
        #expect(MathPreprocessor.preprocess(input) == input)
        #expect(!MathPreprocessor.preprocess(input).contains("swiftmath://"))
    }

    @Test func q1SeedAnswerKeepsBareLogicUnicode() {
        let answer = """
            # Q1 — Logic
            - **P:** P ⇔ (Q = R), i.e., P ⇔ (Q ⇔ R)
            - **Q:** Q ⇔ (¬P ∨ S)
            - **S:** S ⇔ ¬R
            Q ⇔ (¬P ∨ S). With S = T, (¬P ∨ T) is always True.
            """
        let output = MathPreprocessor.preprocess(answer)
        #expect(output.contains("⇔"))
        #expect(output.contains("∨"))
        #expect(!output.contains("swiftmath://"))
    }

    @Test func katexCommandsRender() {
        let fallback = MathRenderer.render(
            base64URL: MathPreprocessor.base64URL("\\notARealLatexCommandZZZ"),
            display: false,
            fontSize: 28,
            color: .white)
        for latex in ["\\iff", "\\lor", "\\land", "\\neg", "\\implies", "\\tfrac{1}{2}", "\\boxed{1}"] {
            let image = MathRenderer.render(
                base64URL: MathPreprocessor.base64URL(latex),
                display: false,
                fontSize: 28,
                color: .white)
            #expect(image.size.width > 8, "\(latex) image too narrow")
            #expect(image.size.height > 8, "\(latex) image too short")
            #expect(
                image.size.width < fallback.size.width,
                "\(latex) should render as a symbol, not the raw command string")
        }
    }
}
