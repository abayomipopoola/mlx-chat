import AppKit
import CoreText
import Testing
@testable import MLXChat

@Suite struct FontCascadeTests {
    private let tofuScalars: [Unicode.Scalar] = [
        Unicode.Scalar(0x22BB)!, // ⊻ XOR
        Unicode.Scalar(0x21D4)!, // ⇔
        Unicode.Scalar(0x221E)!, // ∞
        Unicode.Scalar(0x2228)!, // ∨
        Unicode.Scalar(0x2227)!, // ∧
        Unicode.Scalar(0x21D2)!, // ⇒
        Unicode.Scalar(0x2207)!, // ∇
        Unicode.Scalar(0x00AC)!, // ¬
    ]

    @Test func monaVariableFontRegistersAndAppliesAxes() {
        Studio.registerChatFonts()
        let registered = NSFont(name: "MonaSansVF-Regular", size: 14)
            ?? NSFont(name: "Mona Sans VF", size: 14)
        #expect(registered != nil, "bundled MonaSansVF must register")

        let body = Studio.monaNSFont(size: Studio.chatFontSize)
        #expect(isMonaSansVF(body))
        #expect(body.pointSize == Studio.chatFontSize)
        #expect(Studio.monaNSFont(size: Studio.chatBodySize).pointSize == Studio.chatBodySize)
        if let bodyOpsz = axisValue(in: body, tag: "opsz") {
            #expect(bodyOpsz >= 12 && bodyOpsz <= 20)
        } else {
            Issue.record("body Mona Sans VF instance is missing opsz")
        }

        let headingSizes: [CGFloat] = [22, 18, 16]
        for size in headingSizes {
            let heading = Studio.monaNSFont(size: size, weight: .bold)
            #expect(isMonaSansVF(heading))
            #expect(heading.pointSize == size)
            if let opsz = axisValue(in: heading, tag: "opsz") {
                #expect(opsz == 40 || opsz == 72, "heading \(size)pt opsz should be display 40 or 72")
            } else {
                Issue.record("heading \(size)pt Mona Sans VF instance is missing opsz")
            }
            if let wght = axisValue(in: heading, tag: "wght") {
                #expect(wght == 700)
            }
        }

        let italic = Studio.monaNSFont(size: 14, italic: true)
        #expect(axisValue(in: italic, tag: "ital") == 1)
        #expect(axisValue(in: italic, tag: "wght") == 400)

        let boldItalic = Studio.monaNSFont(size: 14, weight: .bold, italic: true)
        #expect(axisValue(in: boldItalic, tag: "ital") == 1)
        #expect(axisValue(in: boldItalic, tag: "wght") == 700)
    }

    @Test func chromeSansIsSystemAndReadingFaceIsMona() {
        Studio.registerChatFonts()
        let chrome = Studio.systemNSFont(size: 13)
        #expect(!isMonaSansVF(chrome))
        #expect(chrome.pointSize == 13)

        let reading = Studio.monaNSFont(size: 13)
        #expect(isMonaSansVF(reading))
        #expect(reading.pointSize == 13)
    }

    @Test func cascadedComposerHasGlyphsForMathOperators() {
        Studio.registerChatFonts()
        let size = Studio.chatFontSize
        let base = Studio.monaNSFont(size: size)
        #expect(isMonaSansVF(base), "bundled MonaSansVF must register")
        let font = Studio.cascadedNSFont(base: base, size: size)
        for scalar in tofuScalars {
            #expect(
                hasGlyphAfterCascade(in: font, scalar: scalar),
                "cascaded composer missing glyph for U+\(String(scalar.value, radix: 16, uppercase: true))"
            )
        }
    }

    @Test func cascadedMonoHasGlyphsForMathOperators() {
        let font = Studio.cascadedNSFont(
            base: NSFont.monospacedSystemFont(ofSize: 12, weight: .regular),
            size: 12
        )
        for scalar in tofuScalars {
            #expect(
                hasGlyphAfterCascade(in: font, scalar: scalar),
                "cascaded mono missing glyph for U+\(String(scalar.value, radix: 16, uppercase: true))"
            )
        }
    }

    private func isMonaSansVF(_ font: NSFont) -> Bool {
        font.familyName == "Mona Sans VF" || font.fontName.contains("MonaSansVF")
    }

    private func axisValue(in font: NSFont, tag: String) -> CGFloat? {
        let key = Studio.fontAxisTag(tag)
        if let copied = CTFontCopyVariation(font as CTFont) as? [NSNumber: NSNumber],
           let value = copied[key]
        {
            return CGFloat(truncating: value)
        }
        if let attributes = font.fontDescriptor.object(forKey: .variation) as? [NSNumber: NSNumber],
           let value = attributes[key]
        {
            return CGFloat(truncating: value)
        }
        return nil
    }

    /// `CTFontGetGlyphsForCharacters` does not walk fallbacks; check the
    /// font and each cascade-list face for a BMP mapping.
    private func hasGlyphAfterCascade(in font: NSFont, scalar: Unicode.Scalar) -> Bool {
        if hasMappedGlyph(in: font, scalar: scalar) {
            return true
        }
        let cascade = font.fontDescriptor.object(forKey: .cascadeList) as? [NSFontDescriptor] ?? []
        for descriptor in cascade {
            guard let fallback = NSFont(descriptor: descriptor, size: font.pointSize) else {
                continue
            }
            if hasMappedGlyph(in: fallback, scalar: scalar) {
                return true
            }
        }
        return false
    }

    private func hasMappedGlyph(in font: NSFont, scalar: Unicode.Scalar) -> Bool {
        var chars = [UniChar(scalar.value)]
        var glyphs = [CGGlyph](repeating: 0, count: 1)
        return CTFontGetGlyphsForCharacters(font as CTFont, &chars, &glyphs, 1)
            && glyphs[0] != 0
    }
}
