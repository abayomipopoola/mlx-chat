import AppKit
import CoreText
import SwiftUI

// MARK: - Dynamic color plumbing

extension NSColor {
    /// Hex like 0x3E9979.
    convenience init(hex: UInt32, alpha: CGFloat = 1) {
        self.init(
            srgbRed: CGFloat((hex >> 16) & 0xFF) / 255,
            green: CGFloat((hex >> 8) & 0xFF) / 255,
            blue: CGFloat(hex & 0xFF) / 255,
            alpha: alpha)
    }
}

extension Color {
    /// Appearance-tracking color: resolves per-draw so it flips live with the system.
    init(light: NSColor, dark: NSColor) {
        self.init(nsColor: NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
        })
    }

    init(lightHex: UInt32, darkHex: UInt32, lightAlpha: CGFloat = 1, darkAlpha: CGFloat = 1) {
        self.init(
            light: NSColor(hex: lightHex, alpha: lightAlpha),
            dark: NSColor(hex: darkHex, alpha: darkAlpha))
    }
}

// MARK: - Palette (MLX Studio, pixel-sampled; dark variants derived)

extension Color {
    /// Interactive accent green: links, inline pickers, progress.
    static let brandGreen = Color(lightHex: 0x3E9979, darkHex: 0x4FB892)
    /// Composer send button: deeper green in dark mode so it sits with the theme.
    static let sendGreen = Color(lightHex: 0x3E9979, darkHex: 0x2F8563)
    /// Hero badge gradient endpoints.
    static let brandGradientTop = Color(lightHex: 0x42A484, darkHex: 0x4FB892)
    static let brandGradientBottom = Color(lightHex: 0x2E8E68, darkHex: 0x35986F)
    /// Status green used by "Ready"-style badges.
    static let statusGreen = Color(lightHex: 0x59C05F, darkHex: 0x6ECC74)
    static let statusGreenBackground = Color(lightHex: 0xE4F5E6, darkHex: 0x59C05F, darkAlpha: 0.18)
    /// Soft halo behind the hero badge.
    static let mintHalo = Color(lightHex: 0xE3F0EC, darkHex: 0x4FB892, darkAlpha: 0.12)
    /// Flat sidebar pane and gray capsules (model picker).
    static let sidebarBackground = Color(lightHex: 0xF6F6F7, darkHex: 0x18181A)
    /// Detail pane surface.
    static let detailBackground = Color(lightHex: 0xFFFFFF, darkHex: 0x141416)
    /// Card surface.
    static let cardFill = Color(lightHex: 0xFFFFFF, darkHex: 0x1E1E20)
    /// Search fields, composer pill, status rows.
    static let fieldFill = Color(lightHex: 0xF0F0F3, darkHex: 0x222226)
    /// Circle behind composer + / send, a step above the pill fill.
    static let composerButtonFill = Color(lightHex: 0xE2E2E6, darkHex: 0x3F3F46)
    /// 1px hairline on cards and fields.
    static let cardStroke = Color(lightHex: 0xE5E5E7, darkHex: 0x2E2E32)
    /// User message bubble.
    static let bubbleFill = Color(lightHex: 0xF6F6F7, darkHex: 0x2A2A2E)
    /// Secondary copy: hero subtitle, captions, section labels.
    static let subtitleGray = Color(lightHex: 0x8F8F8F, darkHex: 0x9A9AA0)
    /// Suggestion-card icon hues.
    static let iconPurple = Color(lightHex: 0xBB3FD8, darkHex: 0xCC5CE8)
    static let iconOrange = Color(lightHex: 0xF3903E, darkHex: 0xF7A55C)
    static let iconBlue = Color(lightHex: 0x2F7CF9, darkHex: 0x549BFF)
    /// Selected conversation row in the sidebar.
    static let sidebarSelection = Color(lightHex: 0xE4EEE9, darkHex: 0x2A3A33)
    /// Unselected history title, New Chat, and Settings. Darker than system
    /// secondary in Light so the list holds up on the gray pane.
    static let sidebarTitle = Color(lightHex: 0x3F3F44, darkHex: 0xC8C8CE)
    /// Timestamps and search chrome. Darker than system tertiary in Light.
    static let sidebarMeta = Color(lightHex: 0x5C5C62, darkHex: 0x8E8E93)
}

// MARK: - Appearance override

enum AppAppearance {
    /// Applies the stored appearance ("system" | "light" | "dark") app-wide.
    /// NSApp.appearance (unlike .preferredColorScheme(nil)) reliably returns
    /// to following the system, and repaints every window immediately.
    static func apply(_ raw: String) {
        switch raw {
        case "light": NSApp.appearance = NSAppearance(named: .aqua)
        case "dark": NSApp.appearance = NSAppearance(named: .darkAqua)
        default: NSApp.appearance = nil
        }
    }
}

// MARK: - Metrics & type scale

enum Studio {
    static let radiusCard: CGFloat = 12
    static let radiusSuggestion: CGFloat = 14
    static let radiusComposer: CGFloat = 24
    static let radiusField: CGFloat = 8
    static let heroBadgeSize: CGFloat = 145
    static let heroBadgeRadius: CGFloat = 32
    static let haloSize: CGFloat = 330
    /// Composer send and attach controls share this hit target.
    static let sendButtonSize: CGFloat = 28
    static let headerButtonSize: CGFloat = 28
    /// After the green light, before the sidebar toggle. Original 10, doubled.
    static let titlebarAfterLights: CGFloat = 20
    /// Between the sidebar toggle and back. Original 10, doubled.
    static let titlebarToggleToHistory: CGFloat = 20
    static let titlebarHistorySpacing: CGFloat = 2
    /// Top inset so 28pt cluster icons sit on the traffic-light line.
    static let titlebarClusterTop: CGFloat = 4
    /// Shared height for the model capsule and header icon chips.
    static let headerChipHeight: CGFloat = 30
    static let contentMaxWidth: CGFloat = 660
    static let sidebarWidth: CGFloat = 280
    /// Horizontal inset for New Chat, search, history, and Settings.
    static let sidebarInset: CGFloat = 8
    /// Bottom fade so history dissolves before Settings.
    static let sidebarListFade: CGFloat = 36
    /// Gap above the Settings row.
    static let sidebarSettingsTop: CGFloat = 20
    /// Sidebar top inset: aligns "New chat" with the bottom edge of the detail
    /// pane's fixed header (16 top padding + ~28 header control + 4 bottom).
    static let trafficLightInset: CGFloat = 48
    /// Width of the three traffic lights plus their leading window inset.
    static let trafficLightClusterWidth: CGFloat = 66
    /// Leading inset after the green traffic light (cluster + doubled gap).
    static let trafficLightClearance: CGFloat = trafficLightClusterWidth + titlebarAfterLights
    /// Space reserved at the top of the sidebar so New Chat sits below the
    /// window-chrome overlay (cluster top + icon + doubled gap).
    static let titlebarSidebarClearance: CGFloat = titlebarClusterTop + headerButtonSize + 24

    // Type scale measured from MLX Studio captures (band scan + width calibration),
    // hero sizes taken a step down per user preference.
    /// Chat and composer body. Antigravity uses Mona Sans at 14px / 1.6 line height.
    static let chatFontSize: CGFloat = 14
    /// Extra point on the composer textarea and the response pane only.
    static let monaPointBump: CGFloat = 1
    static var chatBodySize: CGFloat { chatFontSize + monaPointBump }
    static func cascadedNSFont(base: NSFont, size: CGFloat) -> NSFont {
        // System language cascade first (Apple's documented tofu fix), then
        // pin math and emoji faces so Mona/SF still find ⇔ ∨ ∇ ∞.
        var cascade: [NSFontDescriptor] = []
        if let defaults = CTFontCopyDefaultCascadeListForLanguages(base as CTFont, nil)
            as? [NSFontDescriptor]
        {
            cascade.append(contentsOf: defaults)
        }
        var seen = Set(cascade.compactMap(\.postscriptName))
        for name in ["Apple Symbols", "STIX Two Math", "Apple Color Emoji"] {
            guard NSFont(name: name, size: size) != nil else { continue }
            guard seen.insert(name).inserted else { continue }
            cascade.append(NSFontDescriptor(name: name, size: size))
        }
        let descriptor = base.fontDescriptor.addingAttributes([
            .cascadeList: cascade,
        ])
        return NSFont(descriptor: descriptor, size: size) ?? base
    }

    /// Mona Sans VF instance. Width stays 100. Weight maps to 400/500/600/700.
    /// Optical size is text (clamped 12–20) below 16pt and display (40 or 72) at 16pt+.
    static func monaNSFont(
        size: CGFloat,
        weight: NSFont.Weight = .regular,
        italic: Bool = false,
        opticalSize: CGFloat? = nil
    ) -> NSFont {
        let opsz = opticalSize ?? monaOpticalSize(for: size)
        let variation: [NSNumber: NSNumber] = [
            fontAxisTag("wdth"): 100,
            fontAxisTag("wght"): monaWeightAxis(weight),
            fontAxisTag("opsz"): NSNumber(value: Double(opsz)),
            fontAxisTag("ital"): italic ? 1 : 0,
        ]
        guard let registered = NSFont(name: "MonaSansVF-Regular", size: size)
            ?? NSFont(name: "Mona Sans VF", size: size)
        else {
            return NSFont.systemFont(ofSize: size, weight: weight)
        }
        let descriptor = registered.fontDescriptor.addingAttributes([
            .variation: variation,
        ])
        return NSFont(descriptor: descriptor, size: size) ?? registered
    }

    /// Text opsz tracks point size (clamped 12–20). Display uses 40 below 20pt and 72 at 20pt+.
    static func monaOpticalSize(for size: CGFloat) -> CGFloat {
        if size < 16 { return min(max(size, 12), 20) }
        if size < 20 { return 40 }
        return 72
    }

    private static func monaWeightAxis(_ weight: NSFont.Weight) -> NSNumber {
        let value: Int
        if weight >= .bold {
            value = 700
        } else if weight >= .semibold {
            value = 600
        } else if weight >= .medium {
            value = 500
        } else {
            value = 400
        }
        return NSNumber(value: value)
    }

    static func fontAxisTag(_ name: String) -> NSNumber {
        var tag: UInt32 = 0
        for byte in name.utf8 {
            tag = (tag << 8) | UInt32(byte)
        }
        return NSNumber(value: tag)
    }

    static func systemNSFont(size: CGFloat, weight: NSFont.Weight = .regular) -> NSFont {
        cascadedNSFont(
            base: NSFont.systemFont(ofSize: size, weight: weight),
            size: size
        )
    }

    /// Chrome face. Settings, menus, title bar, and sidebar controls use this.
    static func sans(_ size: CGFloat, weight: NSFont.Weight = .regular) -> Font {
        Font(systemNSFont(size: size, weight: weight) as CTFont)
    }

    /// Reading face. Composer, transcript markdown, and sidebar history titles.
    static func mona(_ size: CGFloat, weight: NSFont.Weight = .regular) -> Font {
        let base = monaNSFont(size: size, weight: weight)
        return Font(cascadedNSFont(base: base, size: base.pointSize) as CTFont)
    }

    /// Composer field and transcript body — one point above chrome.
    static func chatSans(_ size: CGFloat, weight: NSFont.Weight = .regular) -> Font {
        mona(size + monaPointBump, weight: weight)
    }

    static let titleXL = sans(28, weight: .semibold)
    static let heroSubtitle = sans(18)
    static let pageTitle = sans(17, weight: .bold)
    static let rowTitle = sans(15, weight: .semibold)
    static let body13 = sans(13)
    static let body14Semibold = sans(14, weight: .semibold)
    static let cardCaption = sans(12.5)
    static let composerText = chatSans(chatFontSize)
    static let chatMono = Font(
        cascadedNSFont(
            base: NSFont.monospacedSystemFont(ofSize: 12, weight: .regular),
            size: 12
        ) as CTFont
    )
    static let caption13 = sans(13)
    static let sectionLabel = sans(12, weight: .semibold)

    /// Registers the bundled Mona Sans variable font so chat type can instantiate it.
    static func registerChatFonts() {
        let url = Bundle.main.url(forResource: "MonaSansVF", withExtension: "ttf")
            ?? Bundle.main.url(forResource: "MonaSansVF", withExtension: "ttf", subdirectory: "Fonts")
        if let url {
            CTFontManagerRegisterFontsForURL(url as CFURL, .process, nil)
        }
    }
}

// MARK: - Reusable components

/// White/dark card with hairline stroke — the base surface of every Studio card.
struct StudioCardModifier: ViewModifier {
    var stroke: Color = .cardStroke
    var padding: CGFloat = 16
    var radius: CGFloat = Studio.radiusCard

    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .fill(Color.cardFill))
            .overlay(
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .strokeBorder(stroke, lineWidth: 1))
    }
}

extension View {
    func studioCard(
        stroke: Color = .cardStroke,
        padding: CGFloat = 16,
        radius: CGFloat = Studio.radiusCard
    ) -> some View {
        modifier(StudioCardModifier(stroke: stroke, padding: padding, radius: radius))
    }

    /// Quiet fill + hairline. Used instead of Liquid Glass on standing chrome
    /// so sidebar animation and overlapping controls do not leave ghost copies.
    func studioQuietFill<S: InsettableShape>(in shape: S) -> some View {
        background(shape.fill(Color.fieldFill))
            .overlay(shape.strokeBorder(Color.cardStroke, lineWidth: 1))
    }

    /// Icon chip that matches the model-picker capsule fill, stroke, and height.
    func studioHeaderIconChip() -> some View {
        frame(width: 16, height: 16)
            .frame(width: Studio.headerChipHeight, height: Studio.headerChipHeight)
            .studioQuietFill(in: Capsule())
            .contentShape(Capsule())
    }

    /// Pointing hand while the pointer is over an enabled control.
    func pointingHandCursor(enabled: Bool = true) -> some View {
        onHover { inside in
            if inside && enabled {
                NSCursor.pointingHand.push()
            } else {
                NSCursor.pop()
            }
        }
    }
}

/// Fill appears only while hovered or pressed — no idle chrome.
struct QuietHoverFillButtonStyle: ButtonStyle {
    var cornerRadius: CGFloat = 10

    func makeBody(configuration: Configuration) -> some View {
        QuietHoverFillLabel(configuration: configuration, cornerRadius: cornerRadius)
    }
}

private struct QuietHoverFillLabel: View {
    let configuration: ButtonStyle.Configuration
    var cornerRadius: CGFloat
    @State private var hovered = false

    var body: some View {
        configuration.label
            .background {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(Color.fieldFill.opacity(hovered || configuration.isPressed ? 1 : 0))
            }
            .onHover { hovered = $0 }
    }
}

/// Uppercase tracked section label above a card group.
struct SectionHeader: View {
    let title: String

    init(_ title: String) { self.title = title }

    var body: some View {
        Text(title)
            .font(Studio.sectionLabel)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Mutable bag for the click-outside NSEvent monitor (View is a value type).
private final class SearchFieldClickOutsideContext {
    var fieldFrame: CGRect = .zero
    var keyWindow: NSWindow?
    var clearFocus: (() -> Void)?
}

/// Liquid-glass search field shared by the sidebar and Manage Models.
struct StudioSearchField: View {
    let placeholder: String
    @Binding var text: String
    /// Sidebar keeps the compact 7; Manage Models matches button height (10).
    var verticalPadding: CGFloat = 7
    /// Sidebar keeps 8; Manage Models uses a roomier inset.
    var horizontalPadding: CGFloat = 8
    /// Optional accent ring around the pill (Manage Models brand-green ring).
    var ringColor: Color? = nil
    /// When true, the ring only appears while the field is focused (sidebar).
    var ringOnFocusOnly: Bool = false
    /// Optional accessibility id for the inner TextField (UI tests).
    var accessibilityIdentifier: String? = nil
    /// Magnifying-glass tint. Sidebar uses a darker Light-mode gray.
    var chromeForeground: Color = .secondary

    @FocusState private var focused: Bool
    @State private var fieldFrame: CGRect = .zero
    @State private var clickMonitor: Any?
    @State private var clickOutsideContext = SearchFieldClickOutsideContext()

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(chromeForeground)
            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
                .font(Studio.sans(13))
                .focused($focused)
                .modifier(OptionalAccessibilityIdentifier(accessibilityIdentifier))
        }
        .padding(.horizontal, horizontalPadding)
        .padding(.vertical, verticalPadding)
        .studioQuietFill(in: RoundedRectangle(cornerRadius: Studio.radiusComposer, style: .continuous))
        .onGeometryChange(for: CGRect.self) { proxy in
            proxy.frame(in: .global)
        } action: { frame in
            fieldFrame = frame
            clickOutsideContext.fieldFrame = frame
        }
        .onChange(of: focused) { _, isFocused in
            syncClickOutsideMonitor(isFocused: isFocused)
        }
        .onDisappear {
            removeClickOutsideMonitor()
        }
        .onAppear {
            // AppKit makes the first key view (this field) first responder when
            // the window keys — AFTER onAppear. The ring is a clicked-into
            // indicator, so clear the launch auto-focus on a short delay.
            if ringOnFocusOnly {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { focused = false }
            }
        }
        .overlay {
            if ringOnFocusOnly, focused {
                RoundedRectangle(cornerRadius: Studio.radiusComposer, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.22), lineWidth: 1)
            } else if let ringColor, !ringOnFocusOnly {
                RoundedRectangle(cornerRadius: Studio.radiusComposer, style: .continuous)
                    .strokeBorder(ringColor, lineWidth: 1)
            }
        }
    }

    private func syncClickOutsideMonitor(isFocused: Bool) {
        removeClickOutsideMonitor()
        guard ringOnFocusOnly, isFocused else { return }

        clickOutsideContext.fieldFrame = fieldFrame
        clickOutsideContext.keyWindow = NSApp.keyWindow
        clickOutsideContext.clearFocus = { focused = false }

        let context = clickOutsideContext
        clickMonitor = NSEvent.addLocalMonitorForEvents(
            matching: [.leftMouseDown, .rightMouseDown, .otherMouseDown]
        ) { event in
            if Self.clickIsOutsideSearchField(event: event, context: context) {
                context.clearFocus?()
            }
            return event
        }
    }

    private func removeClickOutsideMonitor() {
        if let clickMonitor {
            NSEvent.removeMonitor(clickMonitor)
            self.clickMonitor = nil
        }
        clickOutsideContext.clearFocus = nil
        clickOutsideContext.keyWindow = nil
    }

    /// AppKit window coords are bottom-left; SwiftUI `.global` is top-left in the content.
    private static func clickIsOutsideSearchField(
        event: NSEvent,
        context: SearchFieldClickOutsideContext
    ) -> Bool {
        guard let eventWindow = event.window else { return true }
        // Menus and other panels use a different window than the one we captured
        // when the field focused; treat those as outside.
        if let keyWindow = context.keyWindow, eventWindow !== keyWindow {
            return true
        }
        guard let contentHeight = eventWindow.contentView?.bounds.height else { return true }
        let flippedY = contentHeight - event.locationInWindow.y
        let point = CGPoint(x: event.locationInWindow.x, y: flippedY)
        return !context.fieldFrame.contains(point)
    }
}

/// Applies `.accessibilityIdentifier` only when a non-nil id is provided.
private struct OptionalAccessibilityIdentifier: ViewModifier {
    let id: String?

    init(_ id: String?) { self.id = id }

    @ViewBuilder
    func body(content: Content) -> some View {
        if let id {
            content.accessibilityIdentifier(id)
        } else {
            content
        }
    }
}

/// Small colored capsule for statuses and model spec badges.
struct CapsuleBadge: View {
    let text: String
    var tint: Color = .brandGreen
    var background: Color?

    var body: some View {
        Text(text)
            .font(Studio.sans(11, weight: .semibold))
            .foregroundStyle(tint)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(Capsule().fill(background ?? tint.opacity(0.14)))
    }
}

/// SF icon inside a tinted circle (model rows, download rows).
struct TintedCircleIcon: View {
    let systemName: String
    var tint: Color = .brandGreen
    var size: CGFloat = 36

    var body: some View {
        Image(systemName: systemName)
            .font(.system(size: size * 0.42, weight: .semibold))
            .foregroundStyle(tint)
            .frame(width: size, height: size)
            .background(Circle().fill(tint.opacity(0.14)))
    }
}

/// In-content page header: centered title. Window back/forward live in the
/// root overlay, so this header does not host chrome or shift with the sidebar.
struct StudioPageHeader: View {
    let title: String

    var body: some View {
        Text(title)
            .font(Studio.pageTitle)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 24)
            .padding(.top, 16)
            .padding(.bottom, 12)
    }
}

/// Green-gradient rounded square with the large + small sparkle pair
/// (MLX Studio's hero mark).
struct HeroBadge: View {
    var size: CGFloat = Studio.heroBadgeSize

    var body: some View {
        RoundedRectangle(cornerRadius: size * Studio.heroBadgeRadius / Studio.heroBadgeSize,
                         style: .continuous)
            .fill(
                LinearGradient(
                    colors: [.brandGradientTop, .brandGradientBottom],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing))
            .frame(width: size, height: size)
            .overlay(
                ZStack {
                    Image(systemName: "sparkle")
                        .font(.system(size: size * 0.40, weight: .medium))
                        .foregroundStyle(.white)
                        .offset(x: -size * 0.02, y: size * 0.03)
                    Image(systemName: "sparkle")
                        .font(.system(size: size * 0.13, weight: .semibold))
                        .foregroundStyle(.white)
                        .offset(x: size * 0.21, y: -size * 0.20)
                })
    }
}
