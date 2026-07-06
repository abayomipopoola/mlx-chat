import AppKit
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
    static let sidebarBackground = Color(lightHex: 0xF6F6F7, darkHex: 0x1E1E20)
    /// Detail pane surface.
    static let detailBackground = Color(lightHex: 0xFFFFFF, darkHex: 0x161618)
    /// Card surface.
    static let cardFill = Color(lightHex: 0xFFFFFF, darkHex: 0x232326)
    /// Search fields, composer pill, status rows.
    static let fieldFill = Color(lightHex: 0xF0F0F3, darkHex: 0x2A2A2E)
    /// 1px hairline on cards and fields.
    static let cardStroke = Color(lightHex: 0xE5E5E7, darkHex: 0x3A3A3E)
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
    static let radiusCard: CGFloat = 14
    static let radiusSuggestion: CGFloat = 16
    static let radiusComposer: CGFloat = 27
    static let radiusField: CGFloat = 8
    static let heroBadgeSize: CGFloat = 145
    static let heroBadgeRadius: CGFloat = 32
    static let haloSize: CGFloat = 330
    static let sendButtonSize: CGFloat = 42
    static let contentMaxWidth: CGFloat = 660
    static let sidebarWidth: CGFloat = 280
    /// Sidebar top inset: aligns "New chat" with the bottom edge of the detail
    /// pane's fixed header (16 top padding + 34 capsule + 4 bottom).
    static let trafficLightInset: CGFloat = 54

    // Type scale measured from MLX Studio captures (band scan + width calibration),
    // hero sizes taken a step down per user preference.
    static let titleXL = Font.system(size: 32, weight: .bold)
    static let heroSubtitle = Font.system(size: 18)
    static let pageTitle = Font.system(size: 17, weight: .bold)
    static let rowTitle = Font.system(size: 15, weight: .semibold)
    static let body13 = Font.system(size: 13)
    static let body14Semibold = Font.system(size: 14, weight: .semibold)
    static let cardCaption = Font.system(size: 12.5)
    /// Matches the transcript body (markdown 15pt) so questions, answers,
    /// and the composer read as one size.
    static let composerText = Font.system(size: 15)
    static let caption13 = Font.system(size: 13)
    static let sectionLabel = Font.system(size: 12, weight: .semibold)
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
}

/// Uppercase tracked section label above a card group.
struct SectionHeader: View {
    let title: String

    init(_ title: String) { self.title = title }

    var body: some View {
        Text(title.uppercased())
            .font(Studio.sectionLabel)
            .tracking(0.5)
            .foregroundStyle(Color.subtitleGray)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Rounded gray search field shared by the sidebar and Manage Models.
struct StudioSearchField: View {
    let placeholder: String
    @Binding var text: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 7)
        .background(
            RoundedRectangle(cornerRadius: Studio.radiusField, style: .continuous)
                .fill(Color.fieldFill))
    }
}

/// Small colored capsule for statuses and model spec badges.
struct CapsuleBadge: View {
    let text: String
    var tint: Color = .brandGreen
    var background: Color?

    var body: some View {
        Text(text)
            .font(.system(size: 11, weight: .semibold))
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

/// In-content page header for pushed pages: back chevron far left, centered title.
struct StudioPageHeader: View {
    let title: String
    /// Pops the page (custom routing; there is no NavigationStack to dismiss).
    var onBack: () -> Void = {}

    var body: some View {
        ZStack {
            HStack {
                Button {
                    onBack()
                } label: {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundStyle(.primary)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                Spacer()
            }
            Text(title)
                .font(Studio.pageTitle)
        }
        .padding(.horizontal, 24)
        .padding(.top, 20)
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
