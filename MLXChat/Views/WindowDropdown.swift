import SwiftUI

/// Shared state for the header dropdowns (model picker, prompt presets).
///
/// The panels are hosted at the window root (see RootView) — NOT in a
/// per-button overlay — because the header lives in a `.safeAreaInset` bar,
/// which clips its content to the bar region: any overlay attached there is
/// invisible below the bar. Root hosting also keeps the panels inside the
/// window, unlike `.popover`/`Menu` (separate windows that can flip above
/// the anchor and hang past the title bar).
@Observable
final class HeaderDropdown {
    enum Kind: Hashable {
        case modelPicker
        case promptPresets
    }

    /// Which dropdown is open; nil when all are closed.
    var open: Kind?
    /// Anchor frames in the "windowRoot" coordinate space, published by the
    /// buttons so the root can position panels beneath them.
    var anchors: [Kind: CGRect] = [:]
    /// Prompt being edited; the editor sheet is hosted at the window root so
    /// closing the dropdown does not tear down the sheet's host view.
    var editingPrompt: PromptPreset?

    func toggle(_ kind: Kind) {
        open = open == kind ? nil : kind
    }
}

/// Coordinate space of RootView's main layout; header buttons publish their
/// frames in it and the root-hosted panels are positioned in it.
let windowRootCoordinateSpace = "windowRoot"

extension View {
    /// Publishes this view's frame (in the windowRoot space) as the anchor
    /// for `kind`'s root-hosted dropdown panel.
    func dropdownAnchor(_ kind: HeaderDropdown.Kind, in dropdown: HeaderDropdown) -> some View {
        onGeometryChange(for: CGRect.self) { proxy in
            proxy.frame(in: .named(windowRootCoordinateSpace))
        } action: { frame in
            dropdown.anchors[kind] = frame
        }
    }

    /// Liquid-glass chrome shared by the root-hosted dropdown panels.
    /// Interactive regular glass (same as the capsule/header circles) gives
    /// the panel the full "liquid" character; 14pt radius reads as a popover.
    func dropdownChrome(width: CGFloat) -> some View {
        frame(width: width)
            .glassEffect(.regular.interactive(), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .shadow(color: .black.opacity(0.25), radius: 12, y: 4)
    }
}
