import SwiftUI

/// In-frame navigation targets shown in the detail column.
enum AppRoute: Hashable {
    case settings
    case models
    case modelFamily(String)
}

/// Window back / forward and the Settings/Models stack. One observable so
/// the title-bar cluster does not rebuild on unrelated RootView body passes.
@Observable
final class WindowNavigation {
    var path: [AppRoute] = []
    var forwardStack: [AppRoute] = []

    var canGoBack: Bool { !path.isEmpty }
    var canGoForward: Bool { !forwardStack.isEmpty }
    var current: AppRoute? { path.last }

    var familyID: String? {
        if case .modelFamily(let id) = current { return id }
        return nil
    }

    var isOnModels: Bool {
        switch current {
        case .models, .modelFamily: return true
        default: return false
        }
    }

    func push(_ route: AppRoute) {
        guard path.last != route else { return }
        path.append(route)
        forwardStack.removeAll()
    }

    func goBack() {
        guard let last = path.popLast() else { return }
        forwardStack.append(last)
    }

    func goForward() {
        guard let next = forwardStack.popLast() else { return }
        path.append(next)
    }

    func goHome() {
        path.removeAll()
        forwardStack.removeAll()
    }
}

/// Sidebar show/hide control that sits beside the traffic lights.
struct SidebarToggleButton: View {
    @AppStorage(Keys.sidebarCollapsed) private var sidebarCollapsed = Keys.Defaults.sidebarCollapsed

    var body: some View {
        Button {
            sidebarCollapsed.toggle()
        } label: {
            Image(systemName: "sidebar.left")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(Color.sidebarTitle)
                .frame(width: Studio.headerButtonSize, height: Studio.headerButtonSize)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(sidebarCollapsed ? "Show Sidebar (⌃⌘S)" : "Hide Sidebar (⌃⌘S)")
    }
}

/// Single window-chrome strip: sidebar toggle, then back / forward.
/// Hosted once on RootView so every page shares the same controls.
struct TitlebarWindowCluster: View {
    var navigation: WindowNavigation

    var body: some View {
        HStack(spacing: Studio.titlebarToggleToHistory) {
            SidebarToggleButton()
            TitlebarHistoryButtons(navigation: navigation)
            TitlebarUpdatePill()
        }
        .padding(.leading, Studio.trafficLightClearance)
        .padding(.top, Studio.titlebarClusterTop)
    }
}

/// Compact install capsule, half the title-bar icon size, shown only when
/// Sparkle reports a ready update.
struct TitlebarUpdatePill: View {
    @Environment(UpdateController.self) private var updates

    var body: some View {
        if let version = updates.availableVersion {
            Button {
                updates.checkForUpdates()
            } label: {
                Image(systemName: "arrow.down.to.line")
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 20, height: 14)
                    .background(Capsule().fill(Color.sendGreen))
                    .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .help("Install MLX Chat \(version)")
            .accessibilityLabel("Install MLX Chat \(version)")
            .accessibilityIdentifier("installUpdateButton")
        }
    }
}

/// Back / forward for in-window Settings and Models.
struct TitlebarHistoryButtons: View {
    var navigation: WindowNavigation

    var body: some View {
        HStack(spacing: Studio.titlebarHistorySpacing) {
            historyButton(
                systemName: "arrow.left",
                enabled: navigation.canGoBack,
                help: "Back",
                identifier: "backButton",
                action: navigation.goBack)
            .keyboardShortcut("[", modifiers: .command)

            historyButton(
                systemName: "arrow.right",
                enabled: navigation.canGoForward,
                help: "Forward",
                identifier: "titlebarForwardButton",
                action: navigation.goForward)
            .keyboardShortcut("]", modifiers: .command)
        }
    }

    private func historyButton(
        systemName: String,
        enabled: Bool,
        help: String,
        identifier: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Color.sidebarTitle)
                .opacity(enabled ? 1 : 0.35)
                .frame(width: Studio.headerButtonSize, height: Studio.headerButtonSize)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .help(help)
        .accessibilityLabel(help)
        .accessibilityIdentifier(identifier)
    }
}
