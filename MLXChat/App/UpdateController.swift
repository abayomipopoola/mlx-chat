import AppKit
import Foundation
import Sparkle

/// One-click auto-update via Sparkle, fed by the project's GitHub Releases appcast
/// (`SUFeedURL` in Info.plist → `…/releases/latest/download/appcast.xml`).
///
/// Sparkle probes silently on launch (`checkForUpdateInformation`) so the app can
/// surface its *own* banner (`availableVersion`) above Settings and a menu item,
/// rather than Sparkle's scheduled pop-ups. Acting on either — the banner button or
/// "Check for Updates…" — calls `checkForUpdates()`, which runs Sparkle's standard
/// download → replace → relaunch flow.
///
/// The updater stays dormant until a maintainer runs `scripts/generate_keys` and
/// pastes the EdDSA public key into `SUPublicEDKey` (see docs/RELEASING.md); this
/// avoids Sparkle raising a "misconfigured" error on first launch of source builds.
@Observable
final class UpdateController: NSObject, SPUUpdaterDelegate {
    /// Non-nil when a newer version is available; carries its display version string.
    var availableVersion: String?

    @ObservationIgnored private var controller: SPUStandardUpdaterController?

    override init() {
        super.init()
        guard Self.isConfigured else { return }
        controller = SPUStandardUpdaterController(
            startingUpdater: true, updaterDelegate: self, userDriverDelegate: nil)
        // Updates surface through our banner, not Sparkle's own scheduled prompts.
        controller?.updater.automaticallyChecksForUpdates = false
    }

    /// The running app's version, e.g. "1.0.0".
    var currentVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "—"
    }

    /// True once a real EdDSA public key is present (not the placeholder).
    static var isConfigured: Bool {
        guard let key = Bundle.main.object(forInfoDictionaryKey: "SUPublicEDKey") as? String
        else { return false }
        return key.count > 40 && !key.contains("PASTE")
    }

    /// Silent probe used to drive the banner; shows no UI unless the user acts.
    func checkInBackground() {
        controller?.updater.checkForUpdateInformation()
    }

    /// Interactive check → Sparkle downloads, replaces the app, and relaunches.
    /// When dormant (no signing key yet in a source/debug build) there is no feed to
    /// query, so give an honest response instead of silently doing nothing.
    func checkForUpdates() {
        if let controller {
            controller.updater.checkForUpdates()
            return
        }
        let alert = NSAlert()
        alert.messageText = "You're up to date"
        alert.informativeText = "MLX Chat \(currentVersion) is the latest version. "
            + "Automatic updates activate once this build is signed for release."
        alert.runModal()
    }

    // MARK: SPUUpdaterDelegate (Sparkle invokes these on the main thread)

    /// Authoritative appcast location (also in Info.plist as `SUFeedURL`); setting it
    /// here keeps the feed correct even if the bundle key is ever missing.
    func feedURLString(for updater: SPUUpdater) -> String? {
        "https://github.com/abayomipopoola/mlx-chat/releases/latest/download/appcast.xml"
    }

    func updater(_ updater: SPUUpdater, didFindValidUpdate item: SUAppcastItem) {
        availableVersion = item.displayVersionString
    }

    func updaterDidNotFindUpdate(_ updater: SPUUpdater) {
        availableVersion = nil
    }
}
