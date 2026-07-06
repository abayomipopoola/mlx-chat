import AppKit
import FoundationModels
import SwiftUI

/// In-frame Settings page, arranged like MLX Studio:
/// MODELS → MEMORY → AI ENGINE → MODEL STORAGE → THEME.
/// White cards with hairline strokes.
struct SettingsPage: View {
    @Environment(ChatController.self) private var chat
    @Environment(UpdateController.self) private var updates
    /// Pops back to the chat (custom routing, no NavigationStack).
    var onBack: () -> Void = {}
    /// Pushes the Manage Models page.
    var onManageModels: () -> Void = {}

    @AppStorage(Keys.autoUnloadMinutes) private var autoUnloadMinutes = Keys.Defaults.autoUnloadMinutes
    @AppStorage(Keys.appearance) private var appearance = Keys.Defaults.appearance

    @State private var usedBytes: Int64 = 0
    @State private var availableBytes: Int64 = 0

    var body: some View {
        VStack(spacing: 0) {
            StudioPageHeader(title: "Settings", onBack: onBack)
            if let version = updates.availableVersion {
                updateBanner(version: version)
            }
            settingsContent
        }
        .background(Color.detailBackground)
        .onAppear {
            refreshUsage()
            updates.checkInBackground()
        }
    }

    /// Notifier shown above the settings when a newer version is on the appcast.
    private func updateBanner(version: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "arrow.down.circle.fill")
                .font(.system(size: 20))
                .foregroundStyle(Color.brandGreen)
            VStack(alignment: .leading, spacing: 1) {
                Text("MLX Chat \(version) is available")
                    .font(Studio.body14Semibold)
                Text("You're on \(updates.currentVersion) — the update installs in place.")
                    .font(Studio.caption13)
                    .foregroundStyle(Color.subtitleGray)
            }
            Spacer(minLength: 12)
            Button("Update Now") { updates.checkForUpdates() }
                .buttonStyle(.borderedProminent)
                .tint(Color.brandGreen)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.brandGreen.opacity(0.10))
        .overlay(alignment: .bottom) {
            Rectangle().fill(Color.cardStroke).frame(height: 1)
        }
    }

    private var settingsContent: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 26) {
                section("Models") {
                    card { manageModelsRow }
                }

                section("Memory") {
                    card { autoUnloadCard }
                }

                section("AI Engine") {
                    card(stroke: AppleIntelligenceEngine.isAvailable
                        ? Color.brandGreen.opacity(0.55) : .cardStroke,
                        padding: 0) {
                        appleIntelligenceCard
                    }
                }

                section("Model Storage") {
                    card { storageRows }
                }

                section("Theme") {
                    card { themeRow }
                }
            }
            .frame(maxWidth: Studio.contentMaxWidth)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 28)
            .padding(.vertical, 24)
        }
    }

    // MARK: - Layout helpers

    private func section(_ title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionHeader(title)
                .padding(.horizontal, 4)
            content()
        }
    }

    private func card(
        stroke: Color = .cardStroke,
        padding: CGFloat = 16,
        @ViewBuilder content: () -> some View
    ) -> some View {
        content()
            .frame(maxWidth: .infinity, alignment: .leading)
            .studioCard(stroke: stroke, padding: padding)
    }

    // MARK: - Models

    private var manageModelsRow: some View {
        Button {
            onManageModels()
        } label: {
            HStack(spacing: 12) {
                Image(systemName: "cpu")
                    .font(.title3)
                Text("Manage Models")
                    .font(Studio.rowTitle)
                Spacer()
                Text("\(chat.modelStore.downloadedCount) downloaded")
                    .foregroundStyle(.secondary)
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.tertiary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Memory (auto-unload)

    private var autoUnloadCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Image(systemName: "timer")
                    .font(.title3)
                    .foregroundStyle(Color.brandGreen)
                Text("Auto-Unload After")
                    .font(Studio.rowTitle)
                Spacer()
                Picker("", selection: $autoUnloadMinutes) {
                    Text("2 minutes").tag(2)
                    Text("10 minutes").tag(10)
                    Text("1 hour").tag(60)
                    Text("Never").tag(0)
                }
                .labelsHidden()
                .fixedSize()
                .tint(Color.brandGreen)
                .id(appearance)
            }
            Text("Frees RAM by unloading models after inactivity.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Apple Intelligence

    private var isAppleIntelligenceSelected: Bool {
        chat.selectedModelID == appleIntelligenceEngineID
    }

    private var aiSymbolName: String {
        NSImage(systemSymbolName: "apple.intelligence", accessibilityDescription: nil) != nil
            ? "apple.intelligence" : "sparkles"
    }

    private var appleIntelligenceCard: some View {
        let availabilityMessage = AppleIntelligenceEngine.availabilityMessage()
        return VStack(spacing: 0) {
            Button {
                if availabilityMessage == nil {
                    chat.selectedModelID = appleIntelligenceEngineID
                }
            } label: {
                HStack(spacing: 12) {
                    RoundedRectangle(cornerRadius: 9, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [.pink, .purple, .blue],
                                startPoint: .topLeading, endPoint: .bottomTrailing))
                        .frame(width: 44, height: 44)
                        .overlay {
                            Image(systemName: aiSymbolName)
                                .font(.system(size: 22))
                                .foregroundStyle(.white)
                        }

                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 8) {
                            Text("Apple Intelligence")
                                .font(Studio.rowTitle)
                            if availabilityMessage == nil {
                                CapsuleBadge(text: "Ready", tint: .statusGreen,
                                             background: Color.statusGreenBackground)
                            } else {
                                CapsuleBadge(text: "Off", tint: .iconOrange)
                                    .help(availabilityMessage ?? "")
                            }
                        }
                        Text(isAppleIntelligenceSelected ? "Using Apple Intelligence" : "Built-in AI, no download")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    if isAppleIntelligenceSelected {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.title2)
                            .foregroundStyle(Color.brandGreen)
                    }
                }
                .padding(16)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(availabilityMessage != nil)

            Divider()

            HStack(spacing: 6) {
                Image(systemName: "info.circle")
                Text(contextCaption)
            }
            .font(.callout)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
        }
    }

    private var contextCaption: String {
        if AppleIntelligenceEngine.isAvailable {
            return "\(AppleIntelligenceEngine.contextTokens) token context • Text only"
        }
        return "Text only"
    }

    // MARK: - Theme

    private var themeRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Image(systemName: "circle.lefthalf.filled")
                    .font(.title3)
                    .foregroundStyle(Color.brandGreen)
                Text("Appearance")
                    .font(Studio.rowTitle)
                Spacer()
                Picker("", selection: $appearance) {
                    Text("Auto").tag("system")
                    Text("Light").tag("light")
                    Text("Dark").tag("dark")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                // The control caches its appearance; rebuild it whenever the
                // app-wide override changes so the labels never go invisible.
                .id(appearance)
            }
            Text("Follows the system appearance by default.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Storage

    private var storageRows: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "folder")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                    .frame(width: 24)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Storage Location")
                        .font(Studio.rowTitle)
                    Text(ModelStore.hubCacheDirectory.path)
                        .font(.system(size: 13).monospaced())
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([ModelStore.hubCacheDirectory])
                }
                .buttonStyle(.plain)
                .font(Studio.body14Semibold)
                .foregroundStyle(Color.brandGreen)
            }

            Divider()

            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "internaldrive")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                    .frame(width: 24)
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Models: \(format(usedBytes))")
                            .font(Studio.rowTitle)
                        Spacer()
                        Button {
                            refreshUsage()
                        } label: {
                            Image(systemName: "arrow.clockwise")
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(Color.brandGreen)
                    }
                    ProgressView(value: usageFraction)
                        .tint(Color.brandGreen)
                    Text("\(format(availableBytes)) available")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var usageFraction: Double {
        let total = Double(usedBytes + availableBytes)
        guard total > 0 else { return 0 }
        return Double(usedBytes) / total
    }

    private func refreshUsage() {
        usedBytes = ModelStore.diskUsageBytes()
        availableBytes = ModelStore.availableBytes()
    }

    private func format(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
