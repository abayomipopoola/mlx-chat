import SwiftUI

enum ModelDownloadMode: String, Hashable {
    case mlx
    case ssdStreamed

    static let defaultMode: ModelDownloadMode = .mlx

    static func availableModes(for modelID: String) -> [ModelDownloadMode] {
        StreamedMoEInstall.supportsSSDStreaming(sourceModelID: modelID)
            ? [.mlx, .ssdStreamed]
            : [.mlx]
    }
}

/// One download flow for all catalog models. MLX is always selected first;
/// the SSD option only appears for the exact MoE checkpoint TurboFieldfare supports.
struct ModelDownloadOptionsSheet: View {
    @Environment(ChatController.self) private var chat
    @Environment(\.dismiss) private var dismiss

    let model: CatalogModel
    @State private var selectedMode: ModelDownloadMode
    @State private var registrationError: String?
    @AppStorage(Keys.streamedModelContextTokens)
    private var streamedContextTokens = Keys.Defaults.streamedModelContextTokens
    @AppStorage(Keys.streamedModelExpertCacheSlots)
    private var streamedExpertCacheSlots = Keys.Defaults.streamedModelExpertCacheSlots

    init(model: CatalogModel, initialMode: ModelDownloadMode = .defaultMode) {
        self.model = model
        _selectedMode = State(initialValue: initialMode)
    }

    private var store: ModelStore { chat.modelStore }
    private var streamedVariant: StreamedMoEVariant? {
        StreamedMoEInstall.variant(forSourceModelID: model.id)
    }
    private var supportsSSDStreaming: Bool { streamedVariant != nil }
    private var streamedModelID: String? { streamedVariant?.modelID }
    private var isSSDReady: Bool {
        streamedModelID.map(store.isAvailable) ?? false
    }
    private var isSSDDownloading: Bool {
        streamedModelID.map(store.isDownloading) ?? false
    }
    private var managedSSDInstallState: StreamedManagedInstallState {
        streamedModelID.map(store.managedStreamedInstallState) ?? .none
    }

    private var sheetTitle: String {
        store.isAvailable(model.id) || isSSDReady || managedSSDInstallState == .complete
            ? "Use \(model.displayName)"
            : "Download \(model.displayName)"
    }

    private var streamedStatusTitle: String {
        if isSSDReady { return "Ready to use" }
        switch managedSSDInstallState {
        case .complete: return "Install found"
        case .resumable: return "Saved download"
        case .none: return "Automatic setup"
        }
    }

    private var streamedStatusIcon: String {
        if isSSDReady { return "checkmark.circle.fill" }
        switch managedSSDInstallState {
        case .complete: return "externaldrive.fill"
        case .resumable: return "arrow.clockwise.circle.fill"
        case .none: return "sparkles"
        }
    }

    private var ssdPrimaryActionTitle: String {
        if isSSDReady || managedSSDInstallState == .complete {
            return "Use SSD Model"
        }
        return managedSSDInstallState == .resumable
            ? "Resume SSD Download" : "Download SSD Model"
    }

    var body: some View {
        VStack(spacing: 0) {
            Text(sheetTitle)
                .font(Studio.sans(13, weight: .semibold))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)

            Divider()

            ScrollView {
                VStack(spacing: 14) {
                    hero
                    modeChoices
                    if selectedMode == .ssdStreamed {
                        streamedRuntimeControls
                        streamedStatus
                    }
                }
                .padding(18)
            }

            Divider()
            footer
        }
        .frame(width: 540, height: supportsSSDStreaming ? 720 : 470)
        .onAppear {
            validateStreamedRuntimeSettings()
            store.refreshStorageStats()
        }
        .onChange(of: isSSDReady) { wasReady, isReady in
            ssdReadinessChanged(wasReady: wasReady, isReady: isReady)
        }
        .onChange(of: streamedContextTokens) { _, _ in
            streamedRuntimeSettingsChanged()
        }
        .onChange(of: streamedExpertCacheSlots) { _, _ in
            streamedRuntimeSettingsChanged()
        }
        .alert(
            "Couldn’t use model",
            isPresented: Binding(
                get: { registrationError != nil },
                set: { if !$0 { registrationError = nil } })
        ) {
            Button("OK", role: .cancel) { registrationError = nil }
        } message: {
            Text(registrationError ?? "Unknown error")
        }
    }

    private var streamedRuntimeControls: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Runtime balance", systemImage: "memorychip")
                .font(Studio.body14Semibold)

            VStack(alignment: .leading, spacing: 7) {
                HStack {
                    Text("Context")
                        .font(Studio.body13)
                    Spacer()
                    Text("Larger values use more RAM for the FP16 KV cache.")
                        .font(Studio.sans(10))
                        .foregroundStyle(.secondary)
                }
                Picker("Context", selection: $streamedContextTokens) {
                    ForEach(StreamedMoERuntimeSettings.allowedContextTokens, id: \.self) { tokens in
                        Text("\(tokens / 1_024)K").tag(tokens)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .disabled(chat.isStreaming)
                .accessibilityIdentifier("ssdContextPicker")
            }

            VStack(alignment: .leading, spacing: 7) {
                HStack {
                    Text("Expert cache")
                        .font(Studio.body13)
                    Spacer()
                    Text("More slots reduce SSD reads and keep more experts in RAM.")
                        .font(Studio.sans(10))
                        .foregroundStyle(.secondary)
                }
                Picker("Expert cache", selection: $streamedExpertCacheSlots) {
                    ForEach(StreamedMoERuntimeSettings.allowedExpertCacheSlots, id: \.self) { slots in
                        Text("\(slots)").tag(slots)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .disabled(chat.isStreaming)
                .accessibilityIdentifier("ssdExpertCachePicker")
            }

            Text("The two controls are independent. Changes reload only the SSD-streamed engine; standard MLX models are unchanged.")
                .font(Studio.sans(10))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .studioCard(stroke: .iconBlue.opacity(0.45))
    }

    private var hero: some View {
        VStack(spacing: 8) {
            Image(systemName: selectedMode == .ssdStreamed
                ? "externaldrive.fill" : "shippingbox.fill")
                .font(.system(size: 21, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 48, height: 48)
                .background(Circle().fill(selectedMode == .ssdStreamed
                    ? Color.iconBlue : Color.brandGreen))
            Text(model.displayName)
                .font(Studio.sans(21, weight: .bold))
            Text(supportsSSDStreaming
                ? "Choose how this MoE model runs. MLX is the default; SSD streaming is an optional low-memory layout."
                : "This model uses the standard MLX download and runtime.")
                .font(Studio.body13)
                .foregroundStyle(Color.subtitleGray)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 430)
        }
        .padding(.bottom, 2)
    }

    private var modeChoices: some View {
        VStack(spacing: 10) {
            DownloadModeCard(
                icon: "shippingbox.fill",
                tint: .brandGreen,
                title: "Standard MLX",
                subtitle: "Loads the normal model files through MLX. Best default when memory is available.",
                badges: ["Default", "MLX", model.quantLabel],
                selected: selectedMode == .mlx,
                accessibilityIdentifier: "downloadMode.mlx") {
                    selectedMode = .mlx
                }

            if supportsSSDStreaming {
                DownloadModeCard(
                    icon: "externaldrive.fill",
                    tint: .iconBlue,
                    title: "SSD-Streamed",
                    subtitle: "Keeps the shared core and KV cache in memory, then reads routed experts from SSD.",
                    badges: ["Opt in", "Low memory", "MoE only"],
                    selected: selectedMode == .ssdStreamed,
                    accessibilityIdentifier: "downloadMode.ssd") {
                        selectedMode = .ssdStreamed
                    }
            }
        }
    }

    @ViewBuilder
    private var streamedStatus: some View {
        if let streamedVariant {
            VStack(alignment: .leading, spacing: 10) {
            Label(streamedStatusTitle, systemImage: streamedStatusIcon)
                .font(Studio.body14Semibold)
                .foregroundStyle(isSSDReady ? Color.brandGreen : .primary)

            if let progress = store.streamedInstallProgress(
                for: streamedVariant.modelID), isSSDDownloading {
                Text(progress.title)
                    .font(Studio.body13)
                    .foregroundStyle(Color.subtitleGray)
                ProgressView(value: progress.fraction ?? 0.01)
                    .progressViewStyle(.linear)
                    .tint(Color.iconBlue)
                    .accessibilityIdentifier("ssdInstallProgress")
                if let downloaded = progress.downloadedBytes,
                   let total = progress.totalBytes {
                    Text("\(Self.storage(downloaded)) of \(Self.storage(total))")
                        .font(Studio.sans(10).monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            } else if isSSDReady {
                Text("MLX Chat validated the streamed layout. The install stays on disk and only the experts needed for each token enter memory.")
                    .font(Studio.body13)
                    .foregroundStyle(Color.subtitleGray)
                    .fixedSize(horizontal: false, vertical: true)
            } else if managedSSDInstallState == .complete {
                Text("MLX Chat found managed .gturbo files that are not active. Use the SSD model to validate and register them, or remove them from Manage Models.")
                    .font(Studio.body13)
                    .foregroundStyle(Color.subtitleGray)
                    .fixedSize(horizontal: false, vertical: true)
            } else if managedSSDInstallState == .resumable {
                Text("MLX Chat found a saved SSD download. Resume it to fetch only the remaining verified ranges, or remove it from Manage Models.")
                    .font(Studio.body13)
                    .foregroundStyle(Color.subtitleGray)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("MLX Chat will download, repack, verify, register, and select the SSD model for you. No Terminal or setup command is needed. The download can resume after cancellation.")
                    .font(Studio.body13)
                    .foregroundStyle(Color.subtitleGray)
                    .fixedSize(horizontal: false, vertical: true)
                Text("Download: about \(Self.storage(streamedVariant.approximateDownloadBytes)) · Free: \(Self.storage(UInt64(max(store.availableBytes, 0))))")
                    .font(Studio.sans(10))
                    .foregroundStyle(.secondary)
            }

            if let error = store.streamedInstallError(
                for: streamedVariant.modelID) {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(Studio.sans(10))
                    .foregroundStyle(Color.iconOrange)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if !isSSDDownloading {
                Button("Use Existing .gturbo Folder…") { chooseInstall() }
                    .buttonStyle(.bordered)
                    .font(Studio.sans(10, weight: .semibold))
                    .accessibilityIdentifier("chooseExistingStreamedInstall")
            }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .studioCard(stroke: isSSDReady ? .brandGreen.opacity(0.45) : .cardStroke)
        }
    }

    private var footer: some View {
        HStack {
            Button(isSSDDownloading ? "Close" : "Cancel") { dismiss() }
                .keyboardShortcut(.cancelAction)
            Spacer()
            if selectedMode == .mlx {
                Button(store.isAvailable(model.id) ? "Use MLX Model" : "Download MLX Model") {
                    useOrDownloadMLX()
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
                .tint(Color.brandGreen)
                .accessibilityIdentifier("confirmModelDownload")
            } else if isSSDDownloading {
                Button("Cancel Download", role: .destructive) {
                    if let streamedModelID {
                        store.cancelDownload(streamedModelID)
                    }
                }
                .accessibilityIdentifier("cancelSSDDownload")
            } else {
                Button(ssdPrimaryActionTitle) {
                    useOrDownloadSSD()
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
                .tint(Color.iconBlue)
                .accessibilityIdentifier("confirmModelDownload")
            }
        }
        .padding(14)
    }

    private func chooseInstall() {
        guard let streamedVariant, let streamedModelID else { return }
        guard let directory = StreamedModelPicker.pickDirectory(
            for: streamedVariant) else { return }
        registrationError = store.registerStreamedModelDirectory(
            modelID: streamedModelID,
            directory: directory)
        if registrationError == nil {
            chat.selectModel(streamedModelID)
            dismiss()
        }
    }

    private func useOrDownloadMLX() {
        if store.isAvailable(model.id) {
            chat.selectModel(model.id)
        } else {
            store.download(model.id)
        }
        dismiss()
    }

    private func useOrDownloadSSD() {
        guard let streamedModelID else { return }
        if isSSDReady {
            let wasSelected = chat.selectedModelID == streamedModelID
            chat.selectModel(streamedModelID)
            if wasSelected {
                chat.preloadSelectedModelIfAvailable()
            }
            dismiss()
        } else {
            store.installStreamedModel(streamedModelID)
        }
    }

    private func ssdReadinessChanged(wasReady: Bool, isReady: Bool) {
        guard selectedMode == .ssdStreamed, !wasReady, isReady else { return }
        guard let streamedModelID else { return }
        chat.selectModel(streamedModelID)
        dismiss()
    }

    private func validateStreamedRuntimeSettings() {
        let settings = StreamedMoERuntimeSettings.current()
        streamedContextTokens = settings.contextTokens
        streamedExpertCacheSlots = settings.expertCacheSlots
    }

    private func streamedRuntimeSettingsChanged() {
        chat.runtime.streamedModelSettingsDidChange()
    }

    private static func storage(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(clamping: bytes), countStyle: .file)
    }
}

private struct DownloadModeCard: View {
    let icon: String
    let tint: Color
    let title: String
    let subtitle: String
    let badges: [String]
    let selected: Bool
    let accessibilityIdentifier: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 14) {
                Image(systemName: icon)
                    .font(.system(size: 19, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 44, height: 44)
                    .background(Circle().fill(tint))

                VStack(alignment: .leading, spacing: 5) {
                    Text(title).font(Studio.rowTitle)
                    Text(subtitle)
                        .font(Studio.body13)
                        .foregroundStyle(Color.subtitleGray)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 6) {
                        ForEach(badges, id: \.self) { badge in
                            CapsuleBadge(text: badge, tint: tint)
                        }
                    }
                }

                Spacer(minLength: 8)
                Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                    .font(.title3)
                    .foregroundStyle(selected ? tint : Color.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
            .studioCard(stroke: selected ? tint : .cardStroke, padding: 14)
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(accessibilityIdentifier)
        .accessibilityValue(selected ? "Selected" : "Not selected")
    }
}

#Preview("MoE download choices") {
    ModelDownloadOptionsSheet(model: ModelCatalog.model(
        for: StreamedMoEInstall.standard.sourceRepoID)!)
        .environment(ChatController())
}
