import AppKit

/// Owns the open-panel setup for registering an external `.gturbo` install.
@MainActor
enum StreamedModelPicker {
    static func pickDirectory(for variant: StreamedMoEVariant) -> URL? {
        let panel = NSOpenPanel()
        panel.title = "Choose \(variant.displayName) .gturbo Install"
        panel.message = "Choose the completed model directory created by TurboFieldfareRepack."
        panel.prompt = "Use Model"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK else { return nil }
        return panel.url
    }
}
