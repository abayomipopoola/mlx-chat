import SwiftUI

// MARK: - Import dialog (compact MLX Studio pattern)

struct ImportModelSheet: View {
    @Environment(ChatController.self) private var chat
    @Environment(\.dismiss) private var dismiss

    @State private var input = ""
    @State private var importError: String?
    @State private var importing = false
    @FocusState private var repoFieldFocused: Bool

    /// Accepts a bare repo id or a full Hugging Face URL.
    private var normalizedRepoID: String {
        Self.normalizeRepoID(input)
    }

    /// Cheap gate for the Import button — avoids full URL normalization per keystroke.
    private var inputIsEmpty: Bool {
        input.allSatisfy(\.isWhitespace)
    }

    var body: some View {
        VStack(spacing: 0) {
            Text("Import Custom Model")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)

            Divider()

            ScrollView {
                VStack(spacing: 16) {
                    Image(systemName: "arrow.down")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(.white)
                        .frame(width: 44, height: 44)
                        .background(Circle().fill(Color.brandGreen))
                        .padding(.top, 18)

                    VStack(spacing: 4) {
                        Text("Import from HuggingFace")
                            .font(.system(size: 20, weight: .bold))
                        Text("Download MLX-compatible models directly from HuggingFace")
                            .font(Studio.body13)
                            .foregroundStyle(Color.subtitleGray)
                            .multilineTextAlignment(.center)
                    }

                    RepoInputCard(
                        input: $input,
                        errorMessage: importError,
                        isFocused: $repoFieldFocused,
                        onSubmit: runImport)

                    ImportRiskCard()
                }
                .padding(.horizontal, 18)
                .padding(.bottom, 14)
            }

            Divider()

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button(importing ? "Importing…" : "Import") { runImport() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .tint(Color.brandGreen)
                    .disabled(importing || inputIsEmpty)
            }
            .padding(14)
        }
        .frame(width: 460, height: 560)
        .onAppear {
            repoFieldFocused = true
        }
    }

    private func runImport() {
        let repo = Self.normalizeRepoID(input)
        guard !repo.isEmpty, !importing else { return }
        importing = true
        importError = nil
        Task {
            let error = await chat.modelStore.importModel(repoID: repo)
            importing = false
            importError = error
            if error == nil { dismiss() }
        }
    }

    /// Accepts a bare repo id or a full Hugging Face URL.
    /// Internal so unit tests can pin the URL → owner/repo edge cases.
    static func normalizeRepoID(_ raw: String) -> String {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        for prefix in ["https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"]
        where text.lowercased().hasPrefix(prefix) {
            text = String(text.dropFirst(prefix.count))
            break
        }
        let parts = text.split(separator: "/")
        if parts.count >= 2 { return "\(parts[0])/\(parts[1])" }
        return text
    }
}

// MARK: - Repo input

private struct RepoInputCard: View {
    @Binding var input: String
    let errorMessage: String?
    var isFocused: FocusState<Bool>.Binding
    let onSubmit: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model ID or URL")
                .font(.system(size: 13, weight: .semibold))
            TextField("mlx-community/ModelName-4bit", text: $input)
                .textFieldStyle(.plain)
                .font(.system(size: 13).monospaced())
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: Studio.radiusField, style: .continuous)
                        .fill(Color.fieldFill))
                .focused(isFocused)
                .onSubmit(onSubmit)
                .accessibilityIdentifier("importRepoField")
            Text("Examples: mlx-community/Qwen2-VL-7B-4bit or a full HuggingFace URL")
                .font(.caption)
                .foregroundStyle(.secondary)
            if let errorMessage {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .studioCard()
    }
}

// MARK: - Risk warning

private struct ImportRiskCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("At Your Own Risk", systemImage: "exclamationmark.triangle.fill")
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(Color.iconOrange)
            VStack(alignment: .leading, spacing: 4) {
                bullet("Model compatibility is not guaranteed")
                bullet("May consume significant storage space")
                bullet("Could crash or perform poorly on your device")
            }
            Text("Only import models from trusted sources. Look for models in the **mlx-community** organization for best compatibility.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Link(destination: URL(string: "https://huggingface.co/mlx-community")!) {
                Label("Browse MLX Models on HuggingFace", systemImage: "globe")
                    .font(.caption.weight(.semibold))
            }
            .foregroundStyle(Color.brandGreen)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(Color.iconOrange.opacity(0.08))
        .overlay(RoundedRectangle(cornerRadius: Studio.radiusCard)
            .stroke(Color.iconOrange.opacity(0.35), lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: Studio.radiusCard))
    }

    private func bullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Text("•").foregroundStyle(.secondary)
            Text(text)
                .font(Studio.caption13)
                .foregroundStyle(.secondary)
        }
    }
}
