import AppKit
import MarkdownUI
import SwiftUI
import WebKit

// MARK: - Expandable artifact (inline in the transcript, opens downward)

/// Collapsed: a compact chip. Expanded: the content unfolds below the chip in
/// place — no side inspector squeezing the transcript.
struct ExpandableArtifactCard: View {
    let artifact: Artifact

    @Environment(\.colorScheme) private var colorScheme
    @State private var expanded = false
    @State private var tab: Tab = .code
    @State private var copied = false

    enum Tab: String, CaseIterable { case preview = "Preview", code = "Code" }

    private var canPreview: Bool { artifact.kind != .code }

    private var icon: String {
        switch artifact.kind {
        case .code: return "chevron.left.forwardslash.chevron.right"
        case .html: return "safari"
        case .svg: return "photo"
        case .markdown: return "doc.richtext"
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            headerButton
            if expanded {
                Divider()
                controls
                Divider()
                content
                    .frame(height: 340)
            }
        }
        .background(Color.cardFill.opacity(expanded ? 1 : 0))
        .overlay(
            RoundedRectangle(cornerRadius: Studio.radiusCard)
                .stroke(Color.cardStroke, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: Studio.radiusCard))
        .frame(maxWidth: expanded ? .infinity : 420, alignment: .leading)
        .animation(.easeOut(duration: 0.18), value: expanded)
        .onAppear { tab = canPreview ? .preview : .code }
    }

    private var headerButton: some View {
        Button {
            expanded.toggle()
        } label: {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundStyle(Color.brandGreen)
                    .frame(width: 28)
                VStack(alignment: .leading, spacing: 2) {
                    Text(artifact.title)
                        .font(.headline)
                        .lineLimit(1)
                    Text(artifact.isComplete ? "\(artifact.lineCount) lines" : "Writing…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .rotationEffect(.degrees(expanded ? 90 : 0))
            }
            .padding(.horizontal, 14)
            .frame(height: 56)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var controls: some View {
        HStack(spacing: 10) {
            if canPreview {
                Picker("", selection: $tab) {
                    ForEach(Tab.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
            }
            Spacer()
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(artifact.content, forType: .string)
                copied = true
                Task {
                    try? await Task.sleep(for: .seconds(1.5))
                    copied = false
                }
            } label: {
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
            }
            .help("Copy")
            .foregroundStyle(copied ? Color.brandGreen : .secondary)
            Button {
                save()
            } label: {
                Image(systemName: "square.and.arrow.down")
            }
            .help("Save…")
            .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    @ViewBuilder
    private var content: some View {
        if tab == .preview, canPreview {
            if !artifact.isComplete {
                VStack(spacing: 8) {
                    ProgressView()
                    Text("Writing… \(artifact.lineCount) lines")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                preview
            }
        } else {
            codeView
        }
    }

    @ViewBuilder
    private var preview: some View {
        switch artifact.kind {
        case .html:
            WebHTMLView(html: artifact.content)
        case .svg:
            WebHTMLView(html: """
                <!doctype html><html><body style="margin:0;display:grid;place-items:center;min-height:100vh">\(artifact.content)</body></html>
                """)
        case .markdown:
            ScrollView {
                Markdown(MarkdownContent(artifact.content))
                    .markdownTheme(.chat)
                    .markdownCodeSyntaxHighlighter(HighlightrHighlighter.forScheme(colorScheme))
                    .padding(16)
            }
        case .code:
            codeView
        }
    }

    private var codeView: some View {
        ScrollView([.horizontal, .vertical]) {
            HighlightrHighlighter.forScheme(colorScheme)
                .highlightCode(artifact.content, language: artifact.language)
                .font(.system(size: 12, design: .monospaced))
                .textSelection(.enabled)
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }

    private func save() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = slug(artifact.title) + "." + artifact.suggestedFileExtension
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            try? artifact.content.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    private func slug(_ text: String) -> String {
        let mapped = text.lowercased().map { $0.isLetter || $0.isNumber ? $0 : "-" }
        let collapsed = String(mapped).split(separator: "-").joined(separator: "-")
        return collapsed.isEmpty ? "artifact" : collapsed
    }
}

// MARK: - WKWebView wrapper (sandboxed preview; external links open in the browser)

struct WebHTMLView: NSViewRepresentable {
    let html: String

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        guard context.coordinator.lastHTML != html else { return }
        context.coordinator.lastHTML = html
        webView.loadHTMLString(html, baseURL: nil)
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        var lastHTML: String?

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
        ) {
            if navigationAction.navigationType == .linkActivated,
               let url = navigationAction.request.url {
                NSWorkspace.shared.open(url)
                decisionHandler(.cancel)
            } else {
                decisionHandler(.allow)
            }
        }
    }
}
