import AppKit
import SwiftData
import SwiftUI

struct RootView: View {
    @Environment(ChatController.self) private var chat
    @Environment(\.modelContext) private var modelContext
    @State private var selectedConversationID: UUID?
    @State private var navigation = WindowNavigation()
    @State private var showSplash = true
    /// Shared open/anchor state for the root-hosted header dropdowns.
    @State private var dropdown = HeaderDropdown()
    @AppStorage(Keys.appearance) private var appearance = Keys.Defaults.appearance
    /// User-dragged sidebar width; stays put when the window resizes
    /// (unlike HSplitView's proportional redistribution).
    @AppStorage("sidebarWidth") private var sidebarWidth = Double(Studio.sidebarWidth)
    /// Collapsed hides the sidebar entirely; expanding restores the dragged width.
    @AppStorage(Keys.sidebarCollapsed) private var sidebarCollapsed = Keys.Defaults.sidebarCollapsed
    @State private var dragBaseWidth: Double?

    @Query(sort: \Conversation.updatedAt, order: .reverse)
    private var conversations: [Conversation]

    private var selectedConversation: Conversation? {
        conversations.first { $0.id == selectedConversationID }
    }

    var body: some View {
        ZStack {
            mainLayout
            if showSplash {
                SplashView()
                    .transition(.opacity)
            }
        }
        .font(Studio.sans(13))
        .onAppear {
            AppAppearance.apply(appearance)
            if chat.conversationStore == nil {
                chat.conversationStore = ConversationStore(context: modelContext)
            }
            // UI-verification hooks: `--route settings|models` deep-links a pushed
            // page; `--open-chat <n>` selects the nth most recent conversation;
            // `--sidebar-expanded` forces the sidebar open for screenshots.
            // Deferred out of the first layout transaction: mutating @AppStorage
            // and @State while the window is being set up can prevent it from
            // ever being ordered onscreen (combined hooks reproduce this on a
            // fresh-preferences machine).
            Task { @MainActor in
                let arguments = ProcessInfo.processInfo.arguments
                if arguments.contains("--sidebar-expanded") {
                    sidebarCollapsed = false
                } else if arguments.contains("--sidebar-collapsed") {
                    sidebarCollapsed = true
                }
                if let index = arguments.firstIndex(of: "--route"), index + 1 < arguments.count {
                    switch arguments[index + 1] {
                    case "settings": navigation.push(.settings)
                    case "models": navigation.push(.models)
                    default: break
                    }
                }
                if let index = arguments.firstIndex(of: "--open-chat"), index + 1 < arguments.count,
                   let n = Int(arguments[index + 1]), conversations.indices.contains(n) {
                    selectedConversationID = conversations[n].id
                }
                // Deterministic content for screenshots and UI regressions.
                let ssdOrnithID = StreamedMoEInstall.ornith15.modelID
                if arguments.contains("--seed-demo-ssd"),
                   let store = chat.conversationStore {
                    selectedConversationID = DemoSeed.insert(using: store, modelID: ssdOrnithID)
                } else if arguments.contains("--seed-html-prompt-ssd"),
                          let store = chat.conversationStore {
                    selectedConversationID = DemoSeed.insertHTMLPrompt(
                        using: store, modelID: ssdOrnithID)
                } else if arguments.contains("--seed-html-preview-ssd"),
                          let store = chat.conversationStore {
                    selectedConversationID = DemoSeed.insertOrnithHTMLPreview(
                        using: store, modelID: ssdOrnithID)
                } else if arguments.contains("--seed-moe-demo-answer"),
                   let store = chat.conversationStore {
                    selectedConversationID = DemoSeed.insertMoEAnswer(using: store)
                } else if arguments.contains("--seed-moe-demo-prompt"),
                          let store = chat.conversationStore {
                    selectedConversationID = DemoSeed.insertMoEPrompt(using: store)
                } else                 if arguments.contains("--seed-q1-logic"),
                          let store = chat.conversationStore {
                    selectedConversationID = DemoSeed.insertQ1Logic(using: store)
                } else if arguments.contains("--seed-markdown-regression"),
                          let store = chat.conversationStore {
                    selectedConversationID = DemoSeed.insertMarkdownRegression(using: store)
                } else if arguments.contains("--seed-ornith-html-preview"),
                          let store = chat.conversationStore {
                    selectedConversationID = DemoSeed.insertOrnithHTMLPreview(using: store)
                } else if arguments.contains("--seed-artifact-preview"),
                          let store = chat.conversationStore {
                    selectedConversationID = DemoSeed.insertArtifactPreview(using: store)
                } else if arguments.contains("--seed-demo"),
                          let store = chat.conversationStore {
                    selectedConversationID = DemoSeed.insert(using: store)
                }
                if arguments.contains("--capture-light") {
                    try? await Task.sleep(for: .milliseconds(800))
                    appearance = "light"
                    AppAppearance.apply(appearance)
                } else if arguments.contains("--capture-dark") {
                    try? await Task.sleep(for: .milliseconds(800))
                    appearance = "dark"
                    AppAppearance.apply(appearance)
                }
            }
        }
        .task {
            chat.preloadSelectedModelIfAvailable()
        }
        .task {
            try? await Task.sleep(for: .seconds(1.0))
            withAnimation(.easeOut(duration: 0.4)) { showSplash = false }
        }
        .onReceive(NotificationCenter.default.publisher(for: .settingsRoute)) { _ in
            navigation.push(.settings)
        }
        .onReceive(NotificationCenter.default.publisher(for: .toggleSidebar)) { _ in
            sidebarCollapsed.toggle()
        }
        .onReceive(NotificationCenter.default.publisher(for: .newChat)) { _ in
            // Same action as SidebarView's New chat button: clear selection so
            // ChatScreen shows the welcome state (conversation is created lazily
            // on first send). Idempotent while already on welcome.
            selectedConversationID = nil
            navigation.goHome()
        }
        .onChange(of: appearance) {
            AppAppearance.apply(appearance)
        }
        // Close any open header dropdown on context changes the scrim can't
        // catch: navigation (glass buttons receive clicks above the scrim),
        // sidebar collapse, and Esc (posted by the app's key monitor).
        .onChange(of: navigation.path) {
            dropdown.open = nil
            if !navigation.path.isEmpty {
                DispatchQueue.main.async { NSApp.keyWindow?.makeFirstResponder(nil) }
            }
        }
        .onChange(of: sidebarCollapsed) { dropdown.open = nil }
        .onReceive(NotificationCenter.default.publisher(for: .dismissHeaderDropdown)) { _ in
            dropdown.open = nil
        }
    }

    /// Flat two-pane layout (MLX Studio chrome): resizable sidebar, hairline seam,
    /// custom route switching in the detail pane — no NavigationStack, so the
    /// hidden-titlebar window keeps its traffic lights and gains no toolbar inset.
    private var mainLayout: some View {
        HStack(spacing: 0) {
            if !sidebarCollapsed {
                SidebarView(
                    selectedConversationID: $selectedConversationID,
                    onOpenSettings: { navigation.push(.settings) },
                    onNavigateHome: { navigation.goHome() })
                .frame(width: sidebarWidth)

                splitHandle
            }

            detailPane
                .frame(minWidth: 400, maxWidth: .infinity)
                .background(Color.detailBackground)
                .overlay(alignment: .bottomTrailing) {
                    DownloadOverlayHost(isOnModels: navigation.isOnModels)
                }
        }
        .animation(.easeOut(duration: 0.22), value: sidebarCollapsed)
        .overlay(alignment: .topLeading) {
            TitlebarWindowCluster(navigation: navigation)
        }
        .ignoresSafeArea()
        .coordinateSpace(name: windowRootCoordinateSpace)
        // Dropdown dismissal scrim: covers the whole window (sidebar included)
        // while a header dropdown is open; first click outside closes it.
        .overlay {
            if dropdown.open != nil {
                Color.black.opacity(0.001)
                    .contentShape(Rectangle())
                    .onTapGesture { dropdown.open = nil }
                    .transition(.opacity)
            }
        }
        // Root-hosted dropdown panel, anchored below its header button. Hosted
        // here (not per-button) because the header's safeAreaInset bar clips
        // its content to the bar region.
        .overlay(alignment: .topLeading) {
            if let kind = dropdown.open, let anchor = dropdown.anchors[kind] {
                dropdownPanel(kind, anchor: anchor)
                    .transition(.opacity)
            }
        }
        .animation(.easeOut(duration: 0.15), value: dropdown.open)
        // Outermost so the overlay layers above inherit it too: an overlay
        // inherits the environment from its wrapper, not from an injection
        // applied earlier in the chain (the panel crashed on a missing
        // HeaderDropdown when it was injected before the overlays).
        .environment(dropdown)
        // Prompt editor, hosted here so closing the dropdown (which removes
        // the panel) cannot tear the sheet down.
        .sheet(item: $dropdown.editingPrompt) { preset in
            EditPromptSheet(preset: preset)
        }
    }

    private var detailPane: some View {
        ZStack {
            ChatScreen(
                conversation: selectedConversation,
                onCreateConversation: {
                    let conversation = chat.conversationStore?.newConversation()
                    selectedConversationID = conversation?.id
                    return conversation
                })
            .allowsHitTesting(navigation.path.isEmpty)
            .accessibilityHidden(!navigation.path.isEmpty)

            if navigation.current == .settings {
                SettingsPage { navigation.push(.models) }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color.detailBackground)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            } else if navigation.current != nil {
                ModelsPage(
                    familyID: navigation.familyID,
                    onOpenFamily: { navigation.push(.modelFamily($0)) })
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color.detailBackground)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
        .animation(.easeOut(duration: 0.22), value: navigation.path)
    }

    @ViewBuilder
    private func dropdownPanel(_ kind: HeaderDropdown.Kind, anchor: CGRect) -> some View {
        switch kind {
        case .modelPicker:
            ModelPickerPanel(onManageModels: { navigation.push(.models) })
                .dropdownChrome(width: 185)
                .offset(x: max(8, anchor.maxX - 185), y: anchor.maxY + 6)
        case .promptPresets:
            PromptPresetPanel()
                .dropdownChrome(width: 175)
                .offset(x: max(8, anchor.maxX - 175), y: anchor.maxY + 6)
        case .attach:
            Color.clear
                .frame(width: 200, height: 1)
                .overlay(alignment: .bottom) {
                    AttachPanel()
                        .dropdownChrome(width: 200)
                }
                .offset(x: anchor.minX, y: anchor.minY - 8)
        }
    }

    /// Hairline seam with an invisible 9pt grab zone for dragging the sidebar width.
    private var splitHandle: some View {
        Rectangle()
            .fill(Color.cardStroke)
            .frame(width: 1)
            .overlay {
                Color.clear
                    .frame(width: 9)
                    .contentShape(Rectangle())
                    .onHover { inside in
                        if inside { NSCursor.resizeLeftRight.push() } else { NSCursor.pop() }
                    }
                    .gesture(
                        DragGesture(minimumDistance: 1, coordinateSpace: .global)
                            .onChanged { value in
                                let base = dragBaseWidth ?? sidebarWidth
                                dragBaseWidth = base
                                sidebarWidth = min(max(base + value.translation.width, 200), 420)
                            }
                            .onEnded { _ in dragBaseWidth = nil }
                    )
            }
    }
}

/// Isolates `downloadProgress` so RootView chrome does not observe the
/// full dictionary on every fraction tick.
private struct DownloadOverlayHost: View {
    @Environment(ChatController.self) private var chat
    let isOnModels: Bool

    var body: some View {
        let isEmpty = chat.modelStore.downloadProgress.isEmpty
        Group {
            if !isEmpty, !isOnModels {
                DownloadQueueView()
                    .frame(width: 320)
                    .shadow(color: .black.opacity(0.15), radius: 12, y: 4)
                    .padding(.trailing, 20)
                    .padding(.bottom, 96)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.easeOut(duration: 0.25), value: isEmpty)
    }
}
