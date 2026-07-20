import SwiftData
import SwiftUI

/// In-frame navigation targets shown in the detail column.
enum AppRoute: Hashable {
    case settings
    case models
}

struct RootView: View {
    @Environment(ChatController.self) private var chat
    @Environment(\.modelContext) private var modelContext
    @State private var selectedConversationID: UUID?
    @State private var path: [AppRoute] = []
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
                }
                if let index = arguments.firstIndex(of: "--route"), index + 1 < arguments.count {
                    switch arguments[index + 1] {
                    case "settings": push(.settings)
                    case "models": push(.models)
                    default: break
                    }
                }
                if let index = arguments.firstIndex(of: "--open-chat"), index + 1 < arguments.count,
                   let n = Int(arguments[index + 1]), conversations.indices.contains(n) {
                    selectedConversationID = conversations[n].id
                }
                // `--seed-demo` inserts a showcase conversation (markdown, LaTeX,
                // code, thinking) for documentation screenshots, then selects it.
                if arguments.contains("--seed-demo"), let store = chat.conversationStore {
                    selectedConversationID = DemoSeed.insert(using: store)
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
            push(.settings)
        }
        .onReceive(NotificationCenter.default.publisher(for: .toggleSidebar)) { _ in
            sidebarCollapsed.toggle()
        }
        .onReceive(NotificationCenter.default.publisher(for: .newChat)) { _ in
            // Same action as SidebarView's New chat button: clear selection so
            // ChatScreen shows the welcome state (conversation is created lazily
            // on first send). Idempotent while already on welcome.
            selectedConversationID = nil
            path.removeAll()
        }
        .onChange(of: appearance) {
            AppAppearance.apply(appearance)
        }
        // Close any open header dropdown on context changes the scrim can't
        // catch: navigation (glass buttons receive clicks above the scrim),
        // sidebar collapse, and Esc (posted by the app's key monitor).
        .onChange(of: path) { dropdown.open = nil }
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
                    onOpenSettings: { push(.settings) },
                    onNavigateHome: { path.removeAll() })
                .frame(width: sidebarWidth)

                splitHandle
            }

            detailPane
                .frame(minWidth: 400, maxWidth: .infinity)
                .background(Color.detailBackground)
                .overlay(alignment: .bottomTrailing) {
                    if !chat.modelStore.downloadProgress.isEmpty, path.last != .models {
                        DownloadQueueView()
                            .frame(width: 320)
                            .shadow(color: .black.opacity(0.15), radius: 12, y: 4)
                            .padding(.trailing, 20)
                            .padding(.bottom, 96)
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                    }
                }
                .animation(.easeOut(duration: 0.25), value: chat.modelStore.downloadProgress.isEmpty)
        }
        .animation(.easeOut(duration: 0.22), value: sidebarCollapsed)
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
                let width: CGFloat = kind == .modelPicker ? 185 : 175
                Group {
                    switch kind {
                    case .modelPicker:
                        ModelPickerPanel(onManageModels: { push(.models) })
                    case .promptPresets:
                        PromptPresetPanel()
                    }
                }
                .dropdownChrome(width: width)
                .offset(x: max(8, anchor.maxX - width), y: anchor.maxY + 6)
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
            .opacity(path.isEmpty ? 1 : 0)
            .allowsHitTesting(path.isEmpty)

            if let route = path.last {
                Group {
                    switch route {
                    case .settings:
                        SettingsPage(onBack: pop) { push(.models) }
                    case .models:
                        ModelsPage(onBack: pop)
                    }
                }
                .background(Color.detailBackground)
                .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
        .animation(.easeOut(duration: 0.22), value: path)
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

    private func push(_ route: AppRoute) {
        guard path.last != route else { return }
        path.append(route)
    }

    private func pop() {
        guard !path.isEmpty else { return }
        path.removeLast()
    }
}
