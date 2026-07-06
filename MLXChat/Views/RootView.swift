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
    @AppStorage(Keys.appearance) private var appearance = Keys.Defaults.appearance
    /// User-dragged sidebar width; stays put when the window resizes
    /// (unlike HSplitView's proportional redistribution).
    @AppStorage("sidebarWidth") private var sidebarWidth = Double(Studio.sidebarWidth)
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
            // page; `--open-chat <n>` selects the nth most recent conversation.
            let arguments = ProcessInfo.processInfo.arguments
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
        .onChange(of: appearance) {
            AppAppearance.apply(appearance)
        }
    }

    /// Flat two-pane layout (MLX Studio chrome): resizable sidebar, hairline seam,
    /// custom route switching in the detail pane — no NavigationStack, so the
    /// hidden-titlebar window keeps its traffic lights and gains no toolbar inset.
    private var mainLayout: some View {
        HStack(spacing: 0) {
            SidebarView(
                selectedConversationID: $selectedConversationID,
                onOpenSettings: { push(.settings) },
                onNavigateHome: { path.removeAll() })
            .frame(width: sidebarWidth)

            splitHandle

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
        .ignoresSafeArea()
    }

    private var detailPane: some View {
        ZStack {
            ChatScreen(
                conversation: selectedConversation,
                onCreateConversation: {
                    let conversation = chat.conversationStore?.newConversation()
                    selectedConversationID = conversation?.id
                    return conversation
                },
                onManageModels: {
                    push(.models)
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
