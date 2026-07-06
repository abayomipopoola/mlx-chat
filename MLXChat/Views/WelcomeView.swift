import SwiftUI

/// Empty-state hero with suggestion chips, matching the MLX Studio arrangement.
struct WelcomeView: View {
    let onTemplate: (String) -> Void

    private struct Chip: Identifiable {
        let id = UUID()
        let title: String
        let subtitle: String
        let template: String
    }

    private let chips: [Chip] = [
        .init(title: "Summarise", subtitle: "this text",
              template: "Could you please provide a concise summary of this text, highlighting the main points and key takeaways? Focus on the most important information while maintaining the core message."),
        .init(title: "Organise", subtitle: "my finances",
              template: "I need help creating a monthly budget and developing better spending habits. Please provide practical tips for saving money, tracking expenses, and planning for future financial goals."),
        .init(title: "Explain", subtitle: "a complex topic simply",
              template: "Could you explain how quantum computing works in simple terms? Please use everyday analogies and avoid technical jargon."),
        .init(title: "Start learning", subtitle: "French",
              template: "I want to start learning French. Could you create a beginner-friendly study plan with daily exercises, recommended resources, and practical conversation starters?"),
    ]

    var body: some View {
        VStack(spacing: 0) {
            Spacer()

            Image("SplashLogo")
                .resizable()
                .scaledToFit()
                .frame(width: 110, height: 110)
                .frame(height: 160)

            Text("What can I help with?")
                .font(Studio.titleXL)

            LazyVGrid(
                columns: [GridItem(.fixed(250), spacing: 12), GridItem(.fixed(250))],
                spacing: 12
            ) {
                ForEach(chips) { chip in
                    ChipButton(chip: chip) {
                        onTemplate(chip.template)
                    }
                }
            }
            .padding(.top, 44)

            Spacer()
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private struct ChipButton: View {
        let chip: Chip
        let action: () -> Void
        @State private var hovering = false

        var body: some View {
            Button(action: action) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(chip.title)
                        .font(.system(size: 15, weight: .bold))
                    Text(chip.subtitle)
                        .font(.system(size: 13))
                        .foregroundStyle(Color.subtitleGray)
                }
                .padding(.horizontal, 18)
                .padding(.vertical, 14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(hovering ? Color.fieldFill : Color.bubbleFill))
                .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            }
            .buttonStyle(.plain)
            .onHover { hovering = $0 }
        }
    }
}
