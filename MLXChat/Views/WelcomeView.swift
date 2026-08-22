import SwiftUI

/// Empty-state hero with suggestion chips, matching the MLX Studio arrangement.
struct WelcomeView: View {
    var isEnabled: Bool = true
    let onTemplate: (String) -> Void

    private struct Chip: Identifiable {
        let title: String
        let subtitle: String
        let template: String
        var id: String { title }
    }

    /// Stable identities: a new UUID per `WelcomeView` init made Liquid Glass
    /// leave a ghost copy when the sidebar collapsed and the view redrew.
    private static let chips: [Chip] = [
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

            Text("What can I help with?")
                .font(Studio.titleXL)
                .multilineTextAlignment(.center)

            Grid(horizontalSpacing: 12, verticalSpacing: 12) {
                GridRow {
                    chipButton(Self.chips[0])
                    chipButton(Self.chips[1])
                }
                GridRow {
                    chipButton(Self.chips[2])
                    chipButton(Self.chips[3])
                }
            }
            .geometryGroup()
            .transaction { $0.animation = nil }
            .padding(.top, 36)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func chipButton(_ chip: Chip) -> some View {
        ChipButton(chip: chip, isEnabled: isEnabled) {
            onTemplate(chip.template)
        }
    }

    private struct ChipButton: View {
        let chip: Chip
        var isEnabled: Bool = true
        let action: () -> Void

        var body: some View {
            Button(action: action) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(chip.title)
                        .font(Studio.sans(15, weight: .bold))
                    Text(chip.subtitle)
                        .font(Studio.sans(13))
                        .foregroundStyle(Color.subtitleGray)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .frame(width: 250, alignment: .leading)
                .studioQuietFill(in: RoundedRectangle(cornerRadius: Studio.radiusSuggestion, style: .continuous))
                .contentShape(RoundedRectangle(cornerRadius: Studio.radiusSuggestion, style: .continuous))
            }
            .buttonStyle(.plain)
            .disabled(!isEnabled)
            .accessibilityIdentifier("welcomeChip.\(chip.id)")
        }
    }
}
