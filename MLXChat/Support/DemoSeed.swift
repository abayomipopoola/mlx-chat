import Foundation

/// Inserts a showcase conversation for documentation screenshots
/// (`--seed-demo` launch argument). Each run inserts a fresh copy;
/// delete it like any chat via the sidebar context menu.
@MainActor
enum DemoSeed {
    /// Returns the id of the inserted demo conversation.
    @discardableResult
    static func insert(using store: ConversationStore) -> UUID {
        let conversation = store.newConversation()
        store.appendMessage(
            role: "user",
            content: "Explain how gradient descent works, with the key math.",
            to: conversation)

        let assistant = store.appendMessage(
            role: "assistant",
            content: answer,
            to: conversation,
            modelID: "mlx-community/Qwen3.5-9B-4bit")
        assistant.thinking = thinking
        assistant.thinkingSeconds = 4.8
        assistant.tokensPerSecond = 41.6
        store.save()
        return conversation.id
    }

    private static let thinking = """
        The user wants an intuitive explanation plus the core math. I should \
        start with the hiking analogy, give the update rule, explain the \
        learning rate trade-off, and keep it compact with a small code sketch.
        """

    private static let answer = """
        Gradient descent is how a model *learns*: feel the slope under your feet on a foggy hillside, step downhill, reassess. Each step updates the parameters $\\theta$ against the gradient of the loss $L$:

        ```math
        \\theta_{t+1} = \\theta_t - \\eta \\, \\nabla L(\\theta_t)
        ```

        - $\\nabla L$ points **uphill** — so we subtract it.
        - $\\eta$ is the **learning rate**: too small and training crawls, too large and you overshoot the valley.

        ## In code

        ```swift
        var theta = initialGuess
        for _ in 0..<steps {
            theta -= learningRate * lossGradient(at: theta)
        }
        ```

        So the whole update rule in one line: \\boxed{\\theta_{t+1} = \\theta_t - \\eta \\, \\nabla L(\\theta_t)}
        """
}
