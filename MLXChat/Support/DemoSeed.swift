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

    /// A small browser artifact whose visible health check depends on all three
    /// preview layers: HTML nodes, applied CSS grid rules, and executed JavaScript.
    @discardableResult
    static func insertArtifactPreview(using store: ConversationStore) -> UUID {
        let conversation = store.newConversation()
        store.appendMessage(
            role: "user",
            content: "Build a small interactive grid.",
            to: conversation)
        store.appendMessage(
            role: "assistant",
            content: artifactPreviewAnswer,
            to: conversation,
            modelID: "streamed-moe/gemma-4-26b-a4b-it-4bit")
        store.save()
        return conversation.id
    }

    /// Reproduces malformed model output where four-space indentation makes
    /// prose and bare LaTeX look like code. Used only by the Markdown UI test.
    @discardableResult
    static func insertMarkdownRegression(using store: ConversationStore) -> UUID {
        let conversation = store.newConversation()
        store.appendMessage(
            role: "user",
            content: "Explain this telescoping sum.",
            to: conversation)
        store.appendMessage(
            role: "assistant",
            content: malformedMarkdownAnswer,
            to: conversation,
            modelID: "streamed-moe/qwen3.6-35b-a3b-4bit")
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

    private static let malformedMarkdownAnswer = """
        This is a telescoping sum:

            F(n) = \\frac{1}{2} \\left(1 + \\frac{1}{n}\\right)

        Most terms cancel, leaving a finite limit.

            The first term converges to $F(n)=\\frac{3}{4}+\\epsilon_n$.
            The second term converges to $0$ because $\\epsilon_n\\to0$.
        """

    private static let artifactPreviewAnswer = """
        ```html
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>Artifact Preview Regression</title>
          <style>
            body { font-family: sans-serif; margin: 20px; }
            .board { display: grid; grid-template-columns: repeat(3, 44px); gap: 4px; }
            .cell { width: 44px; height: 44px; }
          </style>
        </head>
        <body>
          <p id="preview-health">Checking preview…</p>
          <p id="status">Player X's Turn</p>
          <div class="board" id="board">
            <button class="cell" aria-label="Cell 1"></button>
            <button class="cell" aria-label="Cell 2"></button>
            <button class="cell" aria-label="Cell 3"></button>
            <button class="cell" aria-label="Cell 4"></button>
            <button class="cell" aria-label="Cell 5"></button>
            <button class="cell" aria-label="Cell 6"></button>
            <button class="cell" aria-label="Cell 7"></button>
            <button class="cell" aria-label="Cell 8"></button>
            <button class="cell" aria-label="Cell 9"></button>
          </div>
          <script>
            const board = document.getElementById('board');
            const cells = Array.from(document.querySelectorAll('.cell'));
            const columns = getComputedStyle(board).gridTemplateColumns
              .split(' ').filter(Boolean);
            document.getElementById('preview-health').textContent =
              getComputedStyle(board).display === 'grid' &&
              columns.length === 3 && cells.length === 9
                ? 'HTML CSS JS ready'
                : 'Preview regression';
            cells.forEach(cell => cell.addEventListener('click', () => {
              cell.textContent = 'X';
              document.getElementById('status').textContent = "Player O's Turn";
            }));
          </script>
        </body>
        </html>
        ```
        """
}
