import Foundation

/// Inserts a showcase conversation for documentation screenshots
/// (`--seed-demo` launch argument). Each run inserts a fresh copy;
/// delete it like any chat via the sidebar context menu.
@MainActor
enum DemoSeed {
    /// The deterministic streamed-MoE prompt used by the release demo.
    static let streamedMoEQuestion = moeQuestion

    /// The deterministic streamed-MoE reasoning text used by the release demo.
    static let streamedMoEThinking = moeThinking

    /// The deterministic streamed-MoE answer used by the release demo.
    static let streamedMoEAnswer = moeAnswer

    /// Returns the id of the inserted demo conversation.
    @discardableResult
    static func insert(
        using store: ConversationStore,
        modelID: String = "mlx-community/Qwen3.5-9B-4bit"
    ) -> UUID {
        let conversation = store.newConversation()
        store.appendMessage(
            role: "user",
            content: "Explain how gradient descent works, with the key math.",
            to: conversation)

        let assistant = store.appendMessage(
            role: "assistant",
            content: answer,
            to: conversation,
            modelID: modelID)
        assistant.thinking = thinking
        assistant.thinkingSeconds = 4.8
        assistant.tokensPerSecond = 41.6
        store.save()
        return conversation.id
    }

    /// Release 1.2 prompt-only state for native video captures.
    @discardableResult
    static func insertMoEPrompt(using store: ConversationStore) -> UUID {
        let conversation = store.newConversation()
        store.appendMessage(
            role: "user",
            content: moeQuestion,
            to: conversation)
        store.save()
        return conversation.id
    }

    /// Release 1.2 answer state: the SSD cache trade-off, with thinking and
    /// the selected streamed checkpoint represented exactly as the app does.
    @discardableResult
    static func insertMoEAnswer(using store: ConversationStore) -> UUID {
        let conversation = store.newConversation()
        store.appendMessage(
            role: "user",
            content: moeQuestion,
            to: conversation)
        let assistant = store.appendMessage(
            role: "assistant",
            content: moeAnswer,
            to: conversation,
            modelID: "streamed-moe/gemma-4-26b-a4b-it-qat-4bit")
        assistant.thinking = moeThinking
        assistant.thinkingSeconds = 7.8
        assistant.tokensPerSecond = 18.4
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

    /// Release-demo HTML preview, matching the in-chat artifact card used in the
    /// chrome walkthrough. `modelID` lets a capture attribute it to whichever
    /// model the walkthrough selected.
    @discardableResult
    static func insertOrnithHTMLPreview(
        using store: ConversationStore,
        modelID: String = "ornith-ai/Ornith-1.5-9B-MLX-4bit"
    ) -> UUID {
        let conversation = store.newConversation()
        store.appendMessage(
            role: "user",
            content: "build a tic tac toe multi player",
            to: conversation)
        store.appendMessage(
            role: "assistant",
            content: ornithHTMLPreviewAnswer,
            to: conversation,
            modelID: modelID)
        store.save()
        return conversation.id
    }

    /// The same artifact question with no answer yet, so a capture can show the
    /// prompt sitting on its own before the reply arrives.
    @discardableResult
    static func insertHTMLPrompt(
        using store: ConversationStore,
        modelID: String = "ornith-ai/Ornith-1.5-9B-MLX-4bit"
    ) -> UUID {
        let conversation = store.newConversation()
        store.appendMessage(
            role: "user",
            content: "build a tic tac toe multi player",
            to: conversation)
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

    /// Q1 knight/knave operators that Mona VF cannot draw as text.
    @discardableResult
    static func insertQ1Logic(using store: ConversationStore) -> UUID {
        let conversation = store.newConversation()
        store.appendMessage(
            role: "user",
            content: q1LogicQuestion,
            to: conversation)
        store.appendMessage(
            role: "assistant",
            content: q1LogicAnswer,
            to: conversation,
            modelID: "ornith-ai/Ornith-1.5-35B-A3B")
        store.save()
        return conversation.id
    }

    private static let thinking = """
        The user wants an intuitive explanation plus the core math. I should \
        start with the hiking analogy, give the update rule, explain the \
        learning rate trade-off, and keep it compact with a small code sketch.
        """

    private static let moeQuestion =
        "Does a larger expert cache make the SSD model smarter?"

    private static let moeThinking = """
        The cache affects where expert weights wait between tokens, not which
        experts the router selects. I should separate the context setting from
        the expert-cache setting and make the quality trade-off explicit.
        """

    private static let moeAnswer = """
        **No.** A larger expert cache changes RAM use and SSD traffic — not the model's intelligence.

        - **Expert cache:** keeps 8, 16, 24, or 32 routed-expert slots per layer in RAM. More slots can reduce SSD reads.
        - **Routing:** Gemma still selects and runs the same top eight experts for every token, with the same weights.
        - **Context:** 4K, 8K, 16K, or 32K controls retained history and FP16 KV-cache memory independently.

        If you have RAM to spare, use more expert slots for speed. Model quality stays the same.
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

    private static let ornithHTMLPreviewAnswer = """
        Here's a complete multi-player tic-tac-toe game in one HTML file.

        ```html
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>Tic Tac Toe</title>
          <style>
            html,body{margin:0;height:100%;background:#14182a;color:#eef1f8;font-family:-apple-system,sans-serif}
            body{display:grid;place-items:center}
            h1{margin:0 0 18px;font-size:28px;letter-spacing:.2px}
            .board{display:grid;grid-template-columns:repeat(3,92px);gap:10px}
            .cell{width:92px;height:92px;border:0;border-radius:16px;background:#1c2340;font-size:42px;font-weight:700}
            .x{color:#5aa7ff}.o{color:#ff6b8a}
          </style>
        </head>
        <body>
          <div>
            <h1 id="status">Player X wins! 🎉</h1>
            <div class="board">
              <button class="cell x">X</button><button class="cell"></button><button class="cell o">O</button>
              <button class="cell o">O</button><button class="cell o">O</button><button class="cell"></button>
              <button class="cell x">X</button><button class="cell x">X</button><button class="cell x">X</button>
            </div>
          </div>
        </body>
        </html>
        ```

        ## How it works

        Save as `tictactoe.html`. X and O take turns; a full board with no winner is a draw.
        """

    private static let q1LogicQuestion = """
        Q1 — Logic (self-reference with a genuinely undetermined element) On an \
        island, everyone is a knight or a knave. P, Q, R, S. Determine every \
        consistent assignment.
        """

    private static let q1LogicAnswer = """
        # Q1 — Logic

        Let each islander be True (knight) or False (knave).

        - **P:** "Q and R are the same type" → P ⇔ (Q = R), i.e., P ⇔ (Q ⇔ R)
        - **Q:** "P is a knave OR S is a knight" → Q ⇔ (¬P ∨ S)
        - **R:** "An odd number of the four are knaves" → R ⇔ (k is odd)
        - **S:** "R is a knave" → S ⇔ ¬R

        **Key deduction from S:** S ⇔ ¬R means S and R have opposite types.

        **Q's statement:** Q ⇔ (¬P ∨ S). With S = T, (¬P ∨ T) is always True.
        """
}
