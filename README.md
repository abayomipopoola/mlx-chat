<p align="center">
  <picture>
    <img alt="MLX Chat" src="docs/screenshots/icon.png" width="140">
  </picture>
</p>

<h1 align="center">MLX Chat</h1>
<p align="center">
    <b>Run local models on Apple Silicon with MLX</b><br>
    A native macOS chat app for running LLMs fully on-device. Local models run on
    Apple Silicon with <a href="https://github.com/ml-explore/mlx-swift-lm">MLX</a>;
    Apple Intelligence is available as a built-in engine on macOS 26+.
</p>
<p align="center">
    <a href="https://mlxchat.dev/">Download</a> ·
    <a href="https://abayomipo.com/mlx-chat-intelligence-in-a-box/">Read the write-up</a> ·
    <a href="#architecture">Architecture</a>
</p>

## Features

<table>
  <tr>
    <td><img src="docs/screenshots/welcome.png" alt="Welcome screen with suggestion cards"></td>
    <td><img src="docs/screenshots/chat.png" alt="Answer with thinking disclosure, LaTeX math, and syntax-highlighted code"></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/picker.png" alt="Model picker: switch between Apple Intelligence and downloaded local models"></td>
    <td><img src="docs/screenshots/models.png" alt="Model manager with downloads and family cards"></td>
  </tr>
</table>

* **Local MLX models**: curated catalog with Qwen, Llama, Phi, Gemma, DeepSeek, and more, plus support for importing any MLX-compatible Hugging Face repo by ID
* **SSD-streamed MoE models**: an opt-in low-memory path for Gemma 4 26B-A4B (standard and QAT) and Qwen3.6 35B-A3B, with automatic resumable setup and configurable context and expert-cache limits
* **Apple Intelligence engine** (FoundationModels): no downloads required, so you can start chatting immediately on first launch
* Streaming responses with a collapsible thinking phase, per-model thinking toggle, markdown, syntax highlighting, LaTeX math, and artifact preview for live-rendered HTML/SVG in a sandboxed web view
* System-prompt presets: General, Creative, Roleplay, Coding, Reasoning, and Custom, each editable with reset-to-default support
* Global temperature control
* Model manager with download progress, cancel, delete, and disk-usage tools
* Storage in `~/Models`; both plain `<org>/<name>` layouts and the Hugging Face hub cache are recognized
* Auto-unload frees RAM after a configurable idle interval, and memory is actually released (see the performance notes below)
* Light, Dark, and System appearance options backed by a token-based theme in `MLXChat/Support/Theme.swift`

## Installing

Download the `.dmg` from the [MLX Chat product page](https://mlxchat.dev/), then drag the app to Applications.

Requires an Apple Silicon Mac running macOS 26 or later.

## Building from source

Requires Xcode 26+, [XcodeGen](https://github.com/yonaskolb/XcodeGen), and Apple Silicon.

`MLXChat.xcodeproj` is generated and gitignored, so always run `xcodegen generate` after cloning the repo or adding files.

```sh
git clone https://github.com/abayomipopoola/mlx-chat.git
cd mlx-chat
xcodegen generate
xcodebuild -project MLXChat.xcodeproj -scheme MLXChat -configuration Debug \
  -skipPackagePluginValidation -skipMacroValidation build
```

The two skip flags are needed for CLI builds because `mlx-swift` ships a build plugin (`CudaBuild`) and macros that otherwise require interactive trust approval in Xcode.

To run tests, use `xcodebuild … test` with the same flags.

## Architecture

The design follows one rule: **one active model behind one protocol.** Standard MLX models, SSD-streamed MoE models, and Apple's on-device foundation model sit behind a single `ChatEngine` interface, so the UI never needs to know which one it is talking to.

<p align="center">
  <img src="docs/architecture.svg" alt="A message flows from ChatController through EngineRuntime and the ChatEngine protocol to either MLXEngine or AppleIntelligenceEngine; ModelStore handles downloads from the Hugging Face Hub" width="560">
</p>

**A turn, top to bottom.** `ChatController` takes your message and asks `EngineRuntime` for an engine. The runtime owns the single in-RAM slot: a small state machine (*empty → downloading → loading → ready → generating*) that downloads lazily, loads lazily, and evicts the current model before admitting a new one. The downloading itself is `ModelStore`'s job: the curated catalog, Hugging Face imports, progress and cancellation, and honest disk bookkeeping in `~/Models`.

**A turn, bottom to top.** Whatever the runtime returns sits behind `ChatEngine`: `load`, `unload`, and a `stream` that yields answer deltas, thinking deltas, and a completion. As tokens arrive, `ChatController` routes "thinking" tokens away from the answer and flushes visible text to the UI at 10 Hz, because re-parsing markdown hundreds of times a second is how chat apps learn to stutter.

### The three engines

* **`MLXEngine`** wraps `mlx-swift-lm`. It loads quantized weights straight into Apple Silicon's unified memory and keeps a KV-cached `ChatSession` alive across turns (see the performance notes below), so follow-up messages skip re-prefilling the whole conversation.
* **`StreamedMoEEngine`** wraps the vendored TurboFieldfare runtime. It keeps shared weights and context state in memory while reading only router-selected experts from a verified `.gturbo` layout on SSD.
* **`AppleIntelligenceEngine`** wraps the FoundationModels framework. It seeds a `LanguageModelSession` with the conversation transcript and streams the response. There are no weights to ship and no memory to manage, so the engine is barely a hundred lines, most of it error handling.

### Modules

* `MLXChat/App`: `ChatController` orchestrates turns, including engine resolution, streaming with a 10 Hz UI flush, retry/regenerate, and model-switch warming
* `MLXChat/Engine`: `EngineRuntime` owns the single active model slot. `MLXEngine`, `StreamedMoEEngine`, and `AppleIntelligenceEngine` sit behind the common `ChatEngine` protocol, alongside prompt presets and the prompt builder
* `MLXChat/Models`: `ModelStore` handles downloads, disk usage, imports, and the model catalog
* `MLXChat/Views`: SwiftUI views, custom window chrome, hidden titlebar, flat resizable sidebar, and route switching without `NavigationStack`
* `MLXChat/Support/Theme.swift`: the design system, including dynamic light/dark color tokens, metrics, and reusable components

### SSD-streamed MoE architecture

The SSD path is separate and opt-in. Manage Models keeps standard MLX as the default and offers SSD streaming only for pinned MoE checkpoints that the bundled TurboFieldfare revision can repack and run. The installer range-downloads, repacks, verifies, and registers the model without first storing a second full source snapshot, and interrupted installs can resume.

At runtime, the model's router still selects every expert required for each token. The expert cache changes how many expert weights stay in RAM; it does not change routing or model output. Users can set context to 4K, 8K, 16K, or 32K and keep 8, 16, 24, or 32 expert slots resident. Dense models and ordinary MLX checkpoints never enter this path. The streamed layouts are text-only because their repack plan omits vision tensors.

See [MOE_ARCH.md](MOE_ARCH.md) for the install, storage, memory, Metal, inference, lifecycle, and trust design. See [SSD-streamed MoE setup](docs/STREAMED_MOE.md) for the user flow.

For the story behind these decisions, read the write-up: [Intelligence in a Box](https://abayomipo.com/mlx-chat-intelligence-in-a-box/).

## Performance and memory practices

The app uses the official current APIs on all three engine paths:

* **MLX**: `mlx-swift-lm` via `LLMModelFactory.loadContainer`, with a KV-cached `ChatSession` reused across turns. This is the most important MLX performance practice because it avoids re-prefilling the full conversation on every message. Sessions rebuild only when needed, such as after a prompt change, conversation switch, regenerate, or cancel. Temperature updates happen in place on live sessions, and history is trimmed to 70% of the model’s context window before prefill.
* **Streamed MoE**: TurboFieldfare memory-maps the shared core, keeps a configurable FP16 KV cache in RAM, and loads routed experts through a bounded cache. Larger expert caches reduce SSD reads without changing expert choice or quality. Dense models continue to use `MLXEngine` unchanged.
* **Memory hygiene**: `Memory.cacheLimit = 20 MB` caps MLX’s Metal buffer cache after load, and `Memory.clearCache()` runs on unload. Without this, unloaded models can leave gigabytes parked in the buffer cache, which is a common source of apparent MLX app leaks. The auto-unload timer, combined with these two calls, ensures RAM is actually released.
* **Apple Intelligence**: uses transcript-seeded `LanguageModelSession`, cumulative-snapshot deduping, friendly handling for context-window and guardrail errors, and temperature clamping to the supported 0–1 range. Sessions are per-turn, so state does not accumulate.
* **General hygiene**: only one model is kept in RAM at a time, streams cancel through `onTermination`, the UI flush loop is throttled to 10 Hz to reduce markdown reparse cost, download tasks clean themselves out of the task dictionary, and every timer/task closure uses `[weak self]`.

## Feedback

Bug reports, feature ideas, and questions are welcome. Please [open an issue](https://github.com/abayomipopoola/mlx-chat/issues).
