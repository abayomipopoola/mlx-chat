# SSD-streamed MoE setup

MLX Chat can run both instruction-tuned Gemma 4 26B-A4B checkpoints, Qwen3.6
35B-A3B, Ornith 1.5 35B-A3B, and LFM2.5 24B-A2B with
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare). This path applies
only to those five pinned MoE entries. Dense and
ordinary MLX models still use `MLXEngine` without any change.

## What stays in memory

The backend keeps TurboFieldfare's common model core, an FP16 KV cache, and a
configurable expert cache in memory. Gemma's 128, Qwen's 256, or LFM2.5's 64
routed experts per layer stay in the `.gturbo` install. For each token, the
router selects eight experts (four for LFM2.5) and the runtime reads cache
misses from SSD. Gemma installs use about 14.3 GB, LFM2.5 about 13.6 GB, and
Qwen and Ornith about 19.6 GB because their expert files are larger in total.

The SSD model settings expose two independent controls:

- **Context:** 4K, 8K, 16K, or 32K. A larger context raises FP16 KV-cache RAM.
- **Expert cache:** 8, 16, 24, or 32 slots. More slots keep more routed experts
  in RAM and can reduce SSD reads.

The defaults remain 4K context and 16 expert slots. Changing either value
reloads only the SSD-streamed engine. macOS may also keep recent SSD reads in
its file cache, so process and system memory figures can vary.

## Download and use the model

1. Open **Settings**, then **Manage Models**.
2. Open **Gemma 4** and choose either 26B-A4B checkpoint, open **Qwen3.6**
   and choose **Qwen3.6 35B A3B**, open **Ornith** and choose
   **Ornith 1.5 35B A3B**, or open **LFM2.5** and choose **LFM2.5 24B A2B**.
3. Leave **Standard MLX** selected for the normal in-memory model, or opt into
   **SSD-Streamed** for the low-memory backend.
4. Set **Context** and **Expert cache** for the RAM and SSD-I/O balance you want.
5. Choose **Download SSD Model**.

MLX is always selected by default. The SSD option appears only for these five
supported MoE checkpoints; dense models and other MoE models stay on MLX.

MLX Chat bundles TurboFieldfare's pinned installer. It downloads, repacks,
verifies, registers, and selects the `.gturbo` model without opening Terminal
or asking the user to run a command. A Gemma SSD download then fetches the
vision companion (`Name.vision.gturbo`) as a second phase, range-streaming
only the vision tensors. An existing text-only Gemma install can add that
companion later without downloading the text pack again. Progress appears in
the sheet and the Manage Models download queue. Cancelling keeps a verified
checkpoint, so the next download resumes automatically.

The standard Gemma install lands at `~/Models/Gemma4-26B-A4B.gturbo`, QAT at
`~/Models/Gemma4-26B-A4B-QAT.gturbo`, Qwen at
`~/Models/Qwen3.6-35B-A3B.gturbo`, Ornith at
`~/Models/Ornith-1.5-35B-A3B.gturbo`, and LFM2.5 at
`~/Models/LFM2.5-24B-A2B.gturbo`. An MLX
`.safetensors` copy and its `.gturbo` install are separate disk layouts, so
keeping both uses about twice the checkpoint size. MLX Chat can also adopt an
existing verified `.gturbo` folder through **Use Existing .gturbo Folder**.
Removing an MLX Chat-managed SSD install moves its folder and resumable
sidecars to macOS Trash; empty Trash to reclaim the space. Removing a folder
registered through **Use Existing .gturbo Folder** only forgets the bookmark
and never changes the external files.

The supported sources are pinned separately:

- `mlx-community/gemma-4-26b-a4b-it-4bit` at revision
  `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`
- `mlx-community/gemma-4-26B-A4B-it-qat-4bit` at revision
  `0e3cbab38ce568cf6e23543010d08d03b731910c`
- `mlx-community/Qwen3.6-35B-A3B-4bit` at revision
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`
- `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` at revision
  `19504d912fa8fc7622bf6b1de3db5d5d890b1f02`
- `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` at revision
  `667408d494bde191c2cb9dd4f4549827816b9275`

Each install has its own app ID, receipt, resumable checkpoint, and folder. You
can keep or remove any one without changing the others or its normal MLX
download.

The chat picker labels the streamed install with `SSD ·` and an ordinary copy
of the source checkpoint with `MLX ·`. Both may remain installed and selectable.

## Current scope

- Apple Silicon with macOS 26 and Metal 4
- Text chat, system prompts, multi-turn history, and family-specific Gemma,
  Qwen, or LFM thinking output
- Configurable 4K–32K in-memory context; generation uses the space left after
  the prompt
- Configurable 8–32-slot expert cache
- One resident local model at a time, with the normal idle unload timer

Gemma SSD installs accept images only when a vision companion pack sits beside
the text `.gturbo` (`Name.vision.gturbo`) and the Mac is M2 or newer. A fresh
Gemma SSD download writes that companion after the text pack. An already
installed text-only Gemma folder can download the companion alone. Without a
pack the attachment control stays hidden. The ordinary MLX Gemma and Qwen
checkpoints keep their existing image support. Qwen3.6, Ornith 1.5, and
LFM2.5 SSD identities stay text-only.

LFM2.5 always thinks before answering; its template has no switch to turn
thinking off. Its SSD runner processes the prompt one token at a time, so a
long first prompt takes longer to start than on the other SSD models; later
turns reuse the cached conversation.

TurboFieldfare's upstream real-model acceptance run covers Qwen at 4K context
and 16 expert slots. MLX Chat exposes the same 4K–32K and 8–32 controls for both
families; larger Qwen contexts use more RAM and remain less tested than 4K.
