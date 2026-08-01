# SSD-streamed MoE setup

MLX Chat can run both instruction-tuned Gemma 4 26B-A4B checkpoints and Qwen3.6
35B-A3B with [TurboFieldfare](https://github.com/drumih/turbo-fieldfare). This
path is text-only and applies only to those three pinned MoE entries. Dense and
ordinary MLX models still use `MLXEngine` without any change.

## What stays in memory

The backend keeps TurboFieldfare's common model core, an FP16 KV cache, and a
configurable expert cache in memory. Gemma's 128 or Qwen's 256 routed experts
per layer stay in the `.gturbo` install. For each token, the router selects
eight experts and the runtime reads cache misses from SSD. Gemma installs use
about 14.3 GB; Qwen uses about 19.6 GB because its expert files are larger in
total.

The SSD model settings expose two independent controls:

- **Context:** 4K, 8K, 16K, or 32K. A larger context raises FP16 KV-cache RAM.
- **Expert cache:** 8, 16, 24, or 32 slots. More slots keep more routed experts
  in RAM and can reduce SSD reads.

The defaults remain 4K context and 16 expert slots. Changing either value
reloads only the SSD-streamed engine. macOS may also keep recent SSD reads in
its file cache, so process and system memory figures can vary.

## Download and use the model

1. Open **Settings**, then **Manage Models**.
2. Open **Gemma 4** and choose either 26B-A4B checkpoint, or open **Qwen3.6**
   and choose **Qwen3.6 35B A3B**.
3. Leave **Standard MLX** selected for the normal in-memory model, or opt into
   **SSD-Streamed** for the low-memory backend.
4. Set **Context** and **Expert cache** for the RAM and SSD-I/O balance you want.
5. Choose **Download SSD Model**.

MLX is always selected by default. The SSD option appears only for these three
supported MoE checkpoints; dense models and other MoE models stay on MLX.

MLX Chat bundles TurboFieldfare's pinned installer. It downloads, repacks,
verifies, registers, and selects the `.gturbo` model without opening Terminal
or asking the user to run a command. Progress appears in the sheet and the
Manage Models download queue. Cancelling keeps a verified checkpoint, so the
next download resumes automatically.

The standard Gemma install lands at `~/Models/Gemma4-26B-A4B.gturbo`, QAT at
`~/Models/Gemma4-26B-A4B-QAT.gturbo`, and Qwen at
`~/Models/Qwen3.6-35B-A3B.gturbo`. An MLX
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

Each install has its own app ID, receipt, resumable checkpoint, and folder. You
can keep or remove any one without changing the others or its normal MLX
download.

The chat picker labels the streamed install with `SSD ·` and an ordinary copy
of the source checkpoint with `MLX ·`. Both may remain installed and selectable.

## Current scope

- Apple Silicon with macOS 26 and Metal 4
- Text chat, system prompts, multi-turn history, and family-specific Gemma or
  Qwen thinking output
- Configurable 4K–32K in-memory context; generation uses the space left after
  the prompt
- Configurable 8–32-slot expert cache
- One resident local model at a time, with the normal idle unload timer

Image input is not enabled on this backend because TurboFieldfare's current
`.gturbo` install omits the vision weights. The ordinary MLX Qwen3.6 checkpoint
keeps its image support.

TurboFieldfare's upstream real-model acceptance run covers Qwen at 4K context
and 16 expert slots. MLX Chat exposes the same 4K–32K and 8–32 controls for both
families; larger Qwen contexts use more RAM and remain less tested than 4K.
