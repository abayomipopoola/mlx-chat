# MLX Chat patch

This source starts from TurboFieldfare main commit
`76f6611467207a80c4a4d72bd11c2818f0703739` and includes Qwen 3.6 support from
pull-request head `bddb8e5fd55d96e821007bc152412e384a14abba`.

MLX Chat keeps the Qwen and Gemma runtime contracts from that upstream work.
Its repack helper adds the Gemma 4 QAT checkpoint to the `--model` selector so
the app can build any supported checkpoint without downloading the full source
model first:

- `gemma4`: `mlx-community/gemma-4-26b-a4b-it-4bit`
- `gemma4-qat`: `mlx-community/gemma-4-26B-A4B-it-qat-4bit`
- `qwen36`: `mlx-community/Qwen3.6-35B-A3B-4bit`
- `ornith15`: `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`
- `lfm25`: `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` (local; see "LFM2 MoE" below)

The old `--source standard|qat` spellings remain accepted as compatibility
aliases for saved workflows.

The helper verifies each source revision and model-index hash before it writes
an install. It records the source in the verified receipt so MLX Chat cannot
register one variant as the other.

Products stay `TurboFieldfare` and `TurboFieldfareRepack`. There is no
SwiftNIO, SwiftMath, or DecodeService target.

## Upstream #100 (ported, behavior only)

Metal command-buffer failures now throw instead of printing. The local port
adds the file-level `checkCommandBufferError` helper and converts `runSync`,
`waitForCompletion`, `finishPendingRoutedCommand`, and `sampleOnce`. CLI
runtime flags, README, benchmarks, `RuntimeConfiguration`, and
`AppModelTests` from that pull request are not included.

## Upstream #129 (not applicable)

The vanished-entry ENOENT skip is not ported. `VerifiedInstallTool.findUnexpectedEntries`
enumerates the install tree but never fetches throwing resource values and
never stats those enumerated URLs. Every `stat` / `fileSize` / `entryKind`
call on the verify-install path is against a manifest-declared path
(`manifest.json`, `packed_experts/layout.json`, and `manifest.files` keys).
`Model.verifyTrustedReceiptFileSize` is the same: declared receipt paths
only. No local scan can observe the readdir/fstatat race, so no ENOENT-skip
was added. `GTurboDirectoryAccess` exists here only because #87 needs a
directory file descriptor for dual-read loads, not because of #129.

## Upstream #87 (adapted port)

Hardened on-disk format and Model I/O from `d56c8084`. Local additions:

- `TurboFieldfareFormat` target with v1 codecs for the text install, plus
  optional Qwen family fields on `GTurboManifestArchV1` so Qwen/Ornith
  manifests stay valid.
- Dual-read for `manifest.json`, packed-expert layout, and the resident
  index: v1 codec first, then the pre-#87 JSON/legacy parser. Existing
  text-only `.gturbo` directories keep loading.
- Layout decode still caps at 64 MiB (Qwen layouts are ~22 MB). Upstream
  used 16 MiB.
- FD-based SHA-256 and optional `ResidentBuffer.fileDescriptor`.
- `Model` loads through `GTurboModelDirectory`. The Gemma-only
  `validateRuntimeSchema` from #87 is not applied; it would reject Qwen.
- `gdn` stays in `MetalContext` shader modules and subdirectories.

## Upstream #118 (adapted port)

O(1) incremental detokenize from `acefaf17`, re-implemented on this tree
rather than pasted:

- `GemmaDecoding`, `ByteFallbackRun`, `ByteLevelBPE`, and
  `PartialByteBuffer` do the per-token work. `Detokenizer.push` never
  calls `decode(tokens: stableIDs`.
- Hold-back is incremental partial-byte buffering on top of
  `GemmaDecoding`. Trailing byte-fallback and trailing U+FFFD for
  byte-level BPE stay buffered until a later token completes them.
- ChatML (Qwen/Ornith) applies the GPT-2 `unicodeToByte` map to the
  whole token, not only a single scalar. `Ġthe` becomes ` the`. Gemma
  stays on SentencePiece `▁` / `<0xNN>`. Gemma `pushText` is UTF-8
  only; it must not call `rawByte`, or a lone `é` piece is dropped.
- Flush assembles a complete tail and drops an incomplete one
  (`ByteFallbackRun.commit(dropIncomplete: true)`).
- Special and structural marker IDs come from the resolved dialect, not
  a Hub `tokenizer.json` walk. That walk would reject ChatML.
- `RawCompletion` passes `tokenizer.structuralMarkerIDs` as detokenizer
  barriers.

## Upstream #144 (adapted port)

Vision companion packs from `fd3c0b8b`, with these local changes:

- The M2 gate is the upstream one: `VisionRuntime.isSupported` is
  `device.supportsFamily(.apple8)`, and the error says the image tower
  requires an M2 or newer Apple Silicon Mac.
- `MetalContext` registers `vision`, `vision_register_gemm`, and
  `vision_resize` for `privateLibrary`, plus
  `includeVisionTensorOps`. They are not compiled into the shared
  runtime library, so a text-only Qwen or Ornith load does not compile
  the vision shaders. The `gdn` module and `Metal/GDN` subdirectory stay.
- Text repack still excludes `vision_tower.*`, `embed_vision.*`, and
  `audio_tower.*`. `planVisionCompanion` includes only `vision_tower.*`
  and `embed_vision.*`.
- `VisionPackWriter` accepts both Gemma SSD source IDs (standard and
  QAT). Upstream allowed only the standard checkpoint.
- Multimodal prefill (`MultimodalPrefillRunner`, `RawCompletion`
  `multimodalInput`, image embedding override, bidirectional sliding-
  window image blocks) sits on the existing Qwen/GDN `RealForwardRunner`.
  GDN paths are unchanged.
- `PrefillRuntimeConfig.maxChunkTokens` stays 128 for text. Image spans
  use a larger scratch limit (`VisionConfig.maximumPooledTokens`) without
  raising the selectable text chunk list.
- The #144 Apple10 BF16 projector kernels are not in this `tensorops.metal`.
  Vision converts activations to F16 and uses the existing affine F16 QMM.
- App catalog: Gemma/Qwen/Ornith SSD static entries stay
  `supportsVision: false`. A registered Gemma SSD install reports vision
  only when `Name.vision.gturbo/manifest.json` is on disk. Qwen and
  Ornith SSD identities never gain the attachment control. Ordinary MLX
  vision entries are unchanged.
- No 0.6.0 rebase and no #148.

## Upstream #182 (grouped full-attention decode, partial port)

`attention_decode_full_grouped_vec_partial` and the `Attention` routing that
selects it for the 512/16/2 full-attention shape, which only Gemma 4 uses. It
removes the per-position threadgroup barriers of the generic partial kernel.
Also ported: `ArchConfig.maxPositionEmbeddings`, the `KVCacheManager` guard
that refuses a context past it, and `ModelError.kvAllocationFailed`.
Not ported: `ContextAdmission` and the 256K context options; MLX Chat keeps
its own 4K–32K settings.

## Upstream #159 (TensorOps prefill by capability, ported)

`PrefillAttention` builds the MSL 4 TensorOps full-prefill pipeline on any
device that compiles it instead of gating it on Apple10, and keeps the
causal-tiled path as the fallback. The TensorOps shape now also requires a
full-attention layer and a window that never clips. Command-buffer checks read
`status` as well as `error`, because a failed buffer may carry no error
object, and `MetalContext.makeSystemDefaultDevice` sets
`AGX_RELAX_CDM_CTXSTORE_TIMEOUT` before the first device is created.

## LFM2 MoE (local)

LiquidAI's LFM2.5-24B-A2B streams from SSD through its own decode runner,
`LFM2MoEForwardRunner`, so Gemma and Qwen keep `RealForwardRunner` unchanged.

- Repack: `ArchInfo` reads the flat `lfm2_moe` config (layer mask 3 for gated
  short convolution, `linearConvKernelSize` carries `conv_L_cache`) and
  cross-checks the pinned shape. The planner accepts root `model.*` names and
  `feed_forward.switch_mlp` experts. The two dense-FFN layers own no experts,
  so only the 38 routed layers are packed, as packed layers 0–37; manifest and
  layout `numLayers` now record the packed count (equal to the model's for
  Gemma and Qwen). Logical shapes multiply before dividing so 6-bit tensors
  size correctly.
- Manifest: LFM records a 6-bit embedding, a 32-bit (unquantized float32)
  router, and 4-bit dense FFN; `ManifestReader` accepts those for the family.
- Runtime: a private `lfm2` Metal module adds the 6-bit embedding lookup, the
  6-bit tied lm_head GEMV, and the short-convolution decode step. The router
  runs on the CPU (sigmoid scores, `expert_bias` for selection only, top-4
  renormalized). The streamed-expert kernels are fixed at eight slots, so the
  four experts are dispatched twice with the second copy weighted zero.
  `Attention.maxQHeads` is 32 for LFM's 32 query heads.
- Prompts run token by token; there is no chunked LFM prefill.
- Tokenizer: `ChatMLFamily.lfm` prepends `<|startoftext|>`, resolves
  `<|tool_call_start|>` markers, and uses a 128,000-row vocabulary.
