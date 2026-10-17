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
