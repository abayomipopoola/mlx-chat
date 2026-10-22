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
- `flint`: `RecursiveSI/flint-int4` (local; see "Flint" below)

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
  optional family fields on `GTurboManifestArchV1` so Qwen, Ornith, LFM2, and
  Flint manifests validate directly through `GTurboManifestCodec` (including
  packed routed-layer counts for models with leading dense-FFN layers,
  `slidingWindow >= 0`, and `fullAttentionLayerMask` values `0...3`).
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

## Upstream open PRs #187 and #188 (early expert read, adapted port)

From the open pull requests at heads `717797cff41c061107751b8849a60fe9dab581ee`
(#187) and `0ad717015a7e2e13e7ec55a36c25b6e84195dea2` (#188), not yet merged
upstream. Ported: `ExpertStagingBuffer` and the staging API on
`PreadExpertStreamer` (`makeStagingBuffer`, `residentExperts`, `readEarly`,
`adoptStagedExpert`, `executeExpertCachePlan(_:skippingMisses:beforeAdmitting:)`),
`MoE.encodeRouterLogitsGemma4`, `Model.routedExpertStreamer(layer:)` and the
skipping fetch, `NextLayerExpertPrefetcher`, and its wiring into
`RealForwardRunner` decode. `RuntimeConfiguration.earlyExpertRead` is
`off | router`, default `router`.

Not ported: #188's fitted guess (`NextLayerGuess/`, `NextLayerGuessWeights`).
It covers only `mlx-community/gemma-4-26b-a4b-it-4bit`, which could not be
validated here. Default reads per layer is 1, not #188's 2: with the router
guess on a 36 GB M-series Mac, two reads cost Gemma 4 QAT 20% (56% of reads
used), while one read gained 2% on it and 6% on Ornith, with identical greedy
output. `TURBO_FIELDFARE_EARLY_EXPERT_READS` still overrides it. CLI and
server flags are not included.

## Prompt prefill (local)

`runRawPrefill` feeds a prompt into a continuable producer and stops, so the
app can warm the system prompt after load. `RawCompletionScratch.logitsBuffer`
exposes the logits for verification.

`FlintForwardRunner` and `LFM2MoEForwardRunner` prefill layer by layer over
chunks of up to 2,048 tokens (`prefillChunkTokens`). Projections are
matrix multiplies over the chunk; attention runs one threadgroup per (KV head,
token) with an online softmax; router scores come back with one GPU wait per
routed layer, and top-k reuses the decode selection. Routed experts run
through `PrefillGroupedRoutedMoE` tiles planned on the expert cache (cached
experts first, the next tile read while the previous one computes), so each
expert a chunk needs is read once. Every per-token result is independent of
the chunking, which the cross-chat resume test relies on. Results are close
to, not bit-equal with, token-by-token decode.

- Flint uses the tensor-op multiply (`MPPPrefillInt4QMM`) everywhere. On a
  1,235-token prompt the last-token logits correlated 0.9996 with
  token-by-token prefill and greedy output matched. Prefill went from
  27 tok/s to 353 tok/s on a 2,672-token prompt.
- LFM2.5 keeps conv and dense projections on the FP32-weight multiply
  (`exactQMMKinds`): the tensor-op multiply rounds dequantized weights to FP16,
  and with it everywhere the logits correlated only 0.96. With the default
  0.995 and the same top-1; prefill went from 28 tok/s to about 260 tok/s on
  a 1,283-token prompt. The short convolution runs a chunk's tokens in order
  per channel and writes the decode path's conv snapshots.
- `PrefillGroupedRoutedMoE(saturateFP16:)` sets function constant 78 so
  grouped activations and down outputs saturate to +/-65504, as Flint's decode
  clamp does; without it an early token's huge activation turned into NaN.
  Gemma and Qwen leave it unset, which compiles the kernels unchanged.

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
  renormalized) with an allocation-free top-k. The streamed-expert kernels
  are fixed at eight slots; phase 1 runs only the four routed slots and the
  padding slots keep zero activations and zero weights.
  `Attention.maxQHeads` is 32 for LFM's 32 query heads.
- Decode reads layer L+1's likeliest experts (its router over layer L's
  input, top 8) into L+1's cache while layer L runs.
- Prompts run token by token inside `prefillChunked`; only the last token runs
  the output head. Conv snapshots come every 8 positions below 1,024 (every
  64 after), so a resume at the shared system prompt replays at most 7 tokens.
- Tokenizer: `ChatMLFamily.lfm` prepends `<|startoftext|>`, resolves
  `<|tool_call_start|>` markers, and uses a 128,000-row vocabulary.

## Flint (local)

Recursive Flint 40B-A2.5B streams from SSD through `FlintForwardRunner`.
The checkpoint has no `model.safetensors.index.json` and stores affine
weights as U8 bytes; the installer writes a canonical index and folds those
bytes back into U32 words.

- 49 layers. Layers 0 and 1 are dense SwiGLU (width 6144). The other 47
  layers add a 1024-wide shared expert to the top 6 of 256 routed experts.
  Phase 1 runs only the six active slots; the two padding slots stay zero.
- 14 full-attention layers have no rotary encoding. The other 35 use NeoX
  rotary encoding and a 513-token window, including the current token.
  Every query head has a learned attention sink.
- Embeddings are 8-bit. The untied lm_head is 4-bit. The router is BF16.
- Prompts run token by token. The chat template ends at `<|assistant|>`
  with no newline (the Jinja tags trim both sides). With a newline the model
  skips `<think>` and its closing tag leaks into the answer.
- Decode waits on the GPU once per MoE layer, for the router scores (BF16
  GEMV on the GPU, top-k on the CPU). The shared expert then runs while the
  routed experts are read. The same scores pass also guesses layer L+1's top
  8 experts from layer L's input and starts reading them early.
- Attention is split-KV over 32-position chunks (`flint_attention_partial`),
  merged per query head with the sink (`flint_attention_combine`).
- Flint kernels live in `Metal/Flint/flint.metal` only; shared kernels are
  unchanged. Infinity saturation is a Flint kernel, not part of `moe.metal`.
- Sliding layers keep K/V in a ring of `slidingWindow + slidingRewindMargin`
  (default 4,096) slots, so the KV cache no longer grows with context for 35
  of 49 layers. `maxRewindDistance` bounds rewinds; deeper resumes reset.
- Q/K/V is one fused GEMV (`FusedQKVGEMV`), Q/K norms and rotary encoding are
  one kernel (`flint_qk_norm_rope`), and each residual add also writes the
  next RMSNorm (`flint_residual_rmsnorm`, bit-identical to the two kernels).
