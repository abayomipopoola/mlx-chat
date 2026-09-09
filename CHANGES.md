# mlx-chat: multi-file attachments + context-window preflight warning

Base repo: https://github.com/abayomipopoola/mlx-chat (main, as of 2026-09-09)

## How to apply

```
git clone https://github.com/abayomipopoola/mlx-chat.git
cd mlx-chat
git apply /path/to/multi-file-attachments.patch
xcodegen generate
xcodebuild -project MLXChat.xcodeproj -scheme MLXChat -configuration Debug \
  -skipPackagePluginValidation -skipMacroValidation build
```

If you want it as an actual GitHub fork under your account, click "Fork" on the
repo page first, then `git clone` your fork instead of upstream and apply the
patch there — I don't have write access to push it myself.

## What changed

**Multiple file attachments**
- `AttachmentPicker.pickFile()` now sets `allowsMultipleSelection = true` and
  returns every successfully-read file plus a list of any that failed to read,
  so one bad file doesn't block the rest.
- `ComposerView` tracks `pendingFiles: [(name, text)]` instead of a single
  optional. Each attached file gets its own removable chip
  (`PendingAttachmentStrip`), wrapping onto additional lines via a small
  `FlowLayout` once several are attached.
- `PromptBuilder.userText(content:attachments:)` appends one `[File: name]`
  block per attachment. The old single-attachment overload is kept so
  existing call sites/tests still compile.
- `ChatMessage` (SwiftData) gained `attachmentNames`/`attachmentTexts` arrays.
  The old singular `attachmentName`/`attachmentText` fields are kept (nil for
  new messages) purely so previously-saved conversations still load under
  lightweight migration; a computed `attachments` property merges both.

**Context-window preflight warning**
- `PromptBuilder` gained `estimatedTokens`, `budget(ctxTokens:)`, and
  `attachmentsExceedBudget(draft:attachments:ctxTokens:)` — the same ~4
  chars/token estimate and 70%-of-window budget the engines already use for
  history trimming, now exposed for a pre-send check.
- `ChatController.selectedModelContextTokens` surfaces the right budget for
  whichever engine is selected (Apple Intelligence's `contextSize`, or the
  local model's `runtimeContextTokens`).
- `ComposerView` runs this check continuously against the draft + all pending
  files. If it's over budget:
  - the send button disables and its tooltip explains why,
  - an orange `ContextWarningBanner` appears above the composer stating the
    estimated token count vs. the budget, and
  - **nothing is sent** — the check runs client-side before `chat.send(...)`
    is ever called, so it never reaches the model/API only to be silently
    truncated or to fail.

This only prevents the case the existing `PromptBuilder.trimmed()` couldn't
already handle: it never drops the final pending message even when that
message alone is oversized. The new check catches exactly that case, before
the call.

## Files touched
- MLXChat/App/ChatController.swift
- MLXChat/Data/ConversationStore.swift
- MLXChat/Data/Models.swift
- MLXChat/Engine/PromptBuilder.swift
- MLXChat/Support/AttachmentPicker.swift
- MLXChat/Views/ChatScreen.swift
- MLXChat/Views/ComposerSubviews.swift
- MLXChat/Views/ComposerView.swift
- MLXChat/Views/MessageView.swift
- Tests/MLXChatTests/AttachmentTests.swift
- Tests/MLXChatUITests/AttachmentFlowUITests.swift

## Follow-up fix: attach menu no longer requires a vision model
Original code hid the entire `+` attach menu (both photo *and* file) unless
`chat.selectedModelSupportsVision` was true. That's correct for photos, but
wrong for text-file attachment — a plain `.txt` file doesn't need a VLM, it's
just appended to the prompt as text. Most small local MLX models aren't
vision-capable, so this gate was silently hiding the new multi-file feature
entirely for anyone not on a VLM or Apple Intelligence.

Fixed by decoupling the two:
- `ComposerView.showsAttachMenu` is now unconditionally `true` — the `+`
  button always shows.
- `AttachPanel` now gates only the "Photo or Image…" row on
  `chat.selectedModelSupportsVision` (or `--fake-vision-support`); the
  "Files…" row always shows.

## Known limitations
- Two attached files with the same filename will collide in `ForEach`
  identity in `PendingAttachmentStrip` (cosmetic only — SwiftUI diffing
  hiccup, not a crash). Worth de-duping display names if that comes up.
- Token estimate is the same rough ~4-chars/token heuristic the app already
  used for trimming, not a real tokenizer count, so the warning threshold is
  approximate by design (matches existing behavior, just applied earlier).
- I couldn't build/test this in the sandbox (no Xcode/macOS toolchain here) —
  worth a real `xcodebuild` pass before you rely on it.
