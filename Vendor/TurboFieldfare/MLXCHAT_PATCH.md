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
