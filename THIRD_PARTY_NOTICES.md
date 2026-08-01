# Third-party notices

## TurboFieldfare

MLX Chat vendors the `TurboFieldfare` Swift package from main commit
`76f6611467207a80c4a4d72bd11c2818f0703739`, extended with the Qwen 3.6 work
from pull-request head `bddb8e5fd55d96e821007bc152412e384a14abba`. The local
patch keeps the additional Gemma 4 QAT checkpoint and exposes all three pinned
sources through the bundled repack helper.

TurboFieldfare is Copyright Andrey Mikhaylov and contributors and is licensed
under the Apache License, Version 2.0. Its source and license are available at
<https://github.com/drumih/turbo-fieldfare>.

Model weights are not part of MLX Chat or TurboFieldfare. The local installer
uses these pinned sources:

- `mlx-community/gemma-4-26b-a4b-it-4bit` at revision
  `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`
- `mlx-community/gemma-4-26B-A4B-it-qat-4bit` at revision
  `0e3cbab38ce568cf6e23543010d08d03b731910c`
- `mlx-community/Qwen3.6-35B-A3B-4bit` at revision
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`

Those files remain subject to their source terms.
