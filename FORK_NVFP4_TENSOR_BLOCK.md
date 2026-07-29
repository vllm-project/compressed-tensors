# NVFP4 tensor-block fork (`nvfp4-tensor-block`)

Branch **`nvfp4-tensor-block`** adds **opt-in** support for the `NVFP4A16_BLOCK` preset.
Upstream `main` is unchanged unless you merge this branch.

## What is added (all opt-in)

| Change | When it applies |
|--------|-----------------|
| `QuantizationStrategy.TENSOR_BLOCK` | Only when `strategy=tensor_block` in `QuantizationArgs` |
| `NVFP4A16_BLOCK` preset | Only when scheme name is `NVFP4A16_BLOCK` |
| Block/global-scale init & forward paths | Same as above — gated on `TENSOR_BLOCK` |
| `forward_helpers._apply_global_scale` | NVFP4 paths with `global_scale` + FP8 local scales (correctness fix on this fork) |

Standard presets (`NVFP4A16`, `FP8`, etc.) on upstream `main` are not modified by the
`TENSOR_BLOCK` enum value itself; existing code paths are extended with `in (...)` checks.

## Install (optional — only for `NVFP4A16_BLOCK`)

```bash
git clone -b nvfp4-tensor-block https://github.com/GillchLee/compressed-tensors.git
pip install -e compressed-tensors
```

For `NVFP4A16` (1×16 tensor-group) only, upstream PyPI / vllm-project `main` is enough.

## Diff vs upstream `main`

```bash
git diff origin/main...nvfp4-tensor-block
```
