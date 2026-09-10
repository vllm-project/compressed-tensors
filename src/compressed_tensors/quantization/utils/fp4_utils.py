# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import tl, triton, triton_req


__all__ = ["_round_to_fp4", "cast_to_fp4"]


@triton.jit
def _round_to_fp4(x):
    """
    Round float values to the nearest E2M1 representable value.
    FP4 values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and their negatives)

    """
    sign = tl.where(x < 0.0, -32.0, 32.0)
    x = tl.abs(x)

    # moves all values from 0 to .25 to 0. We do this first to clear up space
    # to store the other rounded values temporarily in 0-.25 range.
    x = tl.where(x <= 0.25, 0.0, x)

    # starting with largest bucket, round values to fp4 values divided by 32.
    # this moves each value temporarily into the 0 to .25 range so it won't be
    # picked up by subsequent threshold checks.
    x = tl.where(x > 5.0, 6.0 / 32.0, x)
    x = tl.where(x >= 3.5, 4.0 / 32.0, x)
    x = tl.where(x > 2.5, 3.0 / 32.0, x)
    x = tl.where(x >= 1.75, 2.0 / 32.0, x)
    x = tl.where(x > 1.25, 1.5 / 32.0, x)
    x = tl.where(x >= 0.75, 1.0 / 32.0, x)
    x = tl.where(x > 0.25, 0.5 / 32.0, x)

    #  sign is sign(x_orig)*32 so will rescale everything to exact fp4
    return x * sign


@triton.jit
def _cast_to_fp4_kernel(
    x_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = _round_to_fp4(x)
    tl.store(x_ptr + offsets, result, mask=mask)


@ImplBackend.register("cast_to_fp4", triton_req, 0)
def cast_to_fp4_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation for FP4 E2M1 quantization

    Maps float values to the nearest E2M1 representable value:
    0.0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
    """
    shape = x.shape
    x = x.contiguous().flatten()
    n = x.numel()
    block_size = 1024

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)  # noqa: E731
    _cast_to_fp4_kernel[grid](x, n, BLOCK_SIZE=block_size)

    return x.reshape(shape)


@ImplBackend.entrypoint("cast_to_fp4")
def cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """
    Cast float values to the nearest FP4 E2M1 representable value.

    Uses the Triton kernel for CUDA/XPU tensors, falls back to a
    torch.compile implementation for CPU tensors.

    Maps float values to the nearest E2M1 representable value:
    0.0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
    """
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign
