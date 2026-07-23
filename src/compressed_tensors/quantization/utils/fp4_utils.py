# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import triton
import triton.language as tl


__all__ = ["cast_to_fp4_triton", "cast_to_fp4_torch"]


@triton.jit
def _cast_to_fp4_kernel(
    input_ptr,
    output_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x = x.to(tl.bfloat16)

    sign_bit = x.to(tl.int16, bitcast=True) & (-32768)
    abs_x = tl.abs(x)

    result = tl.zeros_like(abs_x)
    result = tl.where(abs_x > 0.25, 0.5, result)
    result = tl.where(abs_x >= 0.75, 1.0, result)
    result = tl.where(abs_x > 1.25, 1.5, result)
    result = tl.where(abs_x >= 1.75, 2.0, result)
    result = tl.where(abs_x > 2.5, 3.0, result)
    result = tl.where(abs_x >= 3.5, 4.0, result)
    result = tl.where(abs_x > 5.0, 6.0, result)

    result = (result.to(tl.int16, bitcast=True) | sign_bit).to(
        tl.bfloat16, bitcast=True
    )
    tl.store(output_ptr + offsets, result, mask=mask)


def cast_to_fp4_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation for FP4 E2M1 quantization

    Maps float values to the nearest E2M1 representable value:
    0.0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
    """
    shape = x.shape
    x = x.contiguous().flatten()
    output = torch.empty_like(x)
    n = x.numel()
    block_size = 1024

    # Use tile_size as BLOCK_SIZE for Triton kernel
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)  # noqa: E731
    _cast_to_fp4_kernel[grid](x, output, n, BLOCK_SIZE=block_size)

    return output.reshape(shape)


@torch.compile(dynamic=True)
def _cast_to_fp4_torch_impl(x):
    sign = torch.sign(x)
    abs_x = torch.abs(x)

    result = torch.where(abs_x <= 0.25, 0.0, torch.empty_like(x))
    result = torch.where(abs_x > 0.25, 0.5, result)
    result = torch.where(abs_x >= 0.75, 1.0, result)
    result = torch.where(abs_x > 1.25, 1.5, result)
    result = torch.where(abs_x >= 1.75, 2.0, result)
    result = torch.where(abs_x > 2.5, 3.0, result)
    result = torch.where(abs_x >= 3.5, 4.0, result)
    result = torch.where(abs_x > 5.0, 6.0, result)

    return result * sign


def cast_to_fp4_torch(x):
    """
    CPU implementation for FP4 E2M1 quantization

    Maps float values to the nearest E2M1 representable value:
    0.0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
    """
    orig_dtype = x.dtype
    return _cast_to_fp4_torch_impl(x.to(torch.bfloat16)).to(orig_dtype)
