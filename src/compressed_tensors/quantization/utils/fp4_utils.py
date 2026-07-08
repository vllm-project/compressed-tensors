# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.compiler
import triton
import triton.language as tl


__all__ = ["cast_to_fp4"]


@triton.jit
def _cast_to_fp4_kernel(
    input_ptr,
    output_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for FP4 E2M1 quantization.

    Maps float values to the nearest E2M1 representable value:
    0.0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Extract sign and absolute value
    sign = tl.where(x < 0.0, -1.0, 1.0)
    abs_x = tl.abs(x)

    # Map absolute values to FP4 representable values
    # Using sequential tl.where for the quantization mapping
    result = tl.zeros_like(abs_x)
    result = tl.where(abs_x > 0.25, 0.5, result)
    result = tl.where(abs_x >= 0.75, 1.0, result)
    result = tl.where(abs_x > 1.25, 1.5, result)
    result = tl.where(abs_x >= 1.75, 2.0, result)
    result = tl.where(abs_x > 2.5, 3.0, result)
    result = tl.where(abs_x >= 3.5, 4.0, result)
    result = tl.where(abs_x > 5.0, 6.0, result)

    # Restore sign
    result *= sign
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.compile(dynamic=True)
def _cast_to_fp4_cpu(x):
    """
    CPU implementation for FP4 E2M1 quantization

    Maps float values to the nearest E2M1 representable value:
    0.0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
    """
    sign = torch.empty_like(x)
    torch.sign(x, out=sign)
    abs_x = torch.abs(x)

    # Map absolute values to FP4 representable values
    # Using sequential torch.where for readability while maintaining memory efficiency
    result = torch.empty_like(x)
    result = torch.where(abs_x <= 0.25, 0.0, result)
    result = torch.where(abs_x > 0.25, 0.5, result)
    result = torch.where(abs_x >= 0.75, 1.0, result)
    result = torch.where(abs_x > 1.25, 1.5, result)
    result = torch.where(abs_x >= 1.75, 2.0, result)
    result = torch.where(abs_x > 2.5, 3.0, result)
    result = torch.where(abs_x >= 3.5, 4.0, result)
    result = torch.where(abs_x > 5.0, 6.0, result)

    return result * sign.to(result)


def cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round float values to the nearest E2M1 representable value.

    Uses Triton for GPU tensors and torch.compile for CPU tensors.

    :param x: input tensor to quantize
    :param tile_size: block size for Triton kernel (default 128K)
    :return: FP4-quantized tensor with same shape as input
    """
    shape = x.shape
    x = x.flatten()

    match x.device.type:
        case "cpu" | "meta":
            return _cast_to_fp4_cpu(x).reshape(shape)

        case _:
            output = torch.empty_like(x)
            n = x.numel()
            block_size = 1024

            # Use tile_size as BLOCK_SIZE for Triton kernel
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)  # noqa: E731
            _cast_to_fp4_kernel[grid](x, output, n, BLOCK_SIZE=block_size)

            return output.reshape(shape)
