# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Helper functions for packing and unpacking FP4 (E2M1) quantized weights.

FP4 E2M1 format uses 1 sign bit, 2 exponent bits, and 1 mantissa bit,
supporting 8 positive and 8 negative values. This module provides efficient
packing of two FP4 values into a single uint8 for storage.
"""

import torch
import triton
import triton.language as tl
from compressed_tensors.quantization.lifecycle.forward_helpers import QuantBufferPool


__all__ = [
    "pack_fp4_to_uint8",
    "unpack_fp4_from_uint8",
    "QuantBufferPool",
]


FLOAT_TO_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


@triton.jit
def _pack_fp4_kernel(
    x_ptr,
    packed_ptr,
    n_pairs,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for packing FP4 values using sign-based direct computation.

    This kernel extracts the sign bit, converts to absolute values scaled by 2,
    then uses threshold counting to directly compute indices without cascading
    conditionals. The sign bit is applied via bitwise OR.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pairs

    # Load pairs of values
    low_idx = offsets * 2
    high_idx = offsets * 2 + 1

    x_low = tl.load(x_ptr + low_idx, mask=mask, other=0.0)
    x_high = tl.load(x_ptr + high_idx, mask=mask, other=0.0)

    # Extract sign bit directly into bit 3 via bitcast (handles -0.0 correctly)
    sign_low = (x_low.to(tl.int16, bitcast=True) >> 12 & 8).to(tl.uint8)
    sign_high = (x_high.to(tl.int16, bitcast=True) >> 12 & 8).to(tl.uint8)

    # Scale and absolute
    x_low_abs = tl.abs(x_low * 2.0).to(tl.int8)
    x_high_abs = tl.abs(x_high * 2.0).to(tl.int8)

    # Direct index computation via threshold counting
    # Count how many thresholds each value meets or exceeds
    # Thresholds: 1, 2, 3, 4, 6, 8, 12 (scaled FP4 values)
    idx_low = (
        (x_low_abs >= 1).to(tl.uint8)
        + (x_low_abs >= 2).to(tl.uint8)
        + (x_low_abs >= 3).to(tl.uint8)
        + (x_low_abs >= 4).to(tl.uint8)
        + (x_low_abs >= 6).to(tl.uint8)
        + (x_low_abs >= 8).to(tl.uint8)
        + (x_low_abs >= 12).to(tl.uint8)
    )
    idx_low = idx_low | sign_low

    idx_high = (
        (x_high_abs >= 1).to(tl.uint8)
        + (x_high_abs >= 2).to(tl.uint8)
        + (x_high_abs >= 3).to(tl.uint8)
        + (x_high_abs >= 4).to(tl.uint8)
        + (x_high_abs >= 6).to(tl.uint8)
        + (x_high_abs >= 8).to(tl.uint8)
        + (x_high_abs >= 12).to(tl.uint8)
    )
    idx_high = idx_high | sign_high

    # Pack nibbles
    packed = idx_low | (idx_high << 4)

    tl.store(packed_ptr + offsets, packed, mask=mask)


def pack_fp4_to_uint8(
    x: torch.Tensor,
    use_buffer_pool: bool = True,
) -> torch.Tensor:
    """
    Packs a tensor with values in the fp4 range into uint8.
    As there are 16 valid fp4 values, two fp4 values can be
    packed into one uint8. Each fp4 value is mapped to its
    particular index (e.g. 0.5 is mapped to index 1, 6.0 is mapped
    to index 7) which is then represented using 4 bits. Consecutive
    pairs of 4 bits are then packed into an uint8.

    IMPORTANT: This assumes x contains ONLY valid FP4 values. If called with
    non-quantized data, results will be incorrect. This function should only be
    called after _cast_to_fp4() or equivalent quantization.

    :param x: tensor to pack
    :param use_buffer_pool: if True, reuse buffers to avoid cudaMalloc overhead
    :returns: a packed tensor in uint8
    """
    m, n = x.shape

    if n % 2 != 0:
        raise ValueError(
            "tensor must have an even number of columns for nvfp4 compression"
        )

    # Use Triton kernel on GPU (CUDA, ROCm, XPU)
    if x.is_cuda or x.is_xpu:
        x_flat = x.contiguous().flatten()
        n_pairs = x_flat.numel() // 2

        # Use buffer pool to avoid repeated cudaMalloc calls
        if use_buffer_pool:
            packed = QuantBufferPool.get_buffer((n_pairs,), torch.uint8, x.device)
        else:
            packed = torch.empty(n_pairs, dtype=torch.uint8, device=x.device)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_pairs, BLOCK_SIZE),)
        _pack_fp4_kernel[grid](x_flat, packed, n_pairs, BLOCK_SIZE)

        return packed.reshape(m, n // 2)

    # CPU fallback
    # Extract sign before conversion
    sign = torch.signbit(x).to(torch.uint8)

    # Scale by 2 and convert to int8
    x = (x * 2).to(torch.int8).abs_()

    indices = torch.zeros_like(x, dtype=torch.uint8)

    # 8-way assignment (only positive values)
    indices[x == 1] = 1
    indices[x == 2] = 2
    indices[x == 3] = 3
    indices[x == 4] = 4
    indices[x == 6] = 5
    indices[x == 8] = 6
    indices[x >= 12] = 7

    # Apply sign bit
    indices = indices | (sign << 3)

    indices = indices.reshape(-1, 2)
    packed = indices[:, 0] | (indices[:, 1] << 4)

    return packed.reshape(m, n // 2)


# reference: https://github.com/vllm-project/vllm/pull/16362
def unpack_fp4_from_uint8(
    a: torch.Tensor, m: int, n: int, dtype: torch.dtype | None = torch.bfloat16
) -> torch.Tensor:
    """
    Unpacks uint8 values into fp4. Each uint8 consists of two fp4 values
    (i.e. first four bits correspond to one fp4 value, last four correspond to a
    consecutive fp4 value). The bits represent an index, which are mapped to an fp4
    value.

    :param a: tensor to unpack
    :param m: original dim 0 size of the unpacked tensor
    :param n: original dim 1 size of the unpacked tensor
    :param dtype: dense dtype to cast the unpacked tensor to
    """
    assert a.dtype == torch.uint8

    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles

    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()

    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

    # Device-aware lookup and sign application
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n).to(dtype=dtype)
