# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Helper functions for packing and unpacking FP4 (E2M1) quantized weights.

FP4 E2M1 format uses 1 sign bit, 2 exponent bits, and 1 mantissa bit,
supporting 8 positive and 8 negative values. This module provides efficient
packing of two FP4 values into a single uint8 for storage.
"""

import torch
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import tl, triton, triton_req


__all__ = [
    "pack_fp4_to_uint8",
    "unpack_fp4_from_uint8",
    "quantize_and_pack_fp4",
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


@triton.jit
def _quantize_and_pack_fp4_kernel(
    packed_ptr,
    input_ptr,
    scale_ptr,
    zero_point_ptr,
    global_scale_ptr,
    num_rows,
    num_cols,
    group_size,
    BLOCK_SIZE: tl.constexpr,
    has_zero_point: tl.constexpr,
    has_global_scale: tl.constexpr,
):
    """
    Fused Triton kernel that quantizes input values to FP4 and packs them into uint8
    in a single pass, avoiding the intermediate FP4 float buffer.

    This kernel combines the logic from _quantize_kernel (FP4 path) and
    _pack_fp4_kernel:
    1. Scales input by quantization scale (and optionally global_scale)
    2. Adds zero_point if present
    3. Clamps to FP4 range [-6.0, 6.0]
    4. Maps directly to FP4 indices using threshold counting
    5. Packs two consecutive FP4 indices into one uint8

    This kernel assumes that x is contiguous. This is ensured by calling
    x.flatten() before calling this kernel.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Each output byte packs two consecutive input values
    n_pairs = (num_rows * num_cols) // 2
    mask = offsets < n_pairs

    low_idx = offsets * 2
    high_idx = offsets * 2 + 1

    low_row = low_idx // num_cols
    low_col = low_idx % num_cols
    high_row = high_idx // num_cols
    high_col = high_idx % num_cols

    x_low = tl.load(input_ptr + low_idx, mask=mask, other=0.0)
    x_high = tl.load(input_ptr + high_idx, mask=mask, other=0.0)

    num_groups = num_cols // group_size
    scale_low_idx = low_row * num_groups + low_col // group_size
    scale_high_idx = high_row * num_groups + high_col // group_size

    scale_low = tl.load(scale_ptr + scale_low_idx, mask=mask, other=1.0)
    scale_high = tl.load(scale_ptr + scale_high_idx, mask=mask, other=1.0)

    if has_global_scale:
        global_scale = tl.load(global_scale_ptr)
        scale_low = tl.div_rn(scale_low.to(tl.float32), global_scale.to(tl.float32))
        scale_high = tl.div_rn(scale_high.to(tl.float32), global_scale.to(tl.float32))

    x_low = tl.div_rn(x_low.to(tl.float32), scale_low.to(tl.float32))
    x_high = tl.div_rn(x_high.to(tl.float32), scale_high.to(tl.float32))

    if has_zero_point:
        zp_low = tl.load(zero_point_ptr + scale_low_idx, mask=mask, other=0.0)
        zp_high = tl.load(zero_point_ptr + scale_high_idx, mask=mask, other=0.0)
        x_low = x_low + zp_low
        x_high = x_high + zp_high

    # Clamp to FP4 range
    x_low = tl.clamp(x_low, -6.0, 6.0)
    x_high = tl.clamp(x_high, -6.0, 6.0)

    # Extract sign bit into bit 3 position (8 = 0b1000)
    sign_low = tl.where(x_low < 0.0, 8, 0).to(tl.uint8)
    sign_high = tl.where(x_high < 0.0, 8, 0).to(tl.uint8)

    abs_low = tl.abs(x_low)
    abs_high = tl.abs(x_high)

    # Use same float thresholds as _round_to_fp4 to compute FP4 index directly
    # Thresholds: >0.25→1, >=0.75→2, >1.25→3, >=1.75→4, >2.5→5, >=3.5→6, >5.0→7
    idx_low = (
        (abs_low > 0.25).to(tl.uint8)
        + (abs_low >= 0.75).to(tl.uint8)
        + (abs_low > 1.25).to(tl.uint8)
        + (abs_low >= 1.75).to(tl.uint8)
        + (abs_low > 2.5).to(tl.uint8)
        + (abs_low >= 3.5).to(tl.uint8)
        + (abs_low > 5.0).to(tl.uint8)
    )
    idx_low = idx_low | sign_low

    idx_high = (
        (abs_high > 0.25).to(tl.uint8)
        + (abs_high >= 0.75).to(tl.uint8)
        + (abs_high > 1.25).to(tl.uint8)
        + (abs_high >= 1.75).to(tl.uint8)
        + (abs_high > 2.5).to(tl.uint8)
        + (abs_high >= 3.5).to(tl.uint8)
        + (abs_high > 5.0).to(tl.uint8)
    )
    idx_high = idx_high | sign_high

    # Pack nibbles
    packed = idx_low | (idx_high << 4)

    tl.store(packed_ptr + offsets, packed, mask=mask)


def quantize_and_pack_fp4(
    x: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    group_size: int = 16,
) -> torch.Tensor:
    """
    Fused quantization and packing for FP4 (E2M1) format.

    This function combines quantization (scale, clamp, round to FP4) and packing
    (two FP4 values into one uint8) into a single kernel launch, avoiding the
    intermediate FP4 float buffer and providing ~1-2ms savings.

    :param x: input tensor to quantize and pack, shape [m, n] where n is even
    :param scale: quantization scale, shape [m, n // group_size]
    :param global_scale: optional global scale for NVFP4 format
    :param zero_point: optional zero point for asymmetric quantization
    :param group_size: number of elements per quantization group (default 16)
    :returns: packed tensor in uint8, shape [m, n // 2]
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {x.ndim}D")

    m, n = x.shape

    if n % 2 != 0:
        raise ValueError(
            "tensor must have an even number of columns for nvfp4 compression"
        )

    # GPU path using fused Triton kernel
    if triton_req(x):
        # FP4 packing requires contiguous input since we pack consecutive pairs
        x_flat = x.contiguous().flatten()
        n_pairs = x_flat.numel() // 2
        output_shape = (m, n // 2)

        # Flatten scale for kernel access (must be contiguous)
        scale_flat = scale.flatten().contiguous()

        # Handle zero_point - pass x_flat as dummy when None
        zp_flat = (
            zero_point.flatten().contiguous() if zero_point is not None else x_flat
        )

        packed = torch.empty(n_pairs, dtype=torch.uint8, device=x.device)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_pairs, BLOCK_SIZE),)

        # Pass x_flat as dummy for global_scale when None
        _quantize_and_pack_fp4_kernel[grid](
            packed,
            x_flat,
            scale_flat,
            zp_flat,
            global_scale if global_scale is not None else x_flat,
            m,
            n,
            group_size,
            BLOCK_SIZE,
            has_zero_point=zero_point is not None,
            has_global_scale=global_scale is not None,
        )

        return packed.reshape(output_shape)

    # CPU fallback: use separate quantize + pack
    # Import here to avoid circular dependency
    from compressed_tensors.quantization import QuantizationArgs, QuantizationType
    from compressed_tensors.quantization.lifecycle.forward import quantize

    args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        group_size=group_size,
        symmetric=zero_point is None,
    )

    quantized = quantize(
        x=x,
        scale=scale,
        global_scale=global_scale,
        zero_point=zero_point,
        args=args,
    )

    return pack_fp4_to_uint8(quantized)


@ImplBackend.register("pack_fp4_to_uint8", triton_req, 0)
def pack_fp4_to_uint8_triton(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape

    if x.dtype not in (torch.bfloat16, torch.float16):
        x = x.to(torch.bfloat16)
    x_flat = x.contiguous().flatten()
    n_pairs = x_flat.numel() // 2

    packed = torch.empty(n_pairs, dtype=torch.uint8, device=x.device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pairs, BLOCK_SIZE),)
    _pack_fp4_kernel[grid](x_flat, packed, n_pairs, BLOCK_SIZE)

    return packed.reshape(m, n // 2)


@ImplBackend.entrypoint("pack_fp4_to_uint8")
def pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
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
    :returns: a packed tensor in uint8
    """
    m, n = x.shape

    if n % 2 != 0:
        raise ValueError(
            "tensor must have an even number of columns for nvfp4 compression"
        )

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
