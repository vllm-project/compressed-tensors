# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Helper functions for packing and unpacking quantized weights into int32 format.

These functions enable efficient storage of sub-8-bit quantized weights by packing
multiple values into 32-bit integers.
"""

import math
from typing import Literal

import torch
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import tl, triton, triton_req


__all__ = [
    "pack_to_int32",
    "pack_to_int32_accelerated",
    "unpack_from_int32",
]


# =============================================================================
# Triton kernels for bit-packing operations
# =============================================================================


@triton.jit
def _pack_to_int32_row_parallel_kernel(
    input_ptr,
    output_ptr,
    rows_g,
    num_bits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for packing 32-element groups into num_bits int32 words.

    Parallelizes over row groups (each row is a 32-element group).
    Each program instance processes BLOCK_SIZE groups.
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < rows_g

    # 8 accumulators for up to 8 output words (num_bits in [1,8])
    out0 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    out1 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    out2 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    out3 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    out4 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    out5 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    out6 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    out7 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for elem_i in range(32):
        input_offset = row_offsets * 32 + elem_i
        val = tl.load(input_ptr + input_offset, mask=row_mask, other=0)

        bit_start = elem_i * num_bits
        word_idx = bit_start // 32
        bit_offset = bit_start % 32

        shifted_lo = val << bit_offset
        overflow_bits = bit_offset + num_bits - 32
        has_overflow = overflow_bits > 0

        # High bits that overflow to next word (right-shifted)
        shifted_hi = tl.where(has_overflow, val >> (num_bits - overflow_bits), 0)

        # Accumulate into output word (word_idx is compile-time constant per elem_i)
        out0 = tl.where(word_idx == 0, out0 | shifted_lo, out0)
        out1 = tl.where(word_idx == 1, out1 | shifted_lo, out1)
        out2 = tl.where(word_idx == 2, out2 | shifted_lo, out2)
        out3 = tl.where(word_idx == 3, out3 | shifted_lo, out3)
        out4 = tl.where(word_idx == 4, out4 | shifted_lo, out4)
        out5 = tl.where(word_idx == 5, out5 | shifted_lo, out5)
        out6 = tl.where(word_idx == 6, out6 | shifted_lo, out6)
        out7 = tl.where(word_idx == 7, out7 | shifted_lo, out7)

        # Handle overflow to next word (elements spanning word boundaries)
        # next_word == 8 would overflow output bounds, but doesn't happen
        # for valid num_bits in [1,8] with 32 elements
        if has_overflow:
            next_word = word_idx + 1
            out1 = tl.where(next_word == 1, out1 | shifted_hi, out1)
            out2 = tl.where(next_word == 2, out2 | shifted_hi, out2)
            out3 = tl.where(next_word == 3, out3 | shifted_hi, out3)
            out4 = tl.where(next_word == 4, out4 | shifted_hi, out4)
            out5 = tl.where(next_word == 5, out5 | shifted_hi, out5)
            out6 = tl.where(next_word == 6, out6 | shifted_hi, out6)
            out7 = tl.where(next_word == 7, out7 | shifted_hi, out7)

    # Store output words
    if num_bits >= 1:
        tl.store(output_ptr + row_offsets * num_bits + 0, out0, mask=row_mask)
    if num_bits >= 2:
        tl.store(output_ptr + row_offsets * num_bits + 1, out1, mask=row_mask)
    if num_bits >= 3:
        tl.store(output_ptr + row_offsets * num_bits + 2, out2, mask=row_mask)
    if num_bits >= 4:
        tl.store(output_ptr + row_offsets * num_bits + 3, out3, mask=row_mask)
    if num_bits >= 5:
        tl.store(output_ptr + row_offsets * num_bits + 4, out4, mask=row_mask)
    if num_bits >= 6:
        tl.store(output_ptr + row_offsets * num_bits + 5, out5, mask=row_mask)
    if num_bits >= 7:
        tl.store(output_ptr + row_offsets * num_bits + 6, out6, mask=row_mask)
    if num_bits >= 8:
        tl.store(output_ptr + row_offsets * num_bits + 7, out7, mask=row_mask)


def _pack_row_parallel_triton(value_g: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Triton implementation of the core bit-packing operation.

    Takes groups of 32 elements and packs them into num_bits int32 words.

    :param value_g: Input tensor of shape (rows_g, 32), dtype int32, unsigned values
    :param num_bits: Number of bits per element (1-8)
    :return: Packed tensor of shape (rows_g, num_bits), dtype int32
    """
    rows_g = value_g.shape[0]
    output_g = torch.zeros(rows_g, num_bits, dtype=torch.int32, device=value_g.device)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(rows_g, BLOCK_SIZE),)

    _pack_to_int32_row_parallel_kernel[grid](
        value_g,
        output_g,
        rows_g,
        num_bits,
        BLOCK_SIZE,
    )

    return output_g


def _pack_row_parallel_triton_req(value_g: torch.Tensor, num_bits: int) -> bool:
    """Check if Triton implementation can be used."""
    return triton_req(value_g)


@ImplBackend.register("_pack_row_parallel", _pack_row_parallel_triton_req, 0)
def _pack_row_parallel_triton_impl(
    value_g: torch.Tensor, num_bits: int
) -> torch.Tensor:
    """ImplBackend-registered Triton implementation."""
    return _pack_row_parallel_triton(value_g, num_bits)


@ImplBackend.entrypoint("_pack_row_parallel")
def _pack_row_parallel(value_g: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Pack groups of 32 elements into num_bits int32 words.

    This is the core bit-packing operation. PyTorch fallback uses scatter_add.

    :param value_g: Input tensor of shape (rows_g, 32), dtype int32, unsigned values
    :param num_bits: Number of bits per element (1-8)
    :return: Packed tensor of shape (rows_g, num_bits), dtype int32
    """
    rows_g = value_g.shape[0]
    device = value_g.device

    output_g = torch.zeros(rows_g, num_bits, dtype=torch.int32, device=device)

    elem_i = torch.arange(32, device=device, dtype=torch.int32)
    bit_starts = elem_i * num_bits
    word_idx = (bit_starts // 32).long()
    bit_offset = bit_starts % 32

    output_g.scatter_add_(
        1,
        word_idx.unsqueeze(0).expand(rows_g, -1),
        value_g << bit_offset.unsqueeze(0),
    )

    ov = bit_offset + num_bits - 32
    ov_mask = ov > 0
    if ov_mask.any():
        ov_vals = value_g[:, ov_mask] >> (num_bits - ov[ov_mask]).unsqueeze(0)
        output_g.scatter_add_(
            1,
            (word_idx[ov_mask] + 1).unsqueeze(0).expand(rows_g, -1),
            ov_vals,
        )

    return output_g


@triton.jit
def _pack_to_int32_col_parallel_kernel(
    input_ptr,
    output_ptr,
    rows,
    cols,
    packed_rows,
    padded_rows,
    input_row_stride,
    output_row_stride,
    num_bits: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    """
    Triton kernel for packing along dim 0 without transpose.

    Parallelizes over columns. For each column, packs groups of 32 rows
    into num_bits output words.
    """
    pid = tl.program_id(0)
    col_start = pid * BLOCK_SIZE_COL
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COL)
    col_mask = col_offsets < cols

    num_groups = padded_rows // 32

    for group_idx in range(num_groups):
        out0 = tl.zeros((BLOCK_SIZE_COL,), dtype=tl.int32)
        out1 = tl.zeros((BLOCK_SIZE_COL,), dtype=tl.int32)
        out2 = tl.zeros((BLOCK_SIZE_COL,), dtype=tl.int32)
        out3 = tl.zeros((BLOCK_SIZE_COL,), dtype=tl.int32)
        out4 = tl.zeros((BLOCK_SIZE_COL,), dtype=tl.int32)
        out5 = tl.zeros((BLOCK_SIZE_COL,), dtype=tl.int32)
        out6 = tl.zeros((BLOCK_SIZE_COL,), dtype=tl.int32)
        out7 = tl.zeros((BLOCK_SIZE_COL,), dtype=tl.int32)

        row_base = group_idx * 32

        for elem_i in range(32):
            row = row_base + elem_i
            input_offset = row * input_row_stride + col_offsets
            row_valid = row < rows
            val = tl.load(input_ptr + input_offset, mask=col_mask & row_valid, other=0)

            bit_start = elem_i * num_bits
            word_idx = bit_start // 32
            bit_offset = bit_start % 32

            shifted_lo = val << bit_offset
            overflow_bits = bit_offset + num_bits - 32
            has_overflow = overflow_bits > 0
            shifted_hi = tl.where(has_overflow, val >> (num_bits - overflow_bits), 0)

            out0 = tl.where(word_idx == 0, out0 | shifted_lo, out0)
            out1 = tl.where(word_idx == 1, out1 | shifted_lo, out1)
            out2 = tl.where(word_idx == 2, out2 | shifted_lo, out2)
            out3 = tl.where(word_idx == 3, out3 | shifted_lo, out3)
            out4 = tl.where(word_idx == 4, out4 | shifted_lo, out4)
            out5 = tl.where(word_idx == 5, out5 | shifted_lo, out5)
            out6 = tl.where(word_idx == 6, out6 | shifted_lo, out6)
            out7 = tl.where(word_idx == 7, out7 | shifted_lo, out7)

            if has_overflow:
                next_word = word_idx + 1
                out1 = tl.where(next_word == 1, out1 | shifted_hi, out1)
                out2 = tl.where(next_word == 2, out2 | shifted_hi, out2)
                out3 = tl.where(next_word == 3, out3 | shifted_hi, out3)
                out4 = tl.where(next_word == 4, out4 | shifted_hi, out4)
                out5 = tl.where(next_word == 5, out5 | shifted_hi, out5)
                out6 = tl.where(next_word == 6, out6 | shifted_hi, out6)
                out7 = tl.where(next_word == 7, out7 | shifted_hi, out7)

        # Store output words for this group
        out_row_base = group_idx * num_bits

        if num_bits >= 1:
            out_row = out_row_base + 0
            out_mask = col_mask & (out_row < packed_rows)
            tl.store(
                output_ptr + out_row * output_row_stride + col_offsets,
                out0,
                mask=out_mask,
            )
        if num_bits >= 2:
            out_row = out_row_base + 1
            out_mask = col_mask & (out_row < packed_rows)
            tl.store(
                output_ptr + out_row * output_row_stride + col_offsets,
                out1,
                mask=out_mask,
            )
        if num_bits >= 3:
            out_row = out_row_base + 2
            out_mask = col_mask & (out_row < packed_rows)
            tl.store(
                output_ptr + out_row * output_row_stride + col_offsets,
                out2,
                mask=out_mask,
            )
        if num_bits >= 4:
            out_row = out_row_base + 3
            out_mask = col_mask & (out_row < packed_rows)
            tl.store(
                output_ptr + out_row * output_row_stride + col_offsets,
                out3,
                mask=out_mask,
            )
        if num_bits >= 5:
            out_row = out_row_base + 4
            out_mask = col_mask & (out_row < packed_rows)
            tl.store(
                output_ptr + out_row * output_row_stride + col_offsets,
                out4,
                mask=out_mask,
            )
        if num_bits >= 6:
            out_row = out_row_base + 5
            out_mask = col_mask & (out_row < packed_rows)
            tl.store(
                output_ptr + out_row * output_row_stride + col_offsets,
                out5,
                mask=out_mask,
            )
        if num_bits >= 7:
            out_row = out_row_base + 6
            out_mask = col_mask & (out_row < packed_rows)
            tl.store(
                output_ptr + out_row * output_row_stride + col_offsets,
                out6,
                mask=out_mask,
            )
        if num_bits >= 8:
            out_row = out_row_base + 7
            out_mask = col_mask & (out_row < packed_rows)
            tl.store(
                output_ptr + out_row * output_row_stride + col_offsets,
                out7,
                mask=out_mask,
            )


def _pack_col_parallel_triton(
    value: torch.Tensor,
    num_bits: int,
) -> torch.Tensor:
    """
    Triton implementation for packing along dim 0 without transpose.

    :param value: Input tensor of shape (rows, cols), dtype int32, unsigned values
    :param num_bits: Number of bits per element (1-8)
    :return: Packed tensor of shape (packed_rows, cols), dtype int32
    """
    rows, cols = value.shape
    packed_rows = math.ceil(rows * num_bits / 32)
    padded_rows = math.ceil(rows / 32) * 32

    output = torch.zeros(packed_rows, cols, dtype=torch.int32, device=value.device)

    BLOCK_SIZE_COL = 32
    grid = (triton.cdiv(cols, BLOCK_SIZE_COL),)

    _pack_to_int32_col_parallel_kernel[grid](
        value,
        output,
        rows,
        cols,
        packed_rows,
        padded_rows,
        value.stride(0),  # input_row_stride
        output.stride(0),  # output_row_stride
        num_bits,
        BLOCK_SIZE_COL,
    )

    return output


def _pack_col_parallel_triton_req(value: torch.Tensor, num_bits: int) -> bool:
    """Check if Triton implementation can be used."""
    return triton_req(value)


@ImplBackend.register("_pack_col_parallel", _pack_col_parallel_triton_req, 0)
def _pack_col_parallel_triton_impl(value: torch.Tensor, num_bits: int) -> torch.Tensor:
    """ImplBackend-registered Triton implementation."""
    return _pack_col_parallel_triton(value.contiguous(), num_bits)


def _pack_grouped(value: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Pack columns using row-parallel grouped packing.

    Pads input to multiple of 32, reshapes into 32-element groups,
    calls _pack_row_parallel, then reshapes back.

    :param value: Input tensor of shape (rows, cols), dtype int32, unsigned values
    :param num_bits: Number of bits per element (1-8)
    :return: Packed tensor of shape (rows, packed_cols), dtype int32
    """
    rows, cols = value.shape
    packed_cols = math.ceil(cols * num_bits / 32)

    padded_cols = math.ceil(cols / 32) * 32
    if padded_cols > cols:
        value = torch.nn.functional.pad(value, (0, padded_cols - cols))

    num_groups = padded_cols // 32
    rows_g = rows * num_groups
    value_g = value.reshape(rows_g, 32).contiguous()

    output_g = _pack_row_parallel(value_g, num_bits)

    return output_g.view(rows, num_groups * num_bits)[:, :packed_cols]


@ImplBackend.entrypoint("_pack_col_parallel")
def _pack_col_parallel(value: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Pack along dim 0 with automatic backend dispatch.

    Uses col-parallel Triton kernel on CUDA/XPU, falls back to
    transpose + row-parallel grouped packing on CPU.

    :param value: Input tensor of shape (rows, cols), dtype int32, unsigned values
    :param num_bits: Number of bits per element (1-8)
    :return: Packed tensor of shape (packed_rows, cols), dtype int32
    """
    # PyTorch fallback: transpose, grouped pack, transpose back
    return _pack_grouped(value.transpose(0, 1), num_bits).transpose(0, 1)


def pack_to_int32_accelerated(
    value: torch.Tensor,
    num_bits: int,
    packed_dim: Literal[0, 1] = 1,
) -> torch.Tensor:
    """
    Accelerated version of pack_to_int32 using Triton when available.

    Packs a tensor of intB (B=num_bits) quantized weights (stored in int8) into int32s.
    This packing is dense, with no padding bits, where necessary elements are split
    across int32 boundaries. For E elements of intB, we need E*B total bits, which means
    ceil(E*B/32) int32s when packed.

    Uses Triton kernel for the core bit-packing operation on CUDA/XPU devices,
    falls back to PyTorch scatter_add on CPU.

    For packed_dim=0, uses the col-parallel Triton kernel (no transpose).
    For packed_dim=1, uses the row-parallel Triton kernel (grouped packing).

    :param value: tensor to pack (must be torch.int8)
    :param num_bits: number of bits per element, must be in [1, 8]
    :param packed_dim: dimension to pack along (0 or 1)
    :returns: packed int32 tensor
    """
    if value.dtype is not torch.int8:
        raise ValueError("Tensor must be quantized to torch.int8 before packing")

    if not 1 <= num_bits <= 8:
        raise ValueError(
            f"Packing is only supported for num_bits in [1, 8], got {num_bits}"
        )

    # Handle N-dimensional tensors (e.g. MoE 3D weights) by packing each 2D slice
    if value.ndim > 2:
        return torch.stack(
            [
                pack_to_int32_accelerated(value[i], num_bits, packed_dim)
                for i in range(value.shape[0])
            ]
        )

    # Convert to unsigned range for packing
    offset = 1 << (num_bits - 1)
    value = value.to(torch.int32) + offset

    if packed_dim == 0:
        return _pack_col_parallel(value, num_bits)

    return _pack_grouped(value, num_bits)


def pack_to_int32(
    value: torch.Tensor,
    num_bits: int,
    packed_dim: Literal[0, 1] = 1,
) -> torch.Tensor:
    """
    Packs a tensor of intB (B=num_bits) quantized weights (stored in int8) into int32s.
    This packing is dense, with no padding bits, where necessary elements are split
    across int32 boundaries. For E elements of intB, we need E*B total bits, which means
    ceil(E*B/32) int32s when packed.

    :param value: tensor to pack (must be torch.int8)
    :param num_bits: number of bits per element, must be in [1, 8]
    :param packed_dim: dimension to pack along (0 or 1)
    :returns: packed int32 tensor
    """
    if value.dtype is not torch.int8:
        raise ValueError("Tensor must be quantized to torch.int8 before packing")

    if not 1 <= num_bits <= 8:
        raise ValueError(
            f"Packing is only supported for num_bits in [1, 8], got {num_bits}"
        )

    # Handle N-dimensional tensors (e.g. MoE 3D weights) by packing each 2D slice
    if value.ndim > 2:
        return torch.stack(
            [
                pack_to_int32(value[i], num_bits, packed_dim)
                for i in range(value.shape[0])
            ]
        )

    # Convert to unsigned range for packing, matching quantization offset
    offset = 1 << (num_bits - 1)
    value = value.to(torch.int32) + offset
    device = value.device

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, cols = value.shape
    packed_cols = math.ceil(cols * num_bits / 32)

    # Pad to a multiple of 32 so we can reshape into groups
    padded_cols = math.ceil(cols / 32) * 32
    if padded_cols > cols:
        value = torch.nn.functional.pad(value, (0, padded_cols - cols))

    num_groups = padded_cols // 32
    rows_g = rows * num_groups
    value_g = value.reshape(rows_g, 32)
    output_g = torch.zeros(rows_g, num_bits, dtype=torch.int32, device=device)

    elem_i = torch.arange(32, device=device, dtype=torch.int32)
    bit_starts = elem_i * num_bits
    word_idx = (bit_starts // 32).long()
    bit_offset = bit_starts % 32

    output_g.scatter_add_(
        1,
        word_idx.unsqueeze(0).expand(rows_g, -1),
        value_g << bit_offset.unsqueeze(0),
    )

    ov = bit_offset + num_bits - 32
    ov_mask = ov > 0
    if ov_mask.any():
        ov_vals = value_g[:, ov_mask] >> (num_bits - ov[ov_mask]).unsqueeze(0)
        output_g.scatter_add_(
            1,
            (word_idx[ov_mask] + 1).unsqueeze(0).expand(rows_g, -1),
            ov_vals,
        )

    # Truncate to minimum number of int32 words needed
    output = output_g.view(rows, num_groups * num_bits)[:, :packed_cols]

    if packed_dim == 0:
        output = output.transpose(0, 1)

    return output


def unpack_from_int32(
    value: torch.Tensor,
    num_bits: int,
    shape: torch.Size,
    packed_dim: Literal[0, 1] = 1,
) -> torch.Tensor:
    """
    Unpacks a tensor of densely packed int32 weights back to individual int8 values.

    Reverses pack_to_int32: element i is extracted from global bit position
    i*num_bits.

    :param value: packed int32 tensor to unpack
    :param num_bits: number of bits per element, must be in [1, 8]
    :param shape: original (pre-pack) shape, used to determine element count
    :param packed_dim: dimension that was packed (0 or 1)
    :returns: unpacked int8 tensor
    """
    if value.dtype is not torch.int32:
        raise ValueError(
            f"Expected {torch.int32} but got {value.dtype}, Aborting unpack."
        )

    if not 1 <= num_bits <= 8:
        raise ValueError(
            f"Unpacking is only supported for num_bits in [1, 8], got {num_bits}"
        )

    if value.ndim > 2:
        return torch.stack(
            [
                unpack_from_int32(value[i], num_bits, shape[1:], packed_dim)
                for i in range(value.shape[0])
            ]
        )

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, num_words = value.shape
    cols = int(shape[packed_dim])

    # Pad to a multiple of num_bits words so we can reshape into groups
    if num_words % num_bits != 0:
        pad_words = num_bits - (num_words % num_bits)
        value = torch.nn.functional.pad(value, (0, pad_words))
        num_words += pad_words

    num_groups = num_words // num_bits
    rows_g = rows * num_groups
    value_g = value.reshape(rows_g, num_bits)

    elem_i = torch.arange(32, device=value.device, dtype=torch.int32)
    bit_starts = elem_i * num_bits
    word_idx = (bit_starts // 32).long()
    bit_offset = bit_starts % 32
    lo_bits = torch.clamp(32 - bit_offset, max=num_bits)

    output_g = (value_g[:, word_idx] >> bit_offset.unsqueeze(0)) & (
        (1 << lo_bits) - 1
    ).unsqueeze(0)

    ov_mask = lo_bits < num_bits
    hi_bits = num_bits - lo_bits[ov_mask]
    right = (
        value_g[:, word_idx[ov_mask] + 1] & ((1 << hi_bits) - 1).unsqueeze(0)
    ) << lo_bits[ov_mask].unsqueeze(0)
    output_g[:, ov_mask] |= right

    # Unpad to original cols and reshape
    output = output_g.view(rows, num_groups * 32)[:, :cols]

    if packed_dim == 0:
        output = output.transpose(0, 1)

    offset = 1 << (num_bits - 1)
    return (output - offset).to(torch.int8)
