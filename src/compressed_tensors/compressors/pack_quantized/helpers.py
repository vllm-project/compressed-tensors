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


__all__ = ["pack_to_int32", "unpack_from_int32"]


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

    # unpad to original cols and reshape
    output = output_g.view(rows, num_groups * 32)[:, :cols]

    if packed_dim == 0:
        output = output.transpose(0, 1)

    offset = 1 << (num_bits - 1)
    return (output - offset).to(torch.int8)
