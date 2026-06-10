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
    Packs a tensor of quantized weights stored in int8 into int32s using
    dense cross-element bit packing.

    32 consecutive elements are packed into exactly num_bits int32 words,
    with element i placed at bit position i*num_bits (spanning word
    boundaries when necessary). For power-of-2 bit widths this produces
    the same result as element-aligned packing when the packed dimension
    is a multiple of 32.

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
    # Pad to next multiple of 32 (one full packing group)
    padded_cols = math.ceil(cols / 32) * 32
    pad_len = padded_cols - cols
    if pad_len > 0:
        value = torch.nn.functional.pad(value, (0, pad_len))

    num_groups = padded_cols // 32
    rows_g = rows * num_groups
    value_g = value.reshape(rows_g, 32)
    output_g = torch.zeros(rows_g, num_bits, dtype=torch.int32, device=device)

    elem_i = torch.arange(32, device=device, dtype=torch.int32)
    bit_starts = elem_i * num_bits
    word_idx = (bit_starts // 32).long()
    bit_offset = bit_starts % 32

    # Place element i at bit position i*num_bits; addition == OR for non-overlapping fields
    output_g.scatter_add_(1, word_idx.unsqueeze(0).expand(rows_g, -1), value_g << bit_offset.unsqueeze(0))

    ov = bit_offset + num_bits - 32
    ov_mask = ov > 0
    if ov_mask.any():
        ov_vals = value_g[:, ov_mask] >> (num_bits - ov[ov_mask]).unsqueeze(0)
        output_g.scatter_add_(1, (word_idx[ov_mask] + 1).unsqueeze(0).expand(rows_g, -1), ov_vals)

    output = output_g.view(rows, num_groups * num_bits)

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

    Reverses pack_to_int32: each group of num_bits int32 words is expanded
    into 32 elements, with element i extracted from bit position i*num_bits.

    :param value: packed int32 tensor to unpack
    :param num_bits: number of bits per element, must be in [1, 8]
    :param shape: original (pre-pack) shape, used to remove padding
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

    # Handle N-dimensional tensors (e.g. MoE 3D weights) by unpacking each 2D slice
    if value.ndim > 2:
        return torch.stack(
            [
                unpack_from_int32(value[i], num_bits, shape[1:], packed_dim)
                for i in range(value.shape[0])
            ]
        )

    mask = (1 << num_bits) - 1

    if packed_dim == 0:
        value = value.transpose(0, 1)

    # handles tensors packed with old code
    rows, num_words = value.shape
    if num_words % num_bits != 0:
        pad_words = num_bits - (num_words % num_bits)
        value = torch.nn.functional.pad(value, (0, pad_words))
        num_words += pad_words
    
    num_groups = num_words // num_bits

    rows_g = rows * num_groups
    value_g = value.reshape(rows_g, num_bits)
    output_g = torch.zeros(rows_g, 32, dtype=torch.int32, device=value.device)

    elem_i = torch.arange(32, device=value.device, dtype=torch.int32)
    bit_starts = elem_i * num_bits
    word_idx = (bit_starts // 32).long()
    bit_offset = bit_starts % 32
    ov = bit_offset + num_bits - 32

    # Non-overflow elements: extract directly from their single word
    no_ov = ov <= 0
    output_g[:, elem_i[no_ov]] = (value_g[:, word_idx[no_ov]] >> bit_offset[no_ov].unsqueeze(0)) & mask

    # Overflow elements: stitch bits from two consecutive words
    ov_mask = ~no_ov
    if ov_mask.any():
        ov_word = word_idx[ov_mask]
        ov_off = bit_offset[ov_mask]
        ov_ov = ov[ov_mask]
        bif = num_bits - ov_ov
        first = (value_g[:, ov_word] >> ov_off.unsqueeze(0)) & ((1 << bif) - 1).unsqueeze(0)
        second = (value_g[:, ov_word + 1] & ((1 << ov_ov) - 1).unsqueeze(0)) << bif.unsqueeze(0)
        output_g[:, elem_i[ov_mask]] = first | second

    # Remove padding and restore original shape
    output = output_g.view(rows, num_groups * 32)[:, : int(shape[packed_dim])]

    if packed_dim == 0:
        output = output.transpose(0, 1)

    # Convert from unsigned back to signed
    offset = 1 << (num_bits - 1)
    return (output - offset).to(torch.int8)
