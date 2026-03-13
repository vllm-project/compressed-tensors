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
    Packs a tensor of quantized weights stored in int8 into int32s with padding

    Pseudocode:
     1. Shift wrt num_bits to convert to unsigned. num_bits=8
        [1,2] -> [129, 130]
     2. Pad to fill in 32 bits
        [129, 130] -> [129, 130, 0, 0]
     3. convert to binary align in order
        [129, 130, 0, 0] -> 00000000 00000000 10000010 10000001
     4. convert aligned binary to number
        00000000000000001000001010000001 -> 33409
     5. covert back to uint32
        33409 -> 33409

    :param value: tensor to pack
    :param num_bits: number of bits used to store underlying data, must be at least 1
    :returns: packed int32 tensor
    """
    if value.dtype is not torch.int8:
        raise ValueError("Tensor must be quantized to torch.int8 before packing")

    if num_bits > 8:
        raise ValueError("Packing is only supported for less than 8 bits")

    if num_bits < 1:
        raise ValueError(f"num_bits must be at least 1, got {num_bits}")

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
    value = (value + offset).to(torch.uint8)
    device = value.device

    pack_factor = 32 // num_bits

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, cols = value.shape
    padded_cols = math.ceil(cols / pack_factor) * pack_factor
    pad_len = padded_cols - cols

    if pad_len > 0:
        value = torch.nn.functional.pad(value, (0, pad_len))

    num_groups = padded_cols // pack_factor

    # Use int32 here
    reshaped = value.view(rows, num_groups, pack_factor).to(torch.int32)
    bit_shifts = torch.arange(pack_factor, device=device, dtype=torch.int32) * num_bits
    packed = (reshaped << bit_shifts).sum(dim=2, dtype=torch.int32)

    if packed_dim == 0:
        packed = packed.transpose(0, 1)

    return packed


def unpack_from_int32(
    value: torch.Tensor,
    num_bits: int,
    shape: torch.Size,
    packed_dim: Literal[0, 1] = 1,
) -> torch.Tensor:
    """
    Unpacks a tensor of packed int32 weights into individual int8s, maintaining the
    original bit range.

    Return tensors in int8

    :param value: tensor to upack
    :param num_bits: number of bits to unpack each data point into
    :param shape: shape to unpack into, used to remove padding
    :returns: unpacked int8 tensor
    """
    if value.dtype is not torch.int32:
        raise ValueError(
            f"Expected {torch.int32} but got {value.dtype}, Aborting unpack."
        )

    if num_bits > 8:
        raise ValueError("Unpacking is only supported for less than 8 bits")

    # Handle N-dimensional tensors (e.g. MoE 3D weights) by unpacking each 2D slice
    if value.ndim > 2:
        return torch.stack(
            [
                unpack_from_int32(value[i], num_bits, shape[1:], packed_dim)
                for i in range(value.shape[0])
            ]
        )

    pack_factor = 32 // num_bits

    # unpack
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked = torch.zeros(
            (value.shape[0], value.shape[1] * pack_factor),
            device=value.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask

        # remove padding
        original_row_size = int(shape[1])
        unpacked = unpacked[:, :original_row_size]
    else:
        unpacked = torch.zeros(
            (value.shape[0] * pack_factor, value.shape[1]),
            device=value.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked[i::pack_factor, :] = (value >> (num_bits * i)) & mask

        # remove padding
        original_row_size = int(shape[0])
        unpacked = unpacked[:original_row_size, :]

    # bits are packed in unsigned format, reformat to signed
    # update the value range from unsigned to signed
    offset = pow(2, num_bits) // 2
    unpacked = (unpacked - offset).to(torch.int8)

    return unpacked
