# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Packing of FP6 E2M3 values: each value maps to its 6-bit code (sign in bit 5),
and four consecutive codes are packed little-endian into three bytes.
"""

import torch


__all__ = ["pack_fp6_to_uint8", "unpack_fp6_from_uint8"]


# magnitudes of the 32 non-negative E2M3 codes, in code order
kE2M3ToFloat = torch.tensor(
    [m / 8 for m in range(8)]
    + [(1 + m / 8) * 2**e for e in range(3) for m in range(8)]
)


def pack_fp6_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """
    Packs a [m, n] tensor of E2M3 values into a [m, 3 * n // 4] uint8 tensor.
    Assumes x contains only valid E2M3 values, e.g. the output of cast_to_fp6.
    """
    m, n = x.shape
    if n % 4 != 0:
        raise ValueError("tensor must have a multiple of 4 columns for fp6 packing")

    table = kE2M3ToFloat.to(x.device)
    codes = torch.searchsorted(table, x.abs().float()).to(torch.int32)
    codes |= torch.signbit(x).to(torch.int32) << 5

    codes = codes.reshape(m, n // 4, 4)
    word = (
        codes[..., 0] | codes[..., 1] << 6 | codes[..., 2] << 12 | codes[..., 3] << 18
    )
    packed = torch.stack([word, word >> 8, word >> 16], dim=-1) & 0xFF
    return packed.to(torch.uint8).reshape(m, 3 * n // 4)


def unpack_fp6_from_uint8(
    a: torch.Tensor, m: int, n: int, dtype: torch.dtype | None = torch.bfloat16
) -> torch.Tensor:
    """
    Unpacks a [m, 3 * n // 4] uint8 tensor into a [m, n] tensor of E2M3 values.
    """
    b = a.reshape(m, n // 4, 3).to(torch.int32)
    word = b[..., 0] | b[..., 1] << 8 | b[..., 2] << 16
    codes = torch.stack([word >> shift & 0x3F for shift in (0, 6, 12, 18)], dim=-1)

    values = kE2M3ToFloat.to(a.device)[codes & 0x1F]
    values = torch.where(codes & 0x20 != 0, -values, values)
    return values.reshape(m, n).to(dtype)
