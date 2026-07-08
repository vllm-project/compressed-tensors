# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Helper functions for packing and unpacking FP4 (E2M1) quantized weights.

FP4 E2M1 format uses 1 sign bit, 2 exponent bits, and 1 mantissa bit,
supporting 8 positive and 8 negative values. This module provides efficient
packing of two FP4 values into a single uint8 for storage.
"""

import torch


__all__ = ["pack_fp4_to_uint8", "unpack_fp4_from_uint8"]


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

    # NOTE: _cast_to_fp4 uses torch.sign() which returns 0 for zero, so it never
    # produces -0.0. All zeros are positive, so we don't need special -0.0 handling.

    # Convert to int8 to save memory (bf16 -> int8 is a 2x reduction)
    x.mul_(2)
    x = x.to(torch.int8)

    indices = torch.zeros_like(x, dtype=torch.uint8)

    indices[x == 1] = 1
    indices[x == 2] = 2
    indices[x == 3] = 3
    indices[x == 4] = 4
    indices[x == 6] = 5
    indices[x == 8] = 6
    indices[x >= 12] = 7

    indices[x == -1] = 9
    indices[x == -2] = 10
    indices[x == -3] = 11
    indices[x == -4] = 12
    indices[x == -6] = 13
    indices[x == -8] = 14
    indices[x <= -12] = 15

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
