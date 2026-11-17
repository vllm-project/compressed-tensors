# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unsigned E5M3 (UE5M3) block-scale codec.

UE5M3 is an 8-bit unsigned floating-point format with five exponent bits and
three mantissa bits. The encoding uses a bias of 15, reserves ``0xFF`` for NaN,
and supports subnormals down to ``2**-17``. Its largest finite value is
``114688.0`` (``0xFE``).
"""

import torch


__all__ = [
    "UE5M3_MAX",
    "UE5M3_MIN_SUBNORMAL",
    "UE5M3_NAN_CODE",
    "ue5m3_decode",
    "ue5m3_encode",
    "ue5m3_cast",
    "encode",
    "decode",
    "cast",
]


UE5M3_EXPONENT_BITS = 5
UE5M3_MANTISSA_BITS = 3
UE5M3_EXPONENT_BIAS = 15
UE5M3_MIN_SUBNORMAL = 2.0**-17
UE5M3_MAX = 114688.0
UE5M3_NAN_CODE = 0xFF


def _decode_codes(codes: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    codes_int = codes.to(torch.int16)
    exponent = (codes_int >> UE5M3_MANTISSA_BITS) & 0x1F
    mantissa = (codes_int & 0x7).to(torch.float64)

    normal = torch.ldexp(
        1.0 + mantissa / 8.0,
        (exponent - UE5M3_EXPONENT_BIAS).to(torch.int32),
    )
    subnormal = torch.ldexp(
        mantissa / 8.0,
        torch.full_like(exponent, 1 - UE5M3_EXPONENT_BIAS, dtype=torch.int32),
    )
    values = torch.where(exponent == 0, subnormal, normal)
    values = values.masked_fill(codes_int == UE5M3_NAN_CODE, torch.nan)
    return values.to(dtype)


def ue5m3_decode(
    codes: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Decode UE5M3 byte patterns into floating-point values.

    ``0xFF`` decodes to NaN. All other byte patterns decode to finite,
    non-negative values.

    :param codes: uint8 tensor containing UE5M3 byte patterns
    :param dtype: floating-point dtype for the decoded tensor
    :return: decoded values
    """
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        raise TypeError(f"UE5M3 decode requires a floating-point dtype, got {dtype}")
    if codes.device.type == "meta":
        return torch.empty(codes.shape, dtype=dtype, device=codes.device)
    return _decode_codes(codes, dtype)


def _finite_decode_table(device: torch.device) -> torch.Tensor:
    codes = torch.arange(UE5M3_NAN_CODE, dtype=torch.uint8, device=device)
    return _decode_codes(codes, torch.float64)


def ue5m3_encode(values: torch.Tensor) -> torch.Tensor:
    """
    Encode non-negative finite values as UE5M3 byte patterns using RNE.

    Values above the largest finite UE5M3 value saturate to ``0xFE``. Negative
    values, NaN, and infinity are rejected.

    :param values: tensor of non-negative finite values
    :return: uint8 tensor containing UE5M3 byte patterns
    """
    if values.is_complex():
        raise TypeError("UE5M3 encode does not support complex tensors")
    if values.device.type == "meta":
        return torch.empty(values.shape, dtype=torch.uint8, device=values.device)

    values = values.to(torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("UE5M3 encode requires finite values")
    if (values < 0).any():
        raise ValueError("UE5M3 encode requires non-negative values")

    table = _finite_decode_table(values.device)
    values = values.clamp(0.0, UE5M3_MAX)

    upper = torch.searchsorted(table, values, right=False).clamp_max(UE5M3_NAN_CODE - 1)
    lower = (upper - 1).clamp_min(0)

    lower_value = table[lower]
    upper_value = table[upper]
    lower_distance = values - lower_value
    upper_distance = upper_value - values

    choose_upper = upper_distance < lower_distance
    midpoint = upper_distance == lower_distance
    choose_upper = choose_upper | (midpoint & (lower.remainder(2) == 1))

    codes = torch.where(choose_upper, upper, lower)
    return codes.to(torch.uint8)


def ue5m3_cast(values: torch.Tensor) -> torch.Tensor:
    """
    Cast values to the nearest UE5M3 value while preserving shape and dtype.

    :param values: tensor of non-negative finite values
    :return: values rounded to UE5M3 with the original floating-point dtype
    """
    dtype = values.dtype if values.is_floating_point() else torch.float32
    return ue5m3_decode(ue5m3_encode(values), dtype=dtype)


encode = ue5m3_encode
decode = ue5m3_decode
cast = ue5m3_cast
