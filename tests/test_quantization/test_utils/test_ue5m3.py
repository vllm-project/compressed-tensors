# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
from compressed_tensors.quantization.utils.ue5m3 import (
    UE5M3_MAX,
    UE5M3_MIN_SUBNORMAL,
    UE5M3_NAN_CODE,
    decode,
    encode,
    ue5m3_cast,
)


def reference_decode(code: int) -> float:
    if code == UE5M3_NAN_CODE:
        return math.nan

    exponent = (code >> 3) & 0x1F
    mantissa = code & 0x7
    if exponent == 0:
        return math.ldexp(mantissa / 8.0, 1 - 15)
    return math.ldexp(1.0 + mantissa / 8.0, exponent - 15)


def test_decode_all_byte_patterns_matches_reference():
    codes = torch.arange(256, dtype=torch.uint8)
    actual = decode(codes)
    expected = torch.tensor(
        [reference_decode(code) for code in range(256)], dtype=torch.float32
    )

    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(actual[~torch.isnan(expected)], expected[~torch.isnan(expected)])


def test_known_values():
    codes = torch.tensor(
        [0x00, 0x01, 0x02, 0x78, 0x79, 0x80, 0xFD, 0xFE],
        dtype=torch.uint8,
    )
    actual = decode(codes)
    expected = torch.tensor(
        [
            0.0,
            UE5M3_MIN_SUBNORMAL,
            2 * UE5M3_MIN_SUBNORMAL,
            1.0,
            1.0 + 1 / 8,
            2.0,
            106496.0,
            UE5M3_MAX,
        ]
    )
    assert torch.equal(actual, expected)


def test_nan_pattern_is_preserved():
    assert torch.isnan(decode(torch.tensor([UE5M3_NAN_CODE], dtype=torch.uint8))).all()


def test_encode_decode_round_trip_all_finite_patterns():
    codes = torch.arange(UE5M3_NAN_CODE, dtype=torch.uint8)
    actual = encode(decode(codes, dtype=torch.float64))
    assert torch.equal(actual, codes)


def test_encode_uses_round_to_nearest_even_at_midpoints():
    values = decode(
        torch.arange(UE5M3_NAN_CODE, dtype=torch.uint8), dtype=torch.float64
    )
    midpoints = (values[:-1] + values[1:]) / 2
    expected = torch.arange(UE5M3_NAN_CODE - 1, dtype=torch.uint8)
    expected = torch.where(expected.remainder(2) == 0, expected, expected + 1)

    assert torch.equal(encode(midpoints), expected)


def test_encode_rounds_to_nearest_representable_value():
    values = decode(torch.tensor([0x00, 0x01, 0x7F], dtype=torch.uint8))
    lower = values
    upper = decode(torch.tensor([0x01, 0x02, 0x80], dtype=torch.uint8))
    near_lower = lower + 0.49 * (upper - lower)
    near_upper = lower + 0.51 * (upper - lower)

    assert torch.equal(encode(near_lower), torch.tensor([0x00, 0x01, 0x7F]))
    assert torch.equal(encode(near_upper), torch.tensor([0x01, 0x02, 0x80]))


def test_encode_saturates_above_max():
    values = torch.tensor([UE5M3_MAX, 2 * UE5M3_MAX, torch.inf])
    with pytest.raises(ValueError, match="finite"):
        encode(values)

    assert torch.equal(
        encode(torch.tensor([UE5M3_MAX, 2 * UE5M3_MAX])),
        torch.tensor([0xFE, 0xFE], dtype=torch.uint8),
    )


@pytest.mark.parametrize("value", [-1.0, torch.nan, torch.inf, -torch.inf])
def test_encode_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        encode(torch.tensor([value]))


def test_cast_preserves_shape_and_dtype():
    values = torch.tensor([[0.0, 0.1], [1.1, 100000.0]], dtype=torch.bfloat16)
    actual = ue5m3_cast(values)
    expected = decode(encode(values), dtype=torch.bfloat16)

    assert actual.shape == values.shape
    assert actual.dtype == values.dtype
    assert torch.equal(actual, expected)
