# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization import round_to_quantized_type_dtype
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils import (
    generate_mx_scales,
    maybe_convert_from_mx_exp,
    round_to_power_2,
    should_generate_mx_scales,
)


def test_should_generate_mx_scales_mxfp8():
    """Test that should_generate_mx_scales returns True for MXFP8 args."""
    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    )
    assert should_generate_mx_scales(args) is True


def test_should_generate_mx_scales_mxfp4():
    """Test that should_generate_mx_scales returns True for MXFP4 args."""
    args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    )
    assert should_generate_mx_scales(args) is True


def test_should_generate_mx_scales_regular_fp8():
    """Test that should_generate_mx_scales returns False for regular FP8."""
    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR,
    )
    assert should_generate_mx_scales(args) is False


def test_should_generate_mx_scales_wrong_group_size():
    """Test that should_generate_mx_scales returns False for non-32 group size."""
    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        group_size=128,
    )
    assert should_generate_mx_scales(args) is False


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mxfp8_scales_e2e(dtype):
    """End-to-end test for MXFP8 scale generation and conversion."""
    mock_weight = torch.normal(mean=0.0002, std=0.0576, size=(2880, 2880))

    x = mock_weight.reshape(*mock_weight.shape[:-1], -1, 32).to(dtype)
    min_vals = torch.amin(x, dim=-1)
    max_vals = torch.amax(x, dim=-1)

    min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
    max_vals = torch.max(max_vals, torch.zeros_like(max_vals))
    block_max = torch.max(torch.abs(min_vals), torch.abs(max_vals))

    args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    )

    scales = generate_mx_scales(block_max, num_bits=8)
    scales = round_to_quantized_type_dtype(scales, dtype=args.scale_dtype)

    converted_ct = maybe_convert_from_mx_exp(args=args, scale=scales)

    scales_exp = torch.log2(converted_ct)
    block_max_exp = torch.floor(torch.log2(round_to_power_2(block_max))) - 8
    assert torch.equal(scales_exp, block_max_exp)
