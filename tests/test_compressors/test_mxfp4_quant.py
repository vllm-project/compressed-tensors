# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationType,
)


def test_compress_scale_without_scale_dtype():
    """
    Test that MXFP4 compressor handles missing scale_dtype.

    (backward compatibility)
    """
    # Create a scale tensor
    scale = torch.randn(10, dtype=torch.bfloat16).abs() + 1e-6  # Ensure positive values

    # Create QuantizationArgs without scale_dtype (as in older models)
    quant_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        symmetric=True,
        group_size=32,
        # scale_dtype is not set (defaults to None)
    )

    # This should not raise an error and should default to uint8
    compressed_scale = MXFP4PackedCompressor._compress_scale(scale, quant_args)

    # Verify the output dtype is uint8
    assert compressed_scale.dtype == torch.uint8


def test_compress_scale_with_scale_dtype():
    """Test that MXFP4 compressor respects explicit scale_dtype"""
    # Create a scale tensor
    scale = torch.randn(10, dtype=torch.bfloat16).abs() + 1e-6  # Ensure positive values

    # Create QuantizationArgs with explicit scale_dtype
    quant_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        symmetric=True,
        group_size=32,
        scale_dtype=torch.uint8,
    )

    # Compress the scale
    compressed_scale = MXFP4PackedCompressor._compress_scale(scale, quant_args)

    # Verify the output dtype matches the specified scale_dtype
    assert compressed_scale.dtype == torch.uint8


def test_decompress_decodes_mx_scales_and_restores_weight():
    quant_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        symmetric=True,
        group_size=32,
        scale_dtype=torch.uint8,
    )
    # group_size=32 is the source of truth, so each scale group covers 32
    # weight columns. Use two groups (scales 0.25 and 0.5) spanning 64 columns.
    scale = torch.tensor([[0.25, 0.5]], dtype=torch.bfloat16)
    group_values = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.bfloat16)
    fp4_values = group_values.repeat(16).unsqueeze(0)  # (1, 64)
    packed = pack_fp4_to_uint8(fp4_values)

    decompressed = MXFP4PackedCompressor.decompress(
        {
            "weight_packed": packed,
            "weight_scale": MXFP4PackedCompressor._compress_scale(scale, quant_args),
        },
        QuantizationScheme(targets=["Linear"], weights=quant_args),
    )

    expected_group0 = group_values * 0.25  # first 32 columns
    expected_group1 = group_values * 0.5  # last 32 columns
    expected_weight = torch.cat(
        [expected_group0.repeat(8), expected_group1.repeat(8)]
    ).unsqueeze(0)

    assert torch.equal(decompressed["weight_scale"], scale)
    assert torch.equal(decompressed["weight"], expected_weight)
