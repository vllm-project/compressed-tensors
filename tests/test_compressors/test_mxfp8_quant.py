# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.mxfp8 import MXFP8QuantizationCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
)
from compressed_tensors.quantization.utils.helpers import calculate_qparams


def test_mxfp8_compress_decompress():
    """
    Test MXFP8 compress/decompress round-trip with group strategy
    and group_size=32. Verifies weights survive the cycle (lossy but close).
    """
    rows, cols = 512, 1024
    group_size = 32
    num_groups = cols // group_size

    quant_args = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
        symmetric=True,
    )

    weight = torch.randn((rows, cols))

    # Compute scales using calculate_qparams (which generates MX scales)
    reshaped = weight.reshape(rows, num_groups, group_size)
    min_vals = reshaped.amin(dim=-1)
    max_vals = reshaped.amax(dim=-1)
    scale, zp = calculate_qparams(min_vals, max_vals, quant_args)

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=quant_args,
    )

    # Build per-module state dict (local names, no module prefix)
    module_sd = {
        "weight": weight,
        "weight_scale": scale,
        "weight_zero_point": zp,
    }

    # Compress
    compressed = MXFP8QuantizationCompressor.compress(module_sd, scheme=scheme)

    # Check compressed weight is FP8
    assert compressed["weight"].dtype == torch.float8_e4m3fn

    # Check scale is stored as uint8 (E8M0 exponent format)
    assert compressed["weight_scale"].dtype == torch.uint8

    # Decompress
    decompressed = MXFP8QuantizationCompressor.decompress(compressed, scheme=scheme)

    # Check shapes match
    assert decompressed["weight"].shape == weight.shape

    # FP8 quantization is lossy, but should be reasonably close
    assert torch.allclose(decompressed["weight"].float(), weight, atol=0.1, rtol=0.1)


def test_mxfp8_scale_roundtrip():
    """
    Test that E8M0 scale encoding/decoding is lossless for power-of-2 scales.
    """
    rows, cols = 128, 256
    group_size = 32
    num_groups = cols // group_size

    quant_args = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
        symmetric=True,
    )

    weight = torch.randn((rows, cols))

    reshaped = weight.reshape(rows, num_groups, group_size)
    min_vals = reshaped.amin(dim=-1)
    max_vals = reshaped.amax(dim=-1)
    scale, zp = calculate_qparams(min_vals, max_vals, quant_args)

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=quant_args,
    )

    module_sd = {
        "weight": weight,
        "weight_scale": scale,
        "weight_zero_point": zp,
    }

    compressed = MXFP8QuantizationCompressor.compress(module_sd, scheme=scheme)

    # E8M0 encoded scale
    e8m0_scale = compressed["weight_scale"]
    assert e8m0_scale.dtype == torch.uint8

    # Decode: 2^(exp - 127)
    scale_exp = e8m0_scale.to(torch.int32) - 127
    decoded_scale = 2.0 ** scale_exp.to(torch.float32)

    # The original scale after floor(log2) should round-trip exactly
    expected_scale = 2.0 ** torch.floor(torch.log2(scale)).to(torch.float32)
    assert torch.allclose(decoded_scale, expected_scale)


def test_mxfp8_can_compress():
    """Test that can_compress matches MXFP8 signature correctly."""
    import torch.nn as nn

    mxfp8_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type="float",
            strategy=QuantizationStrategy.GROUP,
            group_size=32,
            scale_dtype=torch.uint8,
        ),
    )
    assert MXFP8QuantizationCompressor.can_compress(nn.Linear, mxfp8_scheme) is True

    # Non-MXFP8: group_size != 32
    non_mxfp8_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type="float",
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
        ),
    )
    assert (
        MXFP8QuantizationCompressor.can_compress(nn.Linear, non_mxfp8_scheme) is False
    )

    # Non-MXFP8: int type
    int_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type="int",
            strategy=QuantizationStrategy.GROUP,
            group_size=32,
            scale_dtype=torch.uint8,
        ),
    )
    assert MXFP8QuantizationCompressor.can_compress(nn.Linear, int_scheme) is False


def test_compress_scale_without_scale_dtype():
    """
    Test that MXFP8 compressor handles missing scale_dtype.

    (backward compatibility)
    """
    # Create a scale tensor
    scale = torch.randn(10, dtype=torch.bfloat16).abs() + 1e-6  # Ensure positive values

    # Create QuantizationArgs without scale_dtype (as in older models)
    quant_args = QuantizationArgs(
        num_bits=8,
        type="float",
        symmetric=True,
        group_size=32,
        # scale_dtype is not set (defaults to None)
    )

    # This should not raise an error and should default to uint8
    compressed_scale = MXFP8QuantizationCompressor._compress_scale(scale, quant_args)

    # Verify the output dtype is uint8
    assert compressed_scale.dtype == torch.uint8
