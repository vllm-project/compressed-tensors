# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import (
    pack_fp4_to_uint8,
    quantize_and_pack_fp4,
    unpack_fp4_from_uint8,
)
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization import QuantizationArgs, QuantizationType


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pack_unpack(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    x = torch.Tensor(
        [
            [-0.5000, -6.0000, -0.5000, -1.5000, -1.0000, 6.0000, 0.0000, -0.0000],
            [-1.0000, -6.0000, -0.5000, -0.0000, 0.5000, 0.5000, -0.0000, 0.0000],
            [-3.0000, -6.0000, -0.5000, -2.0000, -0.5000, -1.5000, -0.0000, -0.0000],
            [1.5000, 6.0000, -0.0000, -0.5000, 1.0000, 1.0000, -0.0000, 0.0000],
        ]
    )

    dense_dtype = torch.bfloat16
    x = x.to(dtype=dense_dtype, device=device)
    m, n = x.shape
    packed = pack_fp4_to_uint8(x.clone())  # clone to avoid mutation
    assert packed.dtype == torch.uint8
    unpacked = unpack_fp4_from_uint8(packed, m, n, dtype=dense_dtype)
    assert unpacked.dtype == dense_dtype

    assert torch.equal(unpacked, x)  # misleading as -0 and 0 are considered equal
    sign_bitx = torch.signbit(x)
    sign_bitout = torch.signbit(unpacked)
    # For nonzero values, sign bits must match exactly
    nonzero_mask = x != 0
    assert torch.equal(sign_bitout[nonzero_mask], sign_bitx[nonzero_mask])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pack_unpack_odd_dims(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    x = torch.Tensor(
        [
            [-0.5000, -6.0000, -0.5000, -1.5000, -1.0000, 6.0000, 0.0000],
            [-1.0000, -6.0000, -0.5000, -0.0000, 0.5000, 0.5000, -0.0000],
            [1.5000, 6.0000, -0.0000, -0.5000, 1.0000, 1.0000, -0.0000],
        ]
    ).to(device)

    with pytest.raises((ValueError, torch._dynamo.exc.Unsupported)):
        _ = pack_fp4_to_uint8(x)


def test_compress_scale_without_scale_dtype():
    """
    Test that NVFP4 compressor handles missing scale_dtype.

    (backward compatibility)
    """
    # Create a scale tensor
    scale = torch.randn(10, dtype=torch.bfloat16)

    # Create QuantizationArgs without scale_dtype (as in older models)
    quant_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        symmetric=True,
        group_size=16,
        # scale_dtype is not set (defaults to None)
    )

    # This should not raise an error and should default to float8_e4m3fn
    compressed_scale = NVFP4PackedCompressor._compress_scale(scale, quant_args)

    # Verify the output dtype is float8_e4m3fn
    assert compressed_scale.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize(
    "m,n",
    [
        (4, 16),
        (32, 64),
        (64, 128),
        (256, 512),
        (1024, 1024),
    ],
)
def test_quantize_and_pack_fused(m, n):
    """Test that fused quantize+pack produces identical results to separate ops."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    group_size = 16

    x = torch.randn(m, n, dtype=torch.bfloat16, device=device)
    scale = torch.rand(m, n // group_size, dtype=torch.bfloat16, device=device) + 0.1
    global_scale = torch.tensor(1.0, device=device)

    args = QuantizationArgs(
        num_bits=4, type=QuantizationType.FLOAT, group_size=group_size, symmetric=True
    )

    # Separate approach: quantize then pack
    quantized = quantize(
        x=x.clone(),
        scale=scale,
        global_scale=global_scale,
        zero_point=None,
        args=args,
    )
    packed_separate = pack_fp4_to_uint8(quantized)

    # Fused approach: single kernel
    packed_fused = quantize_and_pack_fp4(
        x=x.clone(),
        scale=scale,
        global_scale=global_scale,
        group_size=group_size,
    )

    assert torch.equal(packed_separate, packed_fused)


def test_quantize_and_pack_fused_boundary_values():
    """Test fused kernel handles FP4 boundary values correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    group_size = 16

    # Test exact boundary values for FP4 rounding thresholds
    x = torch.tensor(
        [[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0,
          -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, -6.0]],
        dtype=torch.bfloat16,
        device=device,
    )
    scale = torch.ones(1, 1, dtype=torch.bfloat16, device=device)
    global_scale = torch.tensor(1.0, device=device)

    args = QuantizationArgs(
        num_bits=4, type=QuantizationType.FLOAT, group_size=group_size, symmetric=True
    )

    quantized = quantize(
        x=x.clone(), scale=scale, global_scale=global_scale, zero_point=None, args=args
    )
    packed_separate = pack_fp4_to_uint8(quantized)

    packed_fused = quantize_and_pack_fp4(
        x=x.clone(),
        scale=scale,
        global_scale=global_scale,
        group_size=group_size,
    )

    assert torch.equal(packed_separate, packed_fused)


def test_quantize_and_pack_fused_with_zero_point():
    """Test fused kernel with asymmetric quantization (zero_point)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    m, n = 64, 128
    group_size = 16

    x = torch.randn(m, n, dtype=torch.bfloat16, device=device)
    scale = torch.rand(m, n // group_size, dtype=torch.bfloat16, device=device) + 0.1
    zero_point = torch.randn(m, n // group_size, dtype=torch.bfloat16, device=device) * 0.5
    global_scale = torch.tensor(1.0, device=device)

    args = QuantizationArgs(
        num_bits=4, type=QuantizationType.FLOAT, group_size=group_size, symmetric=False
    )

    quantized = quantize(
        x=x.clone(),
        scale=scale,
        global_scale=global_scale,
        zero_point=zero_point,
        args=args,
    )
    packed_separate = pack_fp4_to_uint8(quantized)

    packed_fused = quantize_and_pack_fp4(
        x=x.clone(),
        scale=scale,
        global_scale=global_scale,
        zero_point=zero_point,
        group_size=group_size,
    )

    assert torch.equal(packed_separate, packed_fused)
