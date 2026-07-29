# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import (
    pack_fp4_to_uint8,
    unpack_fp4_from_uint8,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationType,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize


def test_pack_unpack():
    x = torch.Tensor(
        [
            [-0.5000, -6.0000, -0.5000, -1.5000, -1.0000, 6.0000, 0.0000, -0.0000],
            [-1.0000, -6.0000, -0.5000, -0.0000, 0.5000, 0.5000, -0.0000, 0.0000],
            [-3.0000, -6.0000, -0.5000, -2.0000, -0.5000, -1.5000, -0.0000, -0.0000],
            [1.5000, 6.0000, -0.0000, -0.5000, 1.0000, 1.0000, -0.0000, 0.0000],
        ]
    )

    dense_dtype = torch.bfloat16
    x = x.to(dense_dtype)
    m, n = x.shape
    packed = pack_fp4_to_uint8(x)
    assert packed.dtype == torch.uint8
    unpacked = unpack_fp4_from_uint8(packed, m, n, dtype=dense_dtype)
    assert unpacked.dtype == dense_dtype

    assert torch.equal(unpacked, x)  # misleading as -0 and 0 are considered equal
    sign_bitx = torch.signbit(x)
    sign_bitout = torch.signbit(unpacked)
    assert torch.equal(sign_bitout, sign_bitx)


def test_pack_unpack_odd_dims():
    x = torch.Tensor(
        [
            [-0.5000, -6.0000, -0.5000, -1.5000, -1.0000, 6.0000, 0.0000],
            [-1.0000, -6.0000, -0.5000, -0.0000, 0.5000, 0.5000, -0.0000],
            [1.5000, 6.0000, -0.0000, -0.5000, 1.0000, 1.0000, -0.0000],
        ]
    )

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


def test_nvfp4_can_compress_tensor_block_scheme():
    scheme = preset_name_to_scheme("NVFP4A16_BLOCK", targets=["Linear"])

    assert NVFP4PackedCompressor.can_compress(torch.nn.Linear, scheme)


def test_nvfp4_rejects_non_16x16_tensor_block_scheme():
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            strategy="tensor_block",
            type=QuantizationType.FLOAT,
            num_bits=4,
            block_structure=[8, 16],
        ),
    )

    assert not NVFP4PackedCompressor.can_compress(torch.nn.Linear, scheme)


def test_nvfp4_tensor_block_decompress_uses_configured_block_shape():
    scheme = preset_name_to_scheme("NVFP4A16_BLOCK", targets=["Linear"])
    weight = torch.randn(16, 24, dtype=torch.bfloat16)
    scale = torch.ones(1, 2, dtype=torch.bfloat16)
    global_scale = torch.ones(1, dtype=torch.float32)
    state_dict = {
        "weight": weight,
        "weight_scale": scale,
        "weight_global_scale": global_scale,
    }

    compressed = NVFP4PackedCompressor.compress(state_dict, scheme)
    decompressed = NVFP4PackedCompressor.decompress(compressed, scheme)

    quantized_weight = quantize(
        x=weight,
        scale=scale,
        global_scale=global_scale,
        zero_point=None,
        args=scheme.weights,
    )
    expected = dequantize(
        x_q=quantized_weight,
        scale=scale,
        global_scale=global_scale,
        args=scheme.weights,
        dtype=quantized_weight.dtype,
    )

    assert torch.equal(decompressed["weight"], expected)
