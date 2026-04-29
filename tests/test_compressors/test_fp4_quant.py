# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import (
    pack_fp4_to_uint8,
    unpack_fp4_from_uint8,
)
from compressed_tensors.quantization import QuantizationArgs, QuantizationType


def test_download_model():
    MODEL_ID = "RedHatAI/gemma-3-12b-it-FP8-dynamic"
    print("DOWNLOADING MODEL ", MODEL_ID)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM(MODEL_ID)
    print("GOT MODEL ", model)


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
