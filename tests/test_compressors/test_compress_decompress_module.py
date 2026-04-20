# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn
from compressed_tensors.compressors.base import compress_module, decompress_module
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    ActivationOrdering,
    initialize_module_for_quantization,
    preset_name_to_scheme,
)
from compressed_tensors.utils import get_direct_state_dict


def _run_compress_decompress(scheme_name, expected_format, actorder, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # 1. Initialize module for quantization using a preset scheme.
    # Use 256x256 to avoid degenerate scale shapes (e.g. (N,1) for FP8_BLOCK)
    # that trip up block-vs-channel strategy inference in dequantize.
    # Start in bfloat16 so NVFP4 decompression (which returns bfloat16) preserves dtype.
    module = nn.Linear(256, 256, bias=False).to(dtype=torch.bfloat16, device=device)
    scheme = preset_name_to_scheme(scheme_name, ["Linear"])
    if actorder is not None:
        scheme.weights.actorder = actorder
    initialize_module_for_quantization(module, scheme)

    with torch.no_grad():
        for name, param in list(module.named_parameters()):
            param.fill_(1)

    # Record pre-compression state dict shapes and dtypes.
    # Filter out None entries (e.g. bias=None when bias=False).
    pre_state = {
        name: (tensor.shape, tensor.dtype)
        for name, tensor in get_direct_state_dict(module).items()
        if tensor is not None
    }

    # 2. Compress the module and verify the inferred quantization format.
    compress_module(module)
    assert module.quantization_scheme.format == expected_format

    # 3. Decompress the module and verify shapes and dtypes are restored.
    decompress_module(module)

    post_state_dict = get_direct_state_dict(module)
    for name, tensor in post_state_dict.items():
        if name in pre_state:
            pre_shape, pre_dtype = pre_state[name]
            assert tensor.shape == pre_shape
            assert tensor.dtype == pre_dtype


@pytest.mark.parametrize(
    "scheme_name,expected_format,actorder",
    [
        ("UNQUANTIZED", CompressionFormat.dense, None),
        ("W8A16", CompressionFormat.pack_quantized, None),
        ("W4A16", CompressionFormat.pack_quantized, None),
        ("W4A16", CompressionFormat.pack_quantized, ActivationOrdering.GROUP),
        ("W4A16_ASYM", CompressionFormat.pack_quantized, None),
        ("W4A16_ASYM", CompressionFormat.pack_quantized, ActivationOrdering.GROUP),
        ("W8A8", CompressionFormat.int_quantized, None),
        ("W4A8", CompressionFormat.int_quantized, None),
        ("W4AFP8", CompressionFormat.int_quantized, None),
        ("FP8", CompressionFormat.float_quantized, None),
        ("FP8_DYNAMIC", CompressionFormat.float_quantized, None),
        ("FP8_BLOCK", CompressionFormat.float_quantized, None),
        ("NVFP4A16", CompressionFormat.nvfp4_pack_quantized, None),
        ("NVFP4", CompressionFormat.nvfp4_pack_quantized, None),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "meta", "cuda"])
def test_compress_decompress_module(scheme_name, expected_format, actorder, device):
    _run_compress_decompress(scheme_name, expected_format, actorder, device)


@pytest.mark.parametrize(
    "scheme_name,expected_format",
    [
        ("MXFP4A16", CompressionFormat.mxfp4_pack_quantized),
        ("MXFP4", CompressionFormat.mxfp4_pack_quantized),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "meta", "cuda"])
def test_compress_decompress_module_mxfp4(scheme_name, expected_format, device):
    _run_compress_decompress(scheme_name, expected_format, None, device)
