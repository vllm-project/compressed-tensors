# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.compressors.base import compress_module, decompress_module
from compressed_tensors.quantization import (
    initialize_module_for_quantization,
    preset_name_to_scheme,
)


def _compressed_linear(scheme_name: str, dtype: torch.dtype) -> torch.nn.Linear:
    module = torch.nn.Linear(256, 256, bias=True).to(dtype)
    scheme = preset_name_to_scheme(scheme_name, ["Linear"])
    initialize_module_for_quantization(module, scheme)
    with torch.no_grad():
        module.weight.normal_()
        module.weight_scale.fill_(1)
        if hasattr(module, "weight_global_scale"):
            module.weight_global_scale.fill_(1)
    compress_module(module)
    return module


@pytest.mark.parametrize(
    "scheme_name", ["NVFP4A16", "MXFP4A16", "MXFP6A16", "MXFP8A16", "W4A16", "FP8"]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_decompress_restores_dense_dtype(scheme_name, dtype):
    module = _compressed_linear(scheme_name, dtype)

    decompress_module(module)

    assert module.weight.dtype == dtype
    output = module(torch.randn(2, 256, dtype=dtype))
    assert output.dtype == dtype


@pytest.mark.parametrize("scheme_name", ["NVFP4A16", "MXFP4A16"])
def test_decompress_without_leaving_decompressed_clears_dense_dtype(scheme_name):
    module = _compressed_linear(scheme_name, torch.float16)

    decompress_module(module, leave_decompressed=False)

    assert module.weight.dtype == torch.float16
    assert not hasattr(module, "quantization_dense_dtype")
