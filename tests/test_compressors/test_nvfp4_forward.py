# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import pytest
import torch
from compressed_tensors.compressors.base import compress_module, decompress_module
from compressed_tensors.compressors.nvfp4 import forward as nvfp4_forward
from compressed_tensors.quantization import (
    initialize_module_for_quantization,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from tests.testing_utils import requires_gpu


def _make_module(scheme_name, in_features, out_features, bias, device, dtype):
    module = torch.nn.Linear(in_features, out_features, bias=bias)
    module = module.to(device=device, dtype=dtype)
    scheme = preset_name_to_scheme(scheme_name, ["Linear"])
    initialize_module_for_quantization(module, scheme)

    # calibrate weight qparams from the weight
    with torch.no_grad():
        global_scale = generate_gparam(module.weight.min(), module.weight.max())
        module.weight_global_scale.data = global_scale
        grouped = module.weight.unflatten(-1, (-1, scheme.weights.group_size))
        scale, _ = calculate_qparams(
            grouped.amin(-1), grouped.amax(-1), scheme.weights, global_scale
        )
        module.weight_scale.data = scale.to(module.weight_scale.dtype)

        # calibrate the activation global scale from fake calibration data
        if hasattr(module, "input_global_scale"):
            calib = torch.randn(32, in_features, device=device, dtype=dtype)
            module.input_global_scale.data = generate_gparam(calib.min(), calib.max())

    return module


def _reference_forward(module, input):
    """Fake-quantized forward of the decompressed module"""
    if input.dtype != torch.bfloat16:
        # decompression always produces bfloat16 weights
        return nvfp4_forward._nvfp4_linear_eager(module, input)

    decompress_module(module)
    output = module(input)
    compress_module(module)
    return output


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@requires_gpu
@pytest.mark.parametrize("scheme_name", ["NVFP4A16", "NVFP4"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape", [((4, 256), 384), ((2, 3, 512), 256), ((1, 96), 200), ((70, 320), 300)]
)
def test_nvfp4_forward_matches_reference(scheme_name, bias, dtype, shape):
    torch.manual_seed(0)
    input_shape, out_features = shape
    in_features = input_shape[-1]

    module = _make_module(scheme_name, in_features, out_features, bias, "cuda", dtype)
    dense_weight = module.weight.detach().clone()
    compress_module(module)
    assert module.weight_packed.dtype == torch.uint8
    assert module.weight_scale.dtype == torch.float8_e4m3fn
    assert module.forward._is_compressed_forward

    input = torch.randn(input_shape, device="cuda", dtype=dtype)
    output = module(input)
    expected = _reference_forward(module, input)

    assert output.shape == (*input_shape[:-1], out_features)
    assert output.dtype == dtype
    assert _relative_error(output, expected) < 1e-2

    # error with respect to the unquantized output is no worse than the reference
    with torch.no_grad():
        dense = torch.nn.functional.linear(input, dense_weight, module.bias)
    assert _relative_error(output, dense) < _relative_error(expected, dense) * 1.1


@requires_gpu
@pytest.mark.parametrize(
    "scheme_name,input_fp4", [("NVFP4A16", False), ("NVFP4", True)]
)
def test_nvfp4_forward_resolution(scheme_name, input_fp4):
    module = _make_module(scheme_name, 256, 256, False, "cuda", torch.bfloat16)
    compress_module(module)
    forward = nvfp4_forward.get_nvfp4_linear_forward(module)
    assert isinstance(forward, partial)
    assert forward.func is nvfp4_forward._nvfp4_linear_triton
    assert forward.keywords["input_fp4"] == input_fp4
    assert forward.keywords["has_w_global"]


@pytest.mark.parametrize("scheme_name", ["NVFP4A16", "NVFP4"])
def test_nvfp4_forward_cpu_eager(scheme_name):
    torch.manual_seed(0)
    module = _make_module(scheme_name, 256, 128, True, "cpu", torch.bfloat16)
    compress_module(module)
    forward = nvfp4_forward.get_nvfp4_linear_forward(module)
    assert forward is nvfp4_forward._nvfp4_linear_eager

    input = torch.randn(4, 256, dtype=torch.bfloat16)
    output = module(input)
    expected = _reference_forward(module, input)
    assert _relative_error(output, expected) < 1e-2


@pytest.mark.parametrize("scheme_name", ["MXFP4A16", "MXFP6A16"])
def test_other_fp_formats_keep_forward(scheme_name):
    module = torch.nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
    initialize_module_for_quantization(
        module, preset_name_to_scheme(scheme_name, ["Linear"])
    )
    quantized_forward_func = module.forward.__func__
    compress_module(module)
    assert module.forward.__func__ is quantized_forward_func
