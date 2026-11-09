# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import pytest
import torch
from compressed_tensors.compressors.base import compress_module, decompress_module
from compressed_tensors.compressors.naive_quantized import fp8_forward
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
    initialize_module_for_quantization,
    preset_name_to_scheme,
)
from tests.testing_utils import requires_gpu


def _make_module(scheme, in_features, out_features, bias, device, dtype):
    module = torch.nn.Linear(in_features, out_features, bias=bias)
    module = module.to(device=device, dtype=dtype)
    initialize_module_for_quantization(module, scheme)

    # calibrate weight qparams from the weight
    with torch.no_grad():
        weight = module.weight
        weights = scheme.weights
        if weights.strategy == QuantizationStrategy.BLOCK:
            bn, bk = weights.block_structure
            n, k = weight.shape
            blocks = torch.nn.functional.pad(
                weight, (0, -k % bk, 0, -n % bn)
            ).unflatten(0, (-1, bn))
            amax = blocks.unflatten(-1, (-1, bk)).abs().amax(dim=(1, 3))
        elif weights.strategy == QuantizationStrategy.GROUP:
            amax = weight.unflatten(-1, (-1, weights.group_size)).abs().amax(-1)
        elif weights.strategy == QuantizationStrategy.CHANNEL:
            amax = weight.abs().amax(-1, keepdim=True)
        else:
            amax = weight.abs().amax().reshape(1)
        module.weight_scale.data = (amax / 448.0).to(module.weight_scale.dtype)

        if hasattr(module, "input_scale"):
            module.input_scale.data.fill_(4.0 / 448.0)

    return module


def _reference_forward(module, input):
    """Fake-quantized forward of the decompressed module"""
    decompress_module(module)
    output = module(input)
    compress_module(module)
    return output


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


SCHEMES = {
    "FP8_BLOCK": preset_name_to_scheme("FP8_BLOCK", ["Linear"]),
    "FP8_DYNAMIC": preset_name_to_scheme("FP8_DYNAMIC", ["Linear"]),
    "FP8": preset_name_to_scheme("FP8", ["Linear"]),
    "FP8_GROUP_W8A16": QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.GROUP,
            group_size=64,
        ),
    ),
    "FP8_BLOCK_A_TOKEN": QuantizationScheme(
        targets=["Linear"],
        weights=preset_name_to_scheme("FP8_BLOCK", ["Linear"]).weights,
        input_activations=preset_name_to_scheme(
            "FP8_DYNAMIC", ["Linear"]
        ).input_activations,
    ),
}


@requires_gpu
@pytest.mark.parametrize("scheme_name", list(SCHEMES))
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape", [((4, 256), 384), ((2, 3, 512), 256), ((1, 256), 200), ((70, 320), 300)]
)
def test_fp8_forward_matches_reference(scheme_name, bias, dtype, shape):
    torch.manual_seed(0)
    input_shape, out_features = shape
    in_features = input_shape[-1]
    scheme = SCHEMES[scheme_name].model_copy(deep=True)
    if scheme_name in ("FP8_BLOCK", "FP8_GROUP_W8A16") and in_features % 128 != 0:
        pytest.skip("group size must divide in_features")

    module = _make_module(scheme, in_features, out_features, bias, "cuda", dtype)
    dense_weight = module.weight.detach().clone()
    compress_module(module)
    assert module.weight.dtype == torch.float8_e4m3fn
    assert module.forward._is_compressed_forward

    input = torch.randn(input_shape, device="cuda", dtype=dtype)
    output = module(input)
    expected = _reference_forward(module, input)

    assert output.shape == (*input_shape[:-1], out_features)
    assert output.dtype == dtype
    # quantization noise is not elementwise identical (for example, dynamic token
    # scales are computed per row rather than per tensor for 2D inputs)
    assert _relative_error(output, expected) < 5e-2

    # error with respect to the unquantized output is no worse than the reference
    with torch.no_grad():
        dense = torch.nn.functional.linear(input, dense_weight, module.bias)
    assert _relative_error(output, dense) < _relative_error(expected, dense) * 1.1


@requires_gpu
@pytest.mark.parametrize(
    "scheme_name,input_quant",
    [
        ("FP8_BLOCK", fp8_forward.InputQuant.GROUP),
        ("FP8_DYNAMIC", fp8_forward.InputQuant.TOKEN),
        ("FP8", fp8_forward.InputQuant.TENSOR),
        ("FP8_GROUP_W8A16", fp8_forward.InputQuant.NONE),
        ("FP8_BLOCK_A_TOKEN", fp8_forward.InputQuant.TOKEN),
    ],
)
def test_fp8_forward_resolution(scheme_name, input_quant):
    module = _make_module(SCHEMES[scheme_name], 256, 256, False, "cuda", torch.bfloat16)
    compress_module(module)
    forward = fp8_forward.get_fp8_linear_forward(module)
    assert isinstance(forward, partial)
    assert forward.func is fp8_forward._fp8_linear_triton
    assert forward.keywords["input_quant"] == input_quant


def test_fp8_forward_cpu_eager():
    torch.manual_seed(0)
    module = _make_module(SCHEMES["FP8_BLOCK"], 256, 128, True, "cpu", torch.bfloat16)
    compress_module(module)
    assert fp8_forward.get_fp8_linear_forward(module) is fp8_forward._fp8_linear_eager

    input = torch.randn(4, 256, dtype=torch.bfloat16)
    output = module(input)
    expected = _reference_forward(module, input)
    torch.testing.assert_close(output, expected)


@requires_gpu
def test_fp8_forward_unsupported_block_eager():
    scheme = SCHEMES["FP8_BLOCK"].model_copy(deep=True)
    scheme.weights.block_structure = [128, 96]
    module = _make_module(scheme, 192, 128, False, "cuda", torch.bfloat16)
    compress_module(module)
    assert fp8_forward.get_fp8_linear_forward(module) is fp8_forward._fp8_linear_eager


def test_mxfp8_keeps_forward():
    module = torch.nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
    initialize_module_for_quantization(
        module, preset_name_to_scheme("MXFP8A16", ["Linear"])
    )
    quantized_forward_func = module.forward.__func__
    compress_module(module)
    assert module.forward.__func__ is quantized_forward_func
