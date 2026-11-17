# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import pytest
import torch
from compressed_tensors.compressors.base import compress_module, decompress_module
from compressed_tensors.compressors.naive_quantized import int8_forward
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
        if weights.strategy == QuantizationStrategy.GROUP:
            amax = weight.unflatten(-1, (-1, weights.group_size)).abs().amax(-1)
        elif weights.strategy == QuantizationStrategy.CHANNEL:
            amax = weight.abs().amax(-1, keepdim=True)
        else:
            amax = weight.abs().amax().reshape(1)
        # symmetric INT scale, matching calculate_qparams
        bit_range = 2.0**weights.num_bits - 1
        module.weight_scale.data = (amax / (bit_range / 2)).to(
            module.weight_scale.dtype
        )

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
    "W8A8": preset_name_to_scheme("W8A8", ["Linear"]),
    "W4A8": preset_name_to_scheme("W4A8", ["Linear"]),
    "W4A4": preset_name_to_scheme("W4A4", ["Linear"]),
    "W2A8": preset_name_to_scheme("W2A8", ["Linear"]),
    "W4A8_G64": QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=64,
        ),
        input_activations=preset_name_to_scheme("W4A8", ["Linear"]).input_activations,
    ),
    "W8A8_TENSOR": QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TENSOR,
        ),
        input_activations=preset_name_to_scheme("W8A8", ["Linear"]).input_activations,
    ),
}

# schemes which are not served by the Triton kernel and use the eager fallback
UNSUPPORTED_SCHEMES = {
    # asymmetric weights
    "ASYM_W4A8": QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
            symmetric=False,
        ),
        input_activations=preset_name_to_scheme("W4A8", ["Linear"]).input_activations,
    ),
    # activations are FP8 rather than INT
    "W4AFP8": preset_name_to_scheme("W4AFP8", ["Linear"]),
    # static per-tensor activation scale rather than dynamic per-token
    "STATIC_W8A8": QuantizationScheme(
        targets=["Linear"],
        weights=preset_name_to_scheme("W8A8", ["Linear"]).weights,
        input_activations=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TENSOR,
        ),
    ),
    # activation ordering
    "ACTORDER_W4A8": QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
            actorder="weight",
        ),
        input_activations=preset_name_to_scheme("W4A8", ["Linear"]).input_activations,
    ),
    # weights are stored as int16 rather than int8
    "W16A8": QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=16,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
        ),
        input_activations=preset_name_to_scheme("W8A8", ["Linear"]).input_activations,
    ),
    # activations are quantized to 16 rather than 4 or 8 bits
    "W8A16_INT_ACT": QuantizationScheme(
        targets=["Linear"],
        weights=preset_name_to_scheme("W8A8", ["Linear"]).weights,
        input_activations=QuantizationArgs(
            num_bits=16,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TOKEN,
            dynamic=True,
        ),
    ),
}

# relative error tolerance of the kernel against the reference forward. 4-bit
# activation quantization is coarse, and for 2D inputs the reference computes a
# per-tensor dynamic scale while the kernel computes per-row scales (see the
# note in test_fp8_forward.py)
TOLERANCES = {
    "W8A8": 5e-2,
    "W4A8": 5e-2,
    "W4A4": 2.5e-1,
    "W2A8": 5e-2,
    "W4A8_G64": 5e-2,
    "W8A8_TENSOR": 5e-2,
}


@requires_gpu
@pytest.mark.parametrize("scheme_name", list(SCHEMES))
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape", [((4, 256), 384), ((2, 3, 512), 256), ((1, 256), 200), ((70, 320), 300)]
)
def test_int8_forward_matches_reference(scheme_name, bias, dtype, shape):
    torch.manual_seed(0)
    input_shape, out_features = shape
    in_features = input_shape[-1]
    scheme = SCHEMES[scheme_name].model_copy(deep=True)
    if (
        scheme.weights.strategy == QuantizationStrategy.GROUP
        and in_features % scheme.weights.group_size != 0
    ):
        pytest.skip("group size must divide in_features")

    module = _make_module(scheme, in_features, out_features, bias, "cuda", dtype)
    dense_weight = module.weight.detach().clone()
    compress_module(module)
    assert module.weight.dtype == torch.int8
    assert module.forward._is_compressed_forward

    input = torch.randn(input_shape, device="cuda", dtype=dtype)
    output = module(input)
    expected = _reference_forward(module, input)

    assert output.shape == (*input_shape[:-1], out_features)
    assert output.dtype == dtype
    # quantization noise is not elementwise identical (for example, dynamic token
    # scales are computed per row rather than per tensor for 2D inputs)
    assert _relative_error(output, expected) < TOLERANCES[scheme_name]

    # error with respect to the unquantized output is no worse than the reference
    with torch.no_grad():
        dense = torch.nn.functional.linear(input, dense_weight, module.bias)
    assert _relative_error(output, dense) < _relative_error(expected, dense) * 1.1


@requires_gpu
@pytest.mark.parametrize(
    "scheme_name,w_group_n,block_k,x_qmax",
    [
        ("W8A8", 1, 128, 127),
        ("W4A8", 1, 128, 127),
        ("W4A4", 1, 128, 7),
        ("W2A8", 1, 128, 127),
        ("W4A8_G64", 1, 64, 127),
        ("W8A8_TENSOR", 1, 128, 127),
    ],
)
def test_int8_forward_resolution(scheme_name, w_group_n, block_k, x_qmax):
    module = _make_module(SCHEMES[scheme_name], 256, 256, False, "cuda", torch.bfloat16)
    compress_module(module)
    forward = int8_forward.get_int8_linear_forward(module)
    assert isinstance(forward, partial)
    assert forward.func is int8_forward._int8_linear_triton
    assert forward.keywords["w_group_n"] == w_group_n
    assert forward.keywords["block_k"] == block_k
    assert forward.keywords["x_qmax"] == x_qmax


@requires_gpu
@pytest.mark.parametrize("scheme_name", list(UNSUPPORTED_SCHEMES))
def test_int8_forward_unsupported_eager(scheme_name):
    module = _make_module(
        UNSUPPORTED_SCHEMES[scheme_name], 256, 256, False, "cuda", torch.bfloat16
    )
    compress_module(module)
    assert (
        int8_forward.get_int8_linear_forward(module) is int8_forward._int8_linear_eager
    )


def test_int8_forward_cpu_eager():
    torch.manual_seed(0)
    module = _make_module(SCHEMES["W8A8"], 256, 128, True, "cpu", torch.bfloat16)
    compress_module(module)
    assert (
        int8_forward.get_int8_linear_forward(module) is int8_forward._int8_linear_eager
    )

    input = torch.randn(4, 256, dtype=torch.bfloat16)
    output = module(input)
    expected = _reference_forward(module, input)
    torch.testing.assert_close(output, expected)


def test_int8_forward_embedding_keeps_forward():
    module = torch.nn.Embedding(64, 32, dtype=torch.bfloat16)
    initialize_module_for_quantization(
        module, preset_name_to_scheme("W8A8", ["Linear"])
    )
    quantized_forward_func = module.forward.__func__
    compress_module(module)
    assert module.forward.__func__ is quantized_forward_func


@requires_gpu
@pytest.mark.parametrize("scheme_name", ["FP8_DYNAMIC", "W4A16"])
def test_other_formats_get_compressed_forward(scheme_name):
    module = torch.nn.Linear(256, 128, bias=False, dtype=torch.bfloat16, device="cuda")
    initialize_module_for_quantization(
        module, preset_name_to_scheme(scheme_name, ["Linear"])
    )
    quantized_forward_func = module.forward.__func__
    compress_module(module)
    assert module.forward.__func__ is not quantized_forward_func
