# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import pytest
import torch
from compressed_tensors.compressors.base import compress_module, decompress_module
from compressed_tensors.compressors.pack_quantized import forward as pack_forward
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
    initialize_module_for_quantization,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.utils import calculate_qparams
from tests.testing_utils import requires_gpu


def _make_module(scheme, in_features, out_features, bias, device, dtype):
    module = torch.nn.Linear(in_features, out_features, bias=bias)
    module = module.to(device=device, dtype=dtype)
    initialize_module_for_quantization(module, scheme)

    # calibrate weight qparams from the weight
    with torch.no_grad():
        weight = module.weight
        weights = scheme.weights
        if weights.strategy in (
            QuantizationStrategy.GROUP,
            QuantizationStrategy.TENSOR_GROUP,
        ):
            grouped = weight.unflatten(-1, (-1, weights.group_size))
            scale, zero_point = calculate_qparams(
                grouped.amin(-1), grouped.amax(-1), weights
            )
        elif weights.strategy == QuantizationStrategy.CHANNEL:
            scale, zero_point = calculate_qparams(
                weight.amin(-1, keepdim=True), weight.amax(-1, keepdim=True), weights
            )
        else:
            scale, zero_point = calculate_qparams(
                weight.amin().reshape(1), weight.amax().reshape(1), weights
            )
        module.weight_scale.data = scale.to(module.weight_scale.dtype)

        if not weights.symmetric:
            module.weight_zero_point.data = zero_point.to(
                module.weight_zero_point.dtype
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
    "W4A16": preset_name_to_scheme("W4A16", ["Linear"]),
    "W8A16": preset_name_to_scheme("W8A16", ["Linear"]),
    "W4A16_ASYM": preset_name_to_scheme("W4A16_ASYM", ["Linear"]),
    "W8A16_CHANNEL_ASYM": QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=False,
        ),
    ),
    "W4A16_G64": QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=64,
        ),
    ),
}


@requires_gpu
@pytest.mark.parametrize("scheme_name", list(SCHEMES))
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape", [((4, 256), 384), ((2, 3, 512), 256), ((3, 97), 200), ((70, 320), 300)]
)
def test_pack_quantized_forward_matches_reference(scheme_name, bias, dtype, shape):
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
    assert module.weight_packed.dtype == torch.int32
    assert module.forward._is_compressed_forward

    input = torch.randn(input_shape, device="cuda", dtype=dtype)
    output = module(input)
    expected = _reference_forward(module, input)

    assert output.shape == (*input_shape[:-1], out_features)
    assert output.dtype == dtype
    # the kernel and the reference dequantize the same integer values, so they
    # differ only in floating point rounding of the matmul
    assert _relative_error(output, expected) < 5e-2

    # error with respect to the unquantized output is no worse than the reference
    with torch.no_grad():
        dense = torch.nn.functional.linear(input, dense_weight, module.bias)
    assert _relative_error(output, dense) < _relative_error(expected, dense) * 1.1


@requires_gpu
@pytest.mark.parametrize(
    "scheme_name,num_bits,symmetric,group_size",
    [
        ("W4A16", 4, True, 128),
        ("W8A16", 8, True, 128),
        ("W4A16_ASYM", 4, False, 128),
        ("W8A16_CHANNEL_ASYM", 8, False, None),
        ("W4A16_G64", 4, True, 64),
    ],
)
def test_pack_quantized_forward_resolution(
    scheme_name, num_bits, symmetric, group_size
):
    module = _make_module(SCHEMES[scheme_name], 256, 256, False, "cuda", torch.bfloat16)
    compress_module(module)
    forward = pack_forward.get_pack_quantized_linear_forward(module)
    assert isinstance(forward, partial)
    assert forward.func is pack_forward._pack_quantized_linear_triton
    assert forward.keywords["num_bits"] == num_bits
    assert forward.keywords["symmetric"] is symmetric
    assert forward.keywords["group_size"] == group_size


@pytest.mark.parametrize("scheme_name", ["W4A16", "W4A16_ASYM"])
def test_pack_quantized_forward_cpu_eager(scheme_name):
    torch.manual_seed(0)
    module = _make_module(SCHEMES[scheme_name], 256, 128, True, "cpu", torch.bfloat16)
    compress_module(module)
    forward = pack_forward.get_pack_quantized_linear_forward(module)
    assert forward is pack_forward._pack_quantized_linear_eager

    input = torch.randn(4, 256, dtype=torch.bfloat16)
    output = module(input)
    expected = _reference_forward(module, input)
    torch.testing.assert_close(output, expected)


@requires_gpu
def test_pack_quantized_forward_unsupported_bits_eager():
    scheme = SCHEMES["W4A16"].model_copy(deep=True)
    scheme.weights.num_bits = 3  # values span int32 words
    module = _make_module(scheme, 256, 128, False, "cuda", torch.bfloat16)
    compress_module(module)
    assert (
        pack_forward.get_pack_quantized_linear_forward(module)
        is pack_forward._pack_quantized_linear_eager
    )


@requires_gpu
def test_pack_quantized_forward_actorder_eager():
    module = _make_module(SCHEMES["W4A16"], 256, 128, False, "cuda", torch.bfloat16)
    module.register_buffer("weight_g_idx", torch.arange(2))
    compress_module(module)
    assert (
        pack_forward.get_pack_quantized_linear_forward(module)
        is pack_forward._pack_quantized_linear_eager
    )


@pytest.mark.parametrize("scheme_name", ["W8A8", "FP8_DYNAMIC"])
def test_other_formats_get_compressed_forward(scheme_name):
    module = torch.nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
    initialize_module_for_quantization(
        module, preset_name_to_scheme(scheme_name, ["Linear"])
    )
    quantized_forward_func = module.forward.__func__
    compress_module(module)
    assert module.forward.__func__ is not quantized_forward_func


def test_embedding_keeps_forward():
    module = torch.nn.Embedding(64, 128, dtype=torch.bfloat16)
    initialize_module_for_quantization(
        module, preset_name_to_scheme("W4A16", ["Embedding"])
    )
    with torch.no_grad():
        module.weight_scale.data.fill_(0.01)

    quantized_forward_func = module.forward.__func__
    compress_module(module)
    assert pack_forward.get_pack_quantized_linear_forward(module) is None
    assert module.forward.__func__ is quantized_forward_func
