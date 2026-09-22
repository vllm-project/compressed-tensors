# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn
from compressed_tensors.compressors.base import compress_module
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationType,
    initialize_module_for_quantization,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from compressed_tensors.utils.impl_backend import ImplBackend
from tests.testing_utils import requires_gpu


def test_compress_scale_without_scale_dtype():
    """
    Test that MXFP4 compressor handles missing scale_dtype.

    (backward compatibility)
    """
    # Create a scale tensor
    scale = torch.randn(10, dtype=torch.bfloat16).abs() + 1e-6  # Ensure positive values

    # Create QuantizationArgs without scale_dtype (as in older models)
    quant_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        symmetric=True,
        group_size=32,
        # scale_dtype is not set (defaults to None)
    )

    # This should not raise an error and should default to uint8
    compressed_scale = MXFP4PackedCompressor._compress_scale(scale, quant_args)

    # Verify the output dtype is uint8
    assert compressed_scale.dtype == torch.uint8


def test_compress_scale_with_scale_dtype():
    """Test that MXFP4 compressor respects explicit scale_dtype"""
    # Create a scale tensor
    scale = torch.randn(10, dtype=torch.bfloat16).abs() + 1e-6  # Ensure positive values

    # Create QuantizationArgs with explicit scale_dtype
    quant_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        symmetric=True,
        group_size=32,
        scale_dtype=torch.uint8,
    )

    # Compress the scale
    compressed_scale = MXFP4PackedCompressor._compress_scale(scale, quant_args)

    # Verify the output dtype matches the specified scale_dtype
    assert compressed_scale.dtype == torch.uint8


def test_decompress_decodes_mx_scales_and_restores_weight():
    quant_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        symmetric=True,
        group_size=32,
        scale_dtype=torch.uint8,
    )
    # group_size=32 is the source of truth, so each scale group covers 32
    # weight columns. Use two groups (scales 0.25 and 0.5) spanning 64 columns.
    scale = torch.tensor([[0.25, 0.5]], dtype=torch.bfloat16)
    group_values = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.bfloat16)
    fp4_values = group_values.repeat(16).unsqueeze(0)  # (1, 64)
    packed = pack_fp4_to_uint8(fp4_values)

    decompressed = MXFP4PackedCompressor.decompress(
        {
            "weight_packed": packed,
            "weight_scale": MXFP4PackedCompressor._compress_scale(scale, quant_args),
        },
        QuantizationScheme(targets=["Linear"], weights=quant_args),
    )

    expected_group0 = group_values * 0.25  # first 32 columns
    expected_group1 = group_values * 0.5  # last 32 columns
    expected_weight = torch.cat(
        [expected_group0.repeat(8), expected_group1.repeat(8)]
    ).unsqueeze(0)

    assert torch.equal(decompressed["weight_scale"], scale)
    assert torch.equal(decompressed["weight"], expected_weight)


def _emulated_forward(module: nn.Linear, input: torch.Tensor) -> torch.Tensor:
    """
    Reference forward: fake-quantize activations, unpack + dequantize the
    weight, then run a dense linear.
    """
    scheme = module.quantization_scheme
    if scheme.input_activations is not None:
        input = forward_quantize(module, input, "input", scheme.input_activations)

    decompressed = MXFP4PackedCompressor.decompress(
        {
            "weight_packed": module.weight_packed,
            "weight_scale": module.weight_scale,
        },
        scheme,
    )
    weight = decompressed["weight"].to(input.dtype)
    return torch.nn.functional.linear(input, weight, module.bias)


def _make_compressed_mxfp4_linear(scheme_name, in_features, out_features, bias):
    module = nn.Linear(in_features, out_features, bias=bias)
    module = module.to(dtype=torch.bfloat16, device="cuda")

    scheme = preset_name_to_scheme(scheme_name, ["Linear"])
    initialize_module_for_quantization(module, scheme)

    # Calibrate the weight scale from the actual (group-wise) weights so the
    # packed weights are representative rather than degenerate.
    weights = scheme.weights
    group_size = weights.group_size
    grouped = module.weight.data.unflatten(-1, (in_features // group_size, group_size))
    scale, _ = calculate_qparams(grouped.amin(-1), grouped.amax(-1), weights)
    module.weight_scale.data = scale.to(module.weight_scale.dtype)

    compress_module(module)
    return module


@requires_gpu
@pytest.mark.parametrize("scheme_name", ["MXFP4A16", "MXFP4"])
@pytest.mark.parametrize("bias", [False, True])
def test_forward_dispatch_matches_emulated(scheme_name, bias):
    """The dispatched (fastest available) backend matches the emulated forward."""
    torch.manual_seed(0)
    in_features, out_features = 128, 256

    module = _make_compressed_mxfp4_linear(scheme_name, in_features, out_features, bias)
    input = torch.randn(8, in_features, dtype=torch.bfloat16, device="cuda")

    actual = MXFP4PackedCompressor.compressed_forward(module, input)
    expected = _emulated_forward(module, input)

    assert actual.shape == (8, out_features)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_gpu
@pytest.mark.parametrize(
    "backend_fn",
    ["compressed_forward", "mxfp4_forward_emulation", "mxfp4_forward_fp4"],
)
@pytest.mark.parametrize("scheme_name", ["MXFP4A16", "MXFP4"])
@pytest.mark.parametrize("bias", [False, True])
def test_forward_backends_match_emulated(backend_fn, scheme_name, bias):
    """Each registered backend (eager, triton emulation, fp4 tensor cores)
    matches the emulated reference forward."""
    if backend_fn == "mxfp4_forward_fp4" and (
        torch.cuda.get_device_capability()[0] < 10
    ):
        pytest.skip("FP4 tensor cores require Blackwell (SM100+)")

    torch.manual_seed(0)
    in_features, out_features = 128, 256

    module = _make_compressed_mxfp4_linear(scheme_name, in_features, out_features, bias)
    input = torch.randn(8, in_features, dtype=torch.bfloat16, device="cuda")

    # invoke a specific backend directly, bypassing dispatch
    actual = ImplBackend.call(backend_fn, module, input)
    expected = _emulated_forward(module, input)

    assert actual.shape == (8, out_features)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_gpu
def test_compress_module_overwrites_quantized_forward():
    """``compress_module`` overwrites the fake-quantized forward installed by
    ``set_forward_quantized`` with ``compressed_forward``, and the module can be
    called directly through its ``forward`` afterwards."""
    torch.manual_seed(0)
    in_features, out_features = 128, 256

    module = nn.Linear(in_features, out_features, bias=True)
    module = module.to(dtype=torch.bfloat16, device="cuda")

    scheme = preset_name_to_scheme("MXFP4", ["Linear"])
    initialize_module_for_quantization(module, scheme)

    # initialization installs the fake-quantized forward (not yet compressed)
    quantized_func = module.forward.__func__
    assert quantized_func.__name__ != "compressed_forward"

    weights = scheme.weights
    group_size = weights.group_size
    grouped = module.weight.data.unflatten(-1, (in_features // group_size, group_size))
    scale, _ = calculate_qparams(grouped.amin(-1), grouped.amax(-1), weights)
    module.weight_scale.data = scale.to(module.weight_scale.dtype)

    compress_module(module)

    # compression overwrites it with the compressed forward
    assert module.quantization_status == QuantizationStatus.COMPRESSED
    assert module.forward.__func__ is not quantized_func
    assert module.forward.__func__.__name__ == "compressed_forward"

    input = torch.randn(8, in_features, dtype=torch.bfloat16, device="cuda")
    actual = module.forward(input)
    expected = _emulated_forward(module, input)

    assert actual.shape == (8, out_features)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_gpu
def test_forward_supports_multidim_input():
    """A batched (>2D) input is handled by flattening and restoring dims."""
    torch.manual_seed(0)
    in_features, out_features = 128, 256

    module = _make_compressed_mxfp4_linear("MXFP4", in_features, out_features, True)
    input = torch.randn(2, 4, in_features, dtype=torch.bfloat16, device="cuda")

    actual = MXFP4PackedCompressor.compressed_forward(module, input)
    expected = _emulated_forward(module, input)

    assert actual.shape == (2, 4, out_features)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
