# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch.nn as nn
from compressed_tensors.compressors.base import compress_module
from compressed_tensors.compressors.naive_quantized.base import (
    FloatQuantizationCompressor,
    dequantize_fp8_block_weight,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    initialize_module_for_quantization,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.quantization.utils import maybe_pad_tensor_for_block_quant
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from compressed_tensors.utils.impl_backend import ImplBackend
from tests.testing_utils import requires_gpu


def _block_scheme(block_structure=(128, 128)):
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type="float",
            strategy=QuantizationStrategy.BLOCK,
            symmetric=True,
            block_structure=list(block_structure),
        ),
    )


@pytest.mark.parametrize(
    "rows,cols,block_height,block_width",
    [
        (256, 256, 128, 128),
        (256, 512, 128, 128),
        (300, 400, 128, 128),  # non-divisible dims exercise padding
        (300, 256, 128, 128),
    ],
)
def test_dequantize_matches_decompress(rows, cols, block_height, block_width):
    """The standalone block dequant helper matches the compressor's decompress."""
    torch.manual_seed(0)
    scheme = _block_scheme((block_height, block_width))

    num_rb = math.ceil(rows / block_height)
    num_cb = math.ceil(cols / block_width)
    weight = (torch.randn(rows, cols) * 10).clamp(-448, 448).to(torch.float8_e4m3fn)
    weight_scale = torch.rand(num_rb, num_cb) * 0.01 + 0.001

    dequant = dequantize_fp8_block_weight(
        weight, weight_scale, (block_height, block_width), torch.float32
    )
    reference = FloatQuantizationCompressor.decompress(
        {"weight": weight, "weight_scale": weight_scale}, scheme
    )["weight"].to(torch.float32)

    assert dequant.shape == (rows, cols)
    assert torch.equal(dequant, reference)


def _emulated_forward(module: nn.Linear, input: torch.Tensor) -> torch.Tensor:
    """
    Reference forward: fake-quantize activations, dequantize the weight via the
    compressor's decompress, then run a dense linear.
    """
    scheme = module.quantization_scheme
    if scheme.input_activations is not None:
        input = forward_quantize(module, input, "input", scheme.input_activations)

    decompressed = FloatQuantizationCompressor.decompress(
        {"weight": module.weight, "weight_scale": module.weight_scale}, scheme
    )
    weight = decompressed["weight"].to(input.dtype)
    return torch.nn.functional.linear(input, weight, module.bias)


def _make_compressed_fp8_block_linear(in_features, out_features, bias):
    module = nn.Linear(in_features, out_features, bias=bias)
    module = module.to(dtype=torch.bfloat16, device="cuda")

    scheme = preset_name_to_scheme("FP8_BLOCK", ["Linear"])
    initialize_module_for_quantization(module, scheme)

    # Calibrate the weight scale from the actual (block-wise) weights so the
    # quantized weights are representative rather than degenerate.
    weights = scheme.weights
    block_height, block_width = weights.block_structure
    padded = maybe_pad_tensor_for_block_quant(
        module.weight.data, (block_height, block_width)
    )
    num_rb = padded.shape[0] // block_height
    num_cb = padded.shape[1] // block_width
    blocks = padded.reshape(num_rb, block_height, num_cb, block_width)
    scale, _ = calculate_qparams(
        blocks.amin(dim=(1, 3)), blocks.amax(dim=(1, 3)), weights
    )
    module.weight_scale.data = scale.to(module.weight_scale.dtype)

    compress_module(module)
    return module


@requires_gpu
@pytest.mark.parametrize("bias", [False, True])
def test_forward_dispatch_matches_emulated(bias):
    """The dispatched (fastest available) backend matches the emulated forward."""
    torch.manual_seed(0)
    in_features, out_features = 256, 512

    module = _make_compressed_fp8_block_linear(in_features, out_features, bias)
    input = torch.randn(8, in_features, dtype=torch.bfloat16, device="cuda")

    actual = FloatQuantizationCompressor.compressed_forward(module, input)
    expected = _emulated_forward(module, input)

    assert actual.shape == (8, out_features)
    # dispatch may select the real fp8 tensor-core backend, which is inherently
    # less precise than the bf16 emulation reference
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


@requires_gpu
@pytest.mark.parametrize(
    "backend_fn",
    ["fp8_block_compressed_forward", "fp8_block_forward_emulation"],
)
@pytest.mark.parametrize("bias", [False, True])
def test_forward_backends_match_emulated(backend_fn, bias):
    """Each registered backend (eager, triton emulation) matches the emulated
    reference forward."""
    torch.manual_seed(0)
    in_features, out_features = 256, 512

    module = _make_compressed_fp8_block_linear(in_features, out_features, bias)
    input = torch.randn(8, in_features, dtype=torch.bfloat16, device="cuda")

    # invoke a specific backend directly, bypassing dispatch
    actual = ImplBackend.call(backend_fn, module, input)
    expected = _emulated_forward(module, input)

    assert actual.shape == (8, out_features)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_gpu
@pytest.mark.parametrize("bias", [False, True])
def test_fp8_tensorcore_backend_matches_emulated(bias):
    """The real fp8 tensor-core backend matches the emulated reference forward."""
    if torch.get_device_module().get_device_capability() < (8, 9):
        pytest.skip("fp8 tensor cores require compute capability >= 8.9")

    torch.manual_seed(0)
    in_features, out_features = 256, 512

    module = _make_compressed_fp8_block_linear(in_features, out_features, bias)
    input = torch.randn(8, in_features, dtype=torch.bfloat16, device="cuda")

    actual = ImplBackend.call("fp8_block_forward_fp8", module, input)
    expected = _emulated_forward(module, input)

    assert actual.shape == (8, out_features)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


@requires_gpu
def test_compress_module_overwrites_quantized_forward():
    """``compress_module`` overwrites the fake-quantized forward installed by
    ``set_forward_quantized`` with the block compressed forward, and the module
    can be called directly through its ``forward`` afterwards."""
    torch.manual_seed(0)
    in_features, out_features = 256, 512

    module = nn.Linear(in_features, out_features, bias=True)
    module = module.to(dtype=torch.bfloat16, device="cuda")

    scheme = preset_name_to_scheme("FP8_BLOCK", ["Linear"])
    initialize_module_for_quantization(module, scheme)

    # initialization installs the fake-quantized forward (not yet compressed)
    quantized_func = module.forward.__func__
    assert quantized_func.__name__ != "fp8_block_compressed_forward"

    weights = scheme.weights
    block_height, block_width = weights.block_structure
    padded = maybe_pad_tensor_for_block_quant(
        module.weight.data, (block_height, block_width)
    )
    num_rb = padded.shape[0] // block_height
    num_cb = padded.shape[1] // block_width
    blocks = padded.reshape(num_rb, block_height, num_cb, block_width)
    scale, _ = calculate_qparams(
        blocks.amin(dim=(1, 3)), blocks.amax(dim=(1, 3)), weights
    )
    module.weight_scale.data = scale.to(module.weight_scale.dtype)

    compress_module(module)

    # compression overwrites it with the block compressed forward
    assert module.quantization_status == QuantizationStatus.COMPRESSED
    assert module.forward.__func__ is not quantized_func
    assert module.forward.__func__.__name__ == "fp8_block_compressed_forward"

    input = torch.randn(8, in_features, dtype=torch.bfloat16, device="cuda")
    actual = module.forward(input)
    expected = _emulated_forward(module, input)

    assert actual.shape == (8, out_features)
    # dispatch may select the real fp8 tensor-core backend, which is inherently
    # less precise than the bf16 emulation reference
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


@requires_gpu
def test_forward_supports_multidim_input():
    """A batched (>2D) input is handled by flattening and restoring dims."""
    torch.manual_seed(0)
    in_features, out_features = 256, 512

    module = _make_compressed_fp8_block_linear(in_features, out_features, True)
    input = torch.randn(2, 4, in_features, dtype=torch.bfloat16, device="cuda")

    actual = FloatQuantizationCompressor.compressed_forward(module, input)
    expected = _emulated_forward(module, input)

    assert actual.shape == (2, 4, out_features)
    # dispatch may select the real fp8 tensor-core backend, which is inherently
    # less precise than the bf16 emulation reference
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("scheme_name", ["FP8", "FP8_DYNAMIC"])
def test_non_block_fp8_keeps_decompress_forward(scheme_name):
    """Non-block FP8 strategies must not install the block compressed forward."""
    module = nn.Linear(256, 256, bias=False)
    scheme = preset_name_to_scheme(scheme_name, ["Linear"])
    initialize_module_for_quantization(module, scheme)
    module.weight_scale.data.fill_(0.01)

    original_forward = module.forward.__func__
    compress_module(module)

    assert module.forward.__func__ is original_forward
    assert module.forward.__func__.__name__ != "fp8_block_compressed_forward"
