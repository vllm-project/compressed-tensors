# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
from compressed_tensors.compressors import LutBCompressor, ModelCompressor
from compressed_tensors.compressors.base import compress_module, decompress_module
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationType,
    initialize_module_for_quantization,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.utils import dequantize_lut_b, fake_quantize_lut_b
from compressed_tensors.utils import get_direct_state_dict
from pydantic import ValidationError
from safetensors.torch import load_file, save_file


def test_lut_b_preset_and_format_inference():
    layer = torch.nn.Linear(64, 8, bias=False)
    scheme = preset_name_to_scheme("LUTB", ["Linear"])
    initialize_module_for_quantization(layer, scheme)

    compressor = ModelCompressor.from_pretrained_model(layer)

    assert scheme.weights.type == QuantizationType.CODEBOOK
    assert scheme.weights.block_structure == [8, 64]
    assert not hasattr(layer, "weight_scale")
    assert not hasattr(layer, "weight_zero_point")
    assert compressor.quantization_config.format == CompressionFormat.lut_b


def test_lut_b_qdq_linear_forward():
    torch.manual_seed(1)
    layer = torch.nn.Linear(64, 8, bias=False, dtype=torch.bfloat16)
    scheme = preset_name_to_scheme("LUTB", ["Linear"])
    initialize_module_for_quantization(layer, scheme)
    inputs = torch.randn(3, 64, dtype=torch.bfloat16)

    output = layer(inputs)
    expected_weight = fake_quantize_lut_b(layer.weight, scheme.weights)

    torch.testing.assert_close(output, F.linear(inputs, expected_weight))


def test_lut_b_qdq_linear_forward_uses_precomputed_codebook():
    layer = torch.nn.Linear(64, 8, bias=False)
    scheme = preset_name_to_scheme("LUTB", ["Linear"])
    initialize_module_for_quantization(layer, scheme)
    codebooks = torch.linspace(-1.0, 1.0, 8).reshape(1, 1, 8)
    layer.register_parameter(
        "weight_codebook",
        torch.nn.Parameter(codebooks, requires_grad=False),
    )
    inputs = torch.randn(2, 64)

    output = layer(inputs)
    expected_weight = fake_quantize_lut_b(layer.weight, scheme.weights, codebooks)

    torch.testing.assert_close(output, F.linear(inputs, expected_weight))


def test_lut_b_compressor_uses_canonical_checkpoint_tensors():
    torch.manual_seed(2)
    layer = torch.nn.Linear(128, 16, bias=False, dtype=torch.bfloat16)
    scheme = preset_name_to_scheme("LUTB", ["Linear"])
    initialize_module_for_quantization(layer, scheme)

    compress_module(layer)
    state_dict = get_direct_state_dict(layer)

    assert layer.quantization_scheme.format == CompressionFormat.lut_b
    assert "weight" not in state_dict
    assert state_dict["weight_packed"].shape == (2, 2, 192)
    assert state_dict["weight_packed"].dtype == torch.uint8
    assert state_dict["weight_codebook"].shape == (2, 2, 8)
    assert state_dict["weight_codebook"].dtype == torch.float8_e4m3fn
    assert (
        state_dict["weight_packed"].numel() + state_dict["weight_codebook"].numel()
        == 4 * 200
    )

    expected = dequantize_lut_b(
        state_dict["weight_packed"],
        state_dict["weight_codebook"],
        scheme.weights,
    )
    decompress_module(layer)

    assert layer.weight.dtype == torch.bfloat16
    torch.testing.assert_close(layer.weight, expected)
    assert hasattr(layer, "weight_codebook")


def test_lut_b_compressor_preserves_precomputed_codebook():
    layer = torch.nn.Linear(64, 8, bias=False)
    scheme = preset_name_to_scheme("LUTB", ["Linear"])
    initialize_module_for_quantization(layer, scheme)
    codebooks = torch.linspace(-1.0, 1.0, 8).reshape(1, 1, 8)
    layer.register_parameter(
        "weight_codebook",
        torch.nn.Parameter(codebooks, requires_grad=False),
    )

    compress_module(layer)

    expected = codebooks.to(torch.float8_e4m3fn)
    torch.testing.assert_close(layer.weight_codebook, expected)


def test_lut_b_checkpoint_safetensors_round_trip(tmp_path):
    layer = torch.nn.Linear(64, 8, bias=False, dtype=torch.bfloat16)
    scheme = preset_name_to_scheme("LUTB", ["Linear"])
    initialize_module_for_quantization(layer, scheme)
    compress_module(layer)
    state_dict = {
        name: value
        for name, value in get_direct_state_dict(layer).items()
        if value is not None
    }
    checkpoint = tmp_path / "model.safetensors"

    save_file(state_dict, checkpoint)
    loaded = load_file(checkpoint)

    assert set(loaded) == {"weight_packed", "weight_codebook"}
    torch.testing.assert_close(loaded["weight_packed"], layer.weight_packed)
    torch.testing.assert_close(loaded["weight_codebook"], layer.weight_codebook)


def test_lut_b_meta_compression_creates_loadable_shapes():
    layer = torch.nn.Linear(128, 16, bias=False, device="meta")
    scheme = preset_name_to_scheme("LUTB", ["Linear"])
    initialize_module_for_quantization(layer, scheme)

    compress_module(layer)

    assert layer.weight_packed.shape == (2, 2, 192)
    assert layer.weight_packed.device.type == "meta"
    assert layer.weight_codebook.shape == (2, 2, 8)
    assert layer.weight_codebook.device.type == "meta"


def test_lut_b_compression_parameter_names():
    scheme = preset_name_to_scheme("LUTB", ["Linear"])

    assert LutBCompressor.compression_param_names(scheme) == (
        "weight_packed",
        "weight_codebook",
    )


def test_codebook_quantization_rejects_non_lut_b_contract():
    with pytest.raises(ValidationError, match="LUT-B contract"):
        QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=3,
                type="codebook",
                strategy="block",
                block_structure=[8, 32],
            ),
        )


def test_codebook_quantization_rejects_activations():
    with pytest.raises(ValidationError, match="only supported for weights"):
        QuantizationScheme(
            targets=["Linear"],
            input_activations=QuantizationArgs(
                num_bits=3,
                type="codebook",
                strategy="block",
                block_structure=[8, 64],
            ),
        )


def test_codebook_quantization_preserves_observer_configuration():
    args = QuantizationArgs(
        num_bits=3,
        type="codebook",
        strategy="block",
        block_structure=[8, 64],
        observer="custom_lut_b",
        observer_kwargs={"loss": "hessian"},
    )

    restored = QuantizationArgs.model_validate(args.model_dump())

    assert restored.observer == "custom_lut_b"
    assert restored.observer_kwargs == {"loss": "hessian"}
