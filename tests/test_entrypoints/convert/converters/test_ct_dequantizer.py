# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert.converters.ct_dequantizer import (
    CompressedTensorsDequantizer,
)
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)


def _create_dequantizer(ignore=None):
    dequantizer = object.__new__(CompressedTensorsDequantizer)
    dequantizer.dtype = torch.bfloat16

    scheme = QuantizationScheme(
        targets=["re:.*mlp.*"],
        weights=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False,
        ),
        format=CompressionFormat.naive_quantized,
    )

    dequantizer.quant_config = QuantizationConfig(
        config_groups={"group_0": scheme},
        ignore=ignore or [],
    )

    return dequantizer


def _create_dummy_tensors(device: torch.device = torch.device("cpu")):
    with device:
        return {
            "model.layers.0.mlp.up_proj.weight": torch.randint(
                -128, 127, (64, 64), dtype=torch.int8
            ),
            "model.layers.0.mlp.up_proj.weight_scale": torch.rand(
                64, 1, dtype=torch.float32
            ),
            "model.layers.0.mlp.down_proj.weight": torch.randint(
                -128, 127, (64, 64), dtype=torch.int8
            ),
            "model.language_model.layers.0.input_layernorm.weight": torch.randn(
                64, 1, dtype=torch.bfloat16
            ),
            "model.language_model.layers.0.pre_feedforward_layernorm.weight": torch.randn(  # noqa: E501
                64, 1, dtype=torch.bfloat16
            ),
            "model.language_model.layers.0.post_feedforward_layernorm.weight": torch.randn(  # noqa: E501
                64, 1, dtype=torch.bfloat16
            ),
            "model.layers.0.mlp.down_proj.weight_scale": torch.rand(
                64, 1, dtype=torch.float32
            ),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                128, 64, dtype=torch.bfloat16
            ),
            "model.embed_tokens.weight": torch.randn(128, 64, dtype=torch.bfloat16),
        }


@pytest.mark.unit
def test_process_dequantizes_targeted_layers():
    dequantizer = _create_dequantizer(ignore=["model.embed_tokens"])
    tensors = _create_dummy_tensors()
    qproj_weight = tensors["model.layers.0.self_attn.q_proj.weight"].clone()
    embed_tokens_weight = tensors["model.embed_tokens.weight"].clone()

    result = dequantizer.process(tensors)

    assert "model.layers.0.mlp.up_proj.weight" in result
    assert "model.layers.0.mlp.down_proj.weight" in result
    assert result["model.layers.0.mlp.up_proj.weight"].dtype == torch.bfloat16
    assert result["model.layers.0.mlp.down_proj.weight"].dtype == torch.bfloat16

    assert "model.layers.0.mlp.up_proj.weight_scale" not in result
    assert "model.layers.0.mlp.down_proj.weight_scale" not in result

    assert torch.equal(result["model.layers.0.self_attn.q_proj.weight"], qproj_weight)
    assert torch.equal(result["model.embed_tokens.weight"], embed_tokens_weight)


@pytest.mark.unit
def test_validate_passes_with_valid_tensors():
    dequantizer = _create_dequantizer(ignore=["model.embed_tokens"])
    tensors = _create_dummy_tensors(device=torch.device("meta"))

    dequantizer.validate(tensors)


@pytest.mark.unit
def test_validate_raises_on_missing_scale():
    dequantizer = _create_dequantizer(ignore=["model.embed_tokens"])
    tensors = _create_dummy_tensors(device=torch.device("meta"))
    del tensors["model.layers.0.mlp.up_proj.weight_scale"]

    with pytest.raises(ValueError, match="Expected key"):
        dequantizer.validate(tensors)


@pytest.mark.unit
def test_validate_raises_on_unconsumed_key():
    dequantizer = _create_dequantizer(ignore=["model.embed_tokens"])
    tensors = _create_dummy_tensors(device=torch.device("meta"))
    tensors["model.layers.0.mlp.up_proj.extra_param"] = torch.rand(64)

    with pytest.raises(ValueError, match="unconsumed keys"):
        dequantizer.validate(tensors)


@pytest.mark.unit
def test_get_dependencies_returns_scale_for_targeted_weight():
    dequantizer = _create_dequantizer(ignore=["model.embed_tokens"])

    deps = dequantizer.get_dependencies("model.layers.0.mlp.up_proj.weight")
    assert deps == {"model.layers.0.mlp.up_proj.weight_scale"}


@pytest.mark.unit
def test_get_dependencies_returns_empty_for_non_root_param():
    dequantizer = _create_dequantizer(ignore=["model.embed_tokens"])

    deps = dequantizer.get_dependencies("model.layers.0.mlp.up_proj.weight_scale")
    assert deps == set()


@pytest.mark.unit
def test_get_dependencies_returns_empty_for_ignored_module():
    dequantizer = _create_dequantizer(ignore=["model.embed_tokens"])

    deps = dequantizer.get_dependencies("model.embed_tokens.weight")
    assert deps == set()
