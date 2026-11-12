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
from compressed_tensors.quantization.quant_scheme import FP8


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

    result = dequantizer.validate(tensors)

    # Targeted modules: compressed params replaced by single weight in dtype
    assert "model.layers.0.mlp.up_proj.weight" in result
    assert result["model.layers.0.mlp.up_proj.weight"].dtype == torch.bfloat16
    assert "model.layers.0.mlp.down_proj.weight" in result
    assert result["model.layers.0.mlp.down_proj.weight"].dtype == torch.bfloat16

    # Compressed params consumed (removed)
    assert "model.layers.0.mlp.up_proj.weight_scale" not in result
    assert "model.layers.0.mlp.down_proj.weight_scale" not in result

    # Untargeted/ignored tensors pass through
    assert "model.layers.0.self_attn.q_proj.weight" in result
    assert "model.embed_tokens.weight" in result

    # Layernorm weights pass through
    assert "model.language_model.layers.0.input_layernorm.weight" in result


@pytest.mark.unit
def test_validate_raises_on_missing_scale():
    dequantizer = _create_dequantizer(ignore=["model.embed_tokens"])
    tensors = _create_dummy_tensors(device=torch.device("meta"))
    del tensors["model.layers.0.mlp.up_proj.weight_scale"]

    with pytest.raises(ValueError):
        dequantizer.validate(tensors)


@pytest.mark.unit
def test_validate_passes_with_ignored_compressed_module():
    # An ignored module keeps its compressed qparams; the residual check must
    # not flag them as orphans.
    dequantizer = _create_dequantizer(ignore=["model.embed_tokens", "re:.*down_proj.*"])
    tensors = _create_dummy_tensors(device=torch.device("meta"))

    result = dequantizer.validate(tensors)

    # up_proj is dequantized, down_proj is left compressed (ignored)
    assert result["model.layers.0.mlp.up_proj.weight"].dtype == torch.bfloat16
    assert "model.layers.0.mlp.down_proj.weight_scale" in result


@pytest.mark.unit
def test_validate_raises_on_untargeted_scale():
    # A module that is neither targeted nor ignored but carries a qparam is an
    # orphan and must raise.
    dequantizer = _create_dequantizer(ignore=["model.embed_tokens"])
    tensors = _create_dummy_tensors(device=torch.device("meta"))
    tensors["model.layers.0.self_attn.q_proj.weight_scale"] = torch.rand(
        128, 1, dtype=torch.float32, device="meta"
    )

    with pytest.raises(ValueError):
        dequantizer.validate(tensors)


def _create_static_fp8_dequantizer(ignore=None):
    dequantizer = object.__new__(CompressedTensorsDequantizer)
    dequantizer.dtype = torch.bfloat16
    dequantizer.quant_config = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                format=CompressionFormat.float_quantized,
                **FP8,
            )
        },
        ignore=ignore or [],
    )
    return dequantizer


def _create_static_fp8_tensors(device: torch.device = torch.device("cpu")):
    tensors = {}
    with device:
        for proj in ("q_proj", "o_proj"):
            prefix = f"model.layers.0.self_attn.{proj}"
            tensors[f"{prefix}.weight"] = torch.randn(64, 64).to(torch.float8_e4m3fn)
            tensors[f"{prefix}.weight_scale"] = torch.rand(1)
            tensors[f"{prefix}.input_scale"] = torch.rand(1)
    return tensors


@pytest.mark.unit
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("meta")])
def test_static_activation_qparams_are_dropped(device):
    dequantizer = _create_static_fp8_dequantizer()
    tensors = _create_static_fp8_tensors(device=device)

    result = dequantizer.validate(tensors)

    assert sorted(result) == [
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]
    assert result["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16


@pytest.mark.unit
def test_static_activation_qparams_of_ignored_module_are_kept():
    dequantizer = _create_static_fp8_dequantizer(ignore=["re:.*o_proj"])
    tensors = _create_static_fp8_tensors(device=torch.device("meta"))

    result = dequantizer.validate(tensors)

    assert "model.layers.0.self_attn.q_proj.input_scale" not in result
    assert "model.layers.0.self_attn.o_proj.weight_scale" in result
    assert "model.layers.0.self_attn.o_proj.input_scale" in result


@pytest.mark.unit
def test_validate_returns_correct_key_count():
    dequantizer = _create_dequantizer(ignore=["model.embed_tokens"])
    tensors = _create_dummy_tensors(device=torch.device("meta"))

    result = dequantizer.validate(tensors)

    # Input: 9 keys
    #   4 targeted (2 weights + 2 weight_scales) -> consumed
    #   5 untargeted (q_proj, 3 layernorms, embed_tokens) -> passthrough
    # Output: 2 dequantized weights + 5 passthrough = 7 keys
    assert len(result) == 7


@pytest.mark.unit
def test_update_config_returns_none():
    dequantizer = _create_dequantizer()

    # Dequantizer always returns None (strips quantization)
    assert dequantizer.update_config(None) is None

    # Even with an existing config, dequantizer returns None
    existing = QuantizationConfig(
        config_groups={
            "g": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.CHANNEL,
                    symmetric=True,
                    dynamic=False,
                ),
            )
        },
    )
    assert dequantizer.update_config(existing) is None


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
