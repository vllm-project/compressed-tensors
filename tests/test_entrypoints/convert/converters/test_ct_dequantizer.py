# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

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
from safetensors.torch import load_file, save_file


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


@pytest.mark.parametrize("ignore_attention", [False, True])
def test_checkpoint_dequantization_removes_only_attention_qparams(
    tmp_path, ignore_attention
):
    scheme = QuantizationScheme(
        targets=["model.mlp"],
        weights=QuantizationArgs(num_bits=8, strategy="channel"),
        input_activations=QuantizationArgs(num_bits=8, strategy="token", dynamic=True),
        format=CompressionFormat.int_quantized,
    )
    config = QuantizationConfig(config_groups={"weights": scheme})
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": config.model_dump(mode="json")})
    )
    weight = torch.arange(16, dtype=torch.int8).reshape(4, 4)
    tensors = {
        "model.mlp.weight": weight,
        "model.mlp.weight_scale": torch.full((4, 1), 0.25),
    }
    controls = {
        "model.attn.q_proj.weight": torch.ones(4, 4),
        "model.attn.q_shape": torch.tensor([2, 4]),
        "model.attn.k_shape": torch.tensor([2, 4]),
        "model.attn.v_shape": torch.tensor([2, 4]),
        "model.attn.native_q_scale": torch.tensor([3.0]),
        "native_v_scale": torch.tensor([7.0]),
    }
    tensors.update(controls)
    for base in ("q", "k", "v"):
        for suffix in ("global_scale", "scale", "zero_point"):
            tensors[f"model.attn.{base}_{suffix}"] = torch.ones(1)
    checkpoint = tmp_path / "model.safetensors"
    save_file(tensors, checkpoint)
    converter = CompressedTensorsDequantizer(
        tmp_path, ignore=["model.attn"] if ignore_attention else [], dtype=torch.float32
    )
    output = converter.validate(load_file(checkpoint))
    assert set(output) == {"model.mlp.weight", *controls}
    torch.testing.assert_close(
        output["model.mlp.weight"], weight.float() * 0.25, atol=0, rtol=0
    )
    for name, value in controls.items():
        assert torch.equal(output[name], value)


@pytest.mark.parametrize(
    "kind",
    [
        "online",
        "unknown_location",
        "unknown_online_field",
        "offline",
        "offline_annotation",
        "empty",
    ],
)
def test_checkpoint_dequantization_rejects_executable_transforms(tmp_path, kind):
    scheme = QuantizationScheme(
        targets=["model.mlp"],
        weights=QuantizationArgs(num_bits=8, strategy="channel"),
        input_activations=QuantizationArgs(num_bits=8, strategy="token", dynamic=True),
        format=CompressionFormat.int_quantized,
    )
    config = QuantizationConfig(config_groups={"weights": scheme})
    group = {
        "type": "hadamard",
        "apply": [{"targets": ["model.mlp"], "location": "input"}],
    }
    if kind.startswith("offline"):
        group["apply"][0]["location"] = "weight_input"
    if kind == "unknown_location":
        group["apply"][0]["location"] = "future_activation"
    if kind in ("unknown_online_field", "offline_annotation"):
        group["future_annotation"] = True
    data = config.model_dump(mode="json")
    data["transform_config"] = {
        "config_groups": {} if kind == "empty" else {"rotation": group}
    }
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": data}))
    checkpoint = tmp_path / "model.safetensors"
    tensors = {
        "model.mlp.weight": torch.ones(4, 4, dtype=torch.int8),
        "model.mlp.weight_scale": torch.ones(4, 1),
    }
    save_file(tensors, checkpoint)
    before = checkpoint.read_bytes()
    if kind in ("online", "unknown_location", "unknown_online_field"):
        with pytest.raises(ValueError, match="[Tt]ransform|online"):
            CompressedTensorsDequantizer(tmp_path)
        assert checkpoint.read_bytes() == before
    else:
        converter = CompressedTensorsDequantizer(tmp_path, dtype=torch.float32)
        output = converter.validate(load_file(checkpoint))
        assert set(output) == {"model.mlp.weight"}
        assert torch.equal(output["model.mlp.weight"], torch.ones(4, 4))
