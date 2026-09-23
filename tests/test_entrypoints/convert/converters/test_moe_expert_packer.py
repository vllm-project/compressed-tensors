# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from compressed_tensors.entrypoints.convert.converters.moe_expert_packer import (
    MoEExpertPacker,
)
from safetensors import safe_open
from safetensors.torch import save_file


NUM_EXPERTS = 4
OUT = 6
IN = 8
PROJS = ("gate_proj", "up_proj", "down_proj")
PARAM_SHAPES = {
    "weight_packed": (OUT, IN // 2),
    "weight_scale": (OUT, IN // 16 if IN // 16 else 1),
    "weight_global_scale": (1,),
    "input_global_scale": (1,),
}


def _expert_name(layer, expert, proj, param):
    return f"model.layers.{layer}.mlp.experts.{expert}.{proj}.{param}"


def _stacked_name(layer, proj, param):
    return f"model.layers.{layer}.mlp.experts.{proj}.{param}"


@pytest.fixture
def mock_2d_moe_checkpoint(tmp_path):
    """Mock 2D (linearized) NVFP4 MoE checkpoint: one tensor per expert."""
    tensors = {}
    for layer in (0, 1):
        for expert in range(NUM_EXPERTS):
            for proj in PROJS:
                for param, shape in PARAM_SHAPES.items():
                    dtype = torch.uint8 if param == "weight_packed" else torch.float32
                    # value encodes the expert index so stack ordering is checkable
                    tensors[_expert_name(layer, expert, proj, param)] = torch.full(
                        shape, float(expert + 1)
                    ).to(dtype)
        tensors[f"model.layers.{layer}.mlp.gate.weight"] = torch.randn(NUM_EXPERTS, IN)
    tensors["model.embed_tokens.weight"] = torch.randn(10, IN)

    save_file(tensors, str(tmp_path / "model.safetensors"))
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"num_experts": NUM_EXPERTS, "model_type": "test"}, f)
    return tmp_path


def _load(path):
    with safe_open(str(path / "model.safetensors"), framework="pt") as f:
        return {name: f.get_tensor(name) for name in f.keys()}


@pytest.mark.unit
def test_from_pretrained_groups(mock_2d_moe_checkpoint):
    conv = MoEExpertPacker.from_pretrained(str(mock_2d_moe_checkpoint))

    # 2 layers * 3 projections * 4 params
    assert len(conv.groups) == 2 * len(PROJS) * len(PARAM_SHAPES)

    # members are ordered by ascending expert index, with the index stripped
    members = conv.groups[_stacked_name(0, "gate_proj", "weight_packed")]
    assert members == [
        _expert_name(0, e, "gate_proj", "weight_packed") for e in range(NUM_EXPERTS)
    ]


@pytest.mark.unit
def test_from_pretrained_no_match_raises(mock_2d_moe_checkpoint):
    with pytest.raises(ValueError, match="No tensors matched"):
        MoEExpertPacker.from_pretrained(
            str(mock_2d_moe_checkpoint), expert_pattern=r"\.nonexistent\."
        )


@pytest.mark.unit
def test_get_dependencies_only_anchor(mock_2d_moe_checkpoint):
    conv = MoEExpertPacker.from_pretrained(str(mock_2d_moe_checkpoint))

    # the anchor (down_proj is plain-stacked) pulls in its other experts
    anchor = _expert_name(0, 0, "down_proj", "weight_packed")
    assert conv.get_dependencies(anchor) == {
        _expert_name(0, e, "down_proj", "weight_packed") for e in range(1, NUM_EXPERTS)
    }

    # non-anchor experts and unrelated tensors declare nothing
    non_anchor = _expert_name(0, 2, "down_proj", "weight_packed")
    assert conv.get_dependencies(non_anchor) == set()
    assert conv.get_dependencies("model.embed_tokens.weight") == set()


@pytest.mark.unit
def test_process_stacks_experts_in_order(mock_2d_moe_checkpoint):
    conv = MoEExpertPacker.from_pretrained(str(mock_2d_moe_checkpoint))
    result = conv.process(_load(mock_2d_moe_checkpoint))

    out = result[_stacked_name(0, "down_proj", "weight_packed")]
    assert out.shape == (NUM_EXPERTS, OUT, IN // 2)
    for e in range(NUM_EXPERTS):
        # expert e was filled with (e + 1); stacking must preserve order
        assert torch.all(out[e] == float(e + 1))

    # scalar per-expert qparam becomes [E, 1]
    assert result[_stacked_name(0, "down_proj", "weight_global_scale")].shape == (
        NUM_EXPERTS,
        1,
    )

    # sources are consumed; non-expert tensors pass through untouched
    assert _expert_name(0, 0, "down_proj", "weight_packed") not in result
    assert "model.embed_tokens.weight" in result
    assert "model.layers.0.mlp.gate.weight" in result


@pytest.mark.unit
def test_fuse_gate_up(mock_2d_moe_checkpoint):
    """gate and up fuse into gate_up_proj, concatenated along the output dim."""
    conv = MoEExpertPacker.from_pretrained(str(mock_2d_moe_checkpoint))
    assert conv.fuse_gate_up

    # the gate anchor pulls in every gate AND up expert tensor
    anchor = _expert_name(0, 0, "gate_proj", "weight_packed")
    assert conv.get_dependencies(anchor) == {
        _expert_name(0, e, "gate_proj", "weight_packed") for e in range(1, NUM_EXPERTS)
    } | {_expert_name(0, e, "up_proj", "weight_packed") for e in range(NUM_EXPERTS)}

    result = conv.process(_load(mock_2d_moe_checkpoint))

    out = result[_stacked_name(0, "gate_up_proj", "weight_packed")]
    assert out.shape == (NUM_EXPERTS, 2 * OUT, IN // 2)
    for e in range(NUM_EXPERTS):
        assert torch.all(out[e] == float(e + 1))

    # scalar per-expert qparams fuse to [E, 2] (gate, then up)
    assert result[_stacked_name(0, "gate_up_proj", "weight_global_scale")].shape == (
        NUM_EXPERTS,
        2,
    )
    # the separate gate_proj / up_proj outputs no longer exist
    assert _stacked_name(0, "gate_proj", "weight_packed") not in result
    assert _stacked_name(0, "up_proj", "weight_packed") not in result


@pytest.mark.unit
def test_no_fuse_keeps_projections_separate(mock_2d_moe_checkpoint):
    conv = MoEExpertPacker.from_pretrained(
        str(mock_2d_moe_checkpoint), fuse_gate_up=False
    )
    result = conv.process(_load(mock_2d_moe_checkpoint))

    # each projection stacks independently, no gate_up_proj output
    for proj in PROJS:
        assert result[_stacked_name(0, proj, "weight_packed")].shape == (
            NUM_EXPERTS,
            OUT,
            IN // 2,
        )
    assert _stacked_name(0, "gate_up_proj", "weight_packed") not in result


@pytest.mark.unit
def test_process_raises_on_missing_member(mock_2d_moe_checkpoint):
    conv = MoEExpertPacker.from_pretrained(str(mock_2d_moe_checkpoint))
    tensors = _load(mock_2d_moe_checkpoint)
    # drop one non-anchor member while keeping its anchor
    del tensors[_expert_name(0, 1, "gate_proj", "weight_packed")]

    with pytest.raises(ValueError, match="missing expert tensors"):
        conv.process(tensors)


@pytest.mark.unit
def test_validate_is_meta_safe(mock_2d_moe_checkpoint):
    conv = MoEExpertPacker.from_pretrained(str(mock_2d_moe_checkpoint))
    meta = {
        name: torch.empty(t.shape, dtype=t.dtype, device="meta")
        for name, t in _load(mock_2d_moe_checkpoint).items()
    }

    result = conv.validate(meta)

    fused = result[_stacked_name(0, "gate_up_proj", "weight_packed")]
    assert fused.shape == (NUM_EXPERTS, 2 * OUT, IN // 2)
    assert all(t.device.type == "meta" for t in result.values())


@pytest.mark.unit
def test_update_config_rewrites_expert_ignore(mock_2d_moe_checkpoint):
    from compressed_tensors.config import CompressionFormat
    from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
    from compressed_tensors.quantization.quant_scheme import NVFP4

    conv = MoEExpertPacker.from_pretrained(str(mock_2d_moe_checkpoint))

    assert conv.update_config(None) is None

    config = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                **NVFP4,
                targets=["Linear"],
                format=CompressionFormat.nvfp4_pack_quantized.value,
            )
        },
        ignore=[
            "lm_head",
            "model.layers.0.mlp.experts.0.gate_proj",
            "model.layers.0.mlp.experts.0.up_proj",
            "model.layers.0.mlp.experts.0.down_proj",
            "model.layers.1.mlp.experts.3.down_proj",
            "re:.*mlp\\.experts\\.\\d+\\.gate_proj",
        ],
    )

    out = conv.update_config(config)

    assert out is config  # mutated + returned
    assert out.ignore == [
        "lm_head",
        # all projections of layer 0 expert 0 collapse to one entry
        "model.layers.0.mlp.experts",
        "model.layers.1.mlp.experts",
        # regex entries are left untouched
        "re:.*mlp\\.experts\\.\\d+\\.gate_proj",
    ]


@pytest.mark.unit
def test_already_packed_raises(tmp_path):
    # a 3D weight_packed that still name-matches the per-expert pattern
    tensors = {
        "model.layers.0.mlp.experts.0.gate_proj.weight_packed": torch.zeros(
            NUM_EXPERTS, OUT, IN // 2, dtype=torch.uint8
        )
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"num_experts": NUM_EXPERTS, "model_type": "test"}, f)

    with pytest.raises(ValueError, match="already"):
        MoEExpertPacker.from_pretrained(str(tmp_path))
