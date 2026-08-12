# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

import pytest
import torch
from compressed_tensors.entrypoints.convert.converters.magnitude_expert_prune import (
    MagnitudeExpertPrune,
    _build_expert_to_router_map,
    _common_prefix_len,
    _compute_retained_experts,
    _extract_expert_index,
    _renumber_expert_name,
)
from safetensors.torch import save_file


@pytest.mark.unit
def test_extract_expert_index():
    assert _extract_expert_index("model.layers.0.mlp.experts.3.gate_proj.weight") == 3
    assert _extract_expert_index("model.layers.0.mlp.experts.0.up_proj.weight") == 0
    assert (
        _extract_expert_index("model.layers.0.block_sparse_moe.experts.7.w1.weight")
        == 7
    )
    assert _extract_expert_index("model.layers.0.mlp.gate.weight") is None


@pytest.mark.unit
def test_renumber_expert_name():
    name = "model.layers.0.mlp.experts.5.gate_proj.weight"
    assert (
        _renumber_expert_name(name, 5, 2)
        == "model.layers.0.mlp.experts.2.gate_proj.weight"
    )
    assert (
        _renumber_expert_name(name, 5, 0)
        == "model.layers.0.mlp.experts.0.gate_proj.weight"
    )


@pytest.mark.unit
def test_common_prefix_len():
    assert _common_prefix_len("a.b.c.d", "a.b.c.e") == 3
    assert _common_prefix_len("a.b.c", "a.b.c") == 3
    assert _common_prefix_len("a.b", "c.d") == 0


@pytest.mark.unit
def test_build_expert_to_router_map():
    expert_names = [
        "model.layers.0.mlp.experts.0.weight",
        "model.layers.0.mlp.experts.1.weight",
        "model.layers.1.mlp.experts.0.weight",
    ]
    router_names = [
        "model.layers.0.mlp.gate.weight",
        "model.layers.1.mlp.gate.weight",
    ]
    mapping = _build_expert_to_router_map(expert_names, router_names)
    assert mapping["model.layers.0.mlp.experts.0.weight"] == (
        "model.layers.0.mlp.gate.weight"
    )
    assert mapping["model.layers.0.mlp.experts.1.weight"] == (
        "model.layers.0.mlp.gate.weight"
    )
    assert mapping["model.layers.1.mlp.experts.0.weight"] == (
        "model.layers.1.mlp.gate.weight"
    )


@pytest.fixture
def mock_2d_checkpoint(tmp_path):
    """Create a mock 2D MoE checkpoint with 4 experts, 1 layer."""
    num_experts = 4
    hidden = 8
    expert_dim = 16

    router_weight = torch.zeros(num_experts, hidden)
    router_weight[0] = torch.ones(hidden) * 10.0
    router_weight[1] = torch.ones(hidden) * 1.0
    router_weight[2] = torch.ones(hidden) * 5.0
    router_weight[3] = torch.ones(hidden) * 8.0

    tensors = {"model.layers.0.mlp.gate.weight": router_weight}
    for i in range(num_experts):
        tensors[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.randn(
            expert_dim, hidden
        )
        tensors[f"model.layers.0.mlp.experts.{i}.up_proj.weight"] = torch.randn(
            expert_dim, hidden
        )

    tensors["model.embed_tokens.weight"] = torch.randn(100, hidden)

    save_file(tensors, str(tmp_path / "model.safetensors"))

    config = {"num_local_experts": num_experts, "model_type": "test"}
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    return tmp_path


@pytest.fixture
def mock_3d_checkpoint(tmp_path):
    """Create a mock 3D MoE checkpoint with stacked expert weights."""
    num_experts = 4
    hidden = 8
    expert_dim = 16

    router_weight = torch.zeros(num_experts, hidden)
    router_weight[0] = torch.ones(hidden) * 10.0
    router_weight[1] = torch.ones(hidden) * 1.0
    router_weight[2] = torch.ones(hidden) * 5.0
    router_weight[3] = torch.ones(hidden) * 8.0

    tensors = {
        "model.layers.0.mlp.gate.weight": router_weight,
        "model.layers.0.mlp.experts.gate_proj.weight": torch.randn(
            num_experts, expert_dim, hidden
        ),
        "model.layers.0.mlp.experts.up_proj.weight": torch.randn(
            num_experts, expert_dim, hidden
        ),
        "model.embed_tokens.weight": torch.randn(100, hidden),
    }

    save_file(tensors, str(tmp_path / "model.safetensors"))

    config = {"num_local_experts": num_experts, "model_type": "test"}
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    return tmp_path


@pytest.mark.unit
def test_from_pretrained_2d(mock_2d_checkpoint):
    converter = MagnitudeExpertPrune.from_pretrained(
        model_name_or_path=str(mock_2d_checkpoint),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\d+\.",
        k=2,
    )

    assert converter.k == 2
    assert not converter.is_3d
    assert converter.num_experts_config_key == "num_local_experts"

    # scores: expert 0 = 80, expert 3 = 64, expert 2 = 40, expert 1 = 8
    # top-2 should be [0, 3]
    retained = converter.retained_experts["model.layers.0.mlp.gate.weight"]
    assert retained == [0, 3]


@pytest.mark.unit
def test_process_2d(mock_2d_checkpoint):
    converter = MagnitudeExpertPrune.from_pretrained(
        model_name_or_path=str(mock_2d_checkpoint),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\d+\.",
        k=2,
    )

    from safetensors import safe_open

    with safe_open(str(mock_2d_checkpoint / "model.safetensors"), framework="pt") as f:
        tensors = {name: f.get_tensor(name) for name in f.keys()}

    original_expert_0_gate = tensors[
        "model.layers.0.mlp.experts.0.gate_proj.weight"
    ].clone()
    original_expert_3_gate = tensors[
        "model.layers.0.mlp.experts.3.gate_proj.weight"
    ].clone()

    result = converter.process(tensors)

    # should have: gate weight (2 rows), 2 experts x 2 params, embed_tokens
    assert "model.layers.0.mlp.gate.weight" in result
    assert result["model.layers.0.mlp.gate.weight"].shape[0] == 2

    # expert 0 stays as 0, expert 3 becomes 1
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in result
    assert "model.layers.0.mlp.experts.1.gate_proj.weight" in result
    assert "model.layers.0.mlp.experts.2.gate_proj.weight" not in result
    assert "model.layers.0.mlp.experts.3.gate_proj.weight" not in result

    assert torch.equal(
        result["model.layers.0.mlp.experts.0.gate_proj.weight"], original_expert_0_gate
    )
    assert torch.equal(
        result["model.layers.0.mlp.experts.1.gate_proj.weight"], original_expert_3_gate
    )

    # non-expert tensors pass through
    assert "model.embed_tokens.weight" in result


@pytest.mark.unit
def test_process_3d(mock_3d_checkpoint):
    converter = MagnitudeExpertPrune.from_pretrained(
        model_name_or_path=str(mock_3d_checkpoint),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\w+\.weight$",
        k=2,
    )

    assert converter.is_3d

    from safetensors import safe_open

    with safe_open(str(mock_3d_checkpoint / "model.safetensors"), framework="pt") as f:
        tensors = {name: f.get_tensor(name) for name in f.keys()}

    original_gate_proj = tensors[
        "model.layers.0.mlp.experts.gate_proj.weight"
    ].clone()

    result = converter.process(tensors)

    # 3D expert tensors should be sliced to k=2 on dim 0
    assert result["model.layers.0.mlp.experts.gate_proj.weight"].shape[0] == 2
    retained = converter.retained_experts["model.layers.0.mlp.gate.weight"]
    expected = original_gate_proj[torch.tensor(retained)]
    assert torch.equal(
        result["model.layers.0.mlp.experts.gate_proj.weight"], expected
    )

    # router sliced
    assert result["model.layers.0.mlp.gate.weight"].shape[0] == 2


@pytest.mark.unit
def test_update_model_config(mock_2d_checkpoint):
    converter = MagnitudeExpertPrune.from_pretrained(
        model_name_or_path=str(mock_2d_checkpoint),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\d+\.",
        k=2,
    )

    converter.update_model_config(mock_2d_checkpoint)

    with open(mock_2d_checkpoint / "config.json") as f:
        config = json.load(f)

    assert config["num_local_experts"] == 2


@pytest.mark.unit
def test_update_model_config_text_config(tmp_path):
    """Config with text_config wrapper (multimodal models)."""
    num_experts = 4
    hidden = 8

    router_weight = torch.ones(num_experts, hidden)
    tensors = {
        "model.layers.0.mlp.gate.weight": router_weight,
        "model.layers.0.mlp.experts.gate_proj.weight": torch.randn(
            num_experts, 16, hidden
        ),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))

    config = {
        "model_type": "test",
        "text_config": {"num_local_experts": num_experts},
    }
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    converter = MagnitudeExpertPrune.from_pretrained(
        model_name_or_path=str(tmp_path),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\w+\.weight$",
        k=2,
    )
    converter.update_model_config(tmp_path)

    with open(tmp_path / "config.json") as f:
        config = json.load(f)
    assert config["text_config"]["num_local_experts"] == 2


@pytest.mark.unit
def test_validate_passes(mock_2d_checkpoint):
    converter = MagnitudeExpertPrune.from_pretrained(
        model_name_or_path=str(mock_2d_checkpoint),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\d+\.",
        k=2,
    )

    from safetensors import safe_open

    with safe_open(str(mock_2d_checkpoint / "model.safetensors"), framework="pt") as f:
        tensors = {name: f.get_tensor(name) for name in f.keys()}

    converter.validate(tensors)


@pytest.mark.unit
def test_validate_fails_unknown_router(mock_2d_checkpoint):
    converter = MagnitudeExpertPrune.from_pretrained(
        model_name_or_path=str(mock_2d_checkpoint),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\d+\.",
        k=2,
    )

    tensors = {"model.layers.99.mlp.gate.weight": torch.randn(4, 8)}
    with pytest.raises(ValueError, match="not found in pre-computed"):
        converter.validate(tensors)
