# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from compressed_tensors.entrypoints.convert.converters.magnitude_expert_pruner import (
    MagnitudeExpertPruner,
)
from compressed_tensors.utils.moe import (
    build_expert_to_router_map,
    common_prefix_len,
    extract_expert_index,
    renumber_expert_name,
)
from safetensors.torch import save_file


@pytest.mark.unit
def test_extract_expert_index():
    assert extract_expert_index("model.layers.0.mlp.experts.3.gate_proj.weight") == 3
    assert extract_expert_index("model.layers.0.mlp.experts.0.up_proj.weight") == 0
    assert (
        extract_expert_index("model.layers.0.block_sparse_moe.experts.7.w1.weight") == 7
    )
    assert extract_expert_index("model.layers.0.mlp.gate.weight") is None


@pytest.mark.unit
def test_renumber_expert_name():
    name = "model.layers.0.mlp.experts.5.gate_proj.weight"
    assert (
        renumber_expert_name(name, 5, 2)
        == "model.layers.0.mlp.experts.2.gate_proj.weight"
    )
    assert (
        renumber_expert_name(name, 5, 0)
        == "model.layers.0.mlp.experts.0.gate_proj.weight"
    )


@pytest.mark.unit
def test_common_prefix_len():
    assert common_prefix_len("a.b.c.d", "a.b.c.e") == 3
    assert common_prefix_len("a.b.c", "a.b.c") == 3
    assert common_prefix_len("a.b", "c.d") == 0


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
    mapping = build_expert_to_router_map(expert_names, router_names)
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
    converter = MagnitudeExpertPruner.from_pretrained(
        model_name_or_path=str(mock_2d_checkpoint),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\d+\.",
        sparsity=0.5,
    )

    assert converter.sparsity == 0.5
    assert not converter.is_3d
    assert converter.num_experts_config_key == "num_local_experts"

    # scores: expert 0 = 80, expert 3 = 64, expert 2 = 40, expert 1 = 8
    # top-2 should be [0, 3]
    retained = converter.retained_experts["model.layers.0.mlp.gate.weight"]
    assert retained == [0, 3]


@pytest.mark.unit
def test_process_2d(mock_2d_checkpoint):
    converter = MagnitudeExpertPruner.from_pretrained(
        model_name_or_path=str(mock_2d_checkpoint),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\d+\.",
        sparsity=0.5,
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
    converter = MagnitudeExpertPruner.from_pretrained(
        model_name_or_path=str(mock_3d_checkpoint),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\w+\.weight$",
        sparsity=0.5,
    )

    assert converter.is_3d

    from safetensors import safe_open

    with safe_open(str(mock_3d_checkpoint / "model.safetensors"), framework="pt") as f:
        tensors = {name: f.get_tensor(name) for name in f.keys()}

    original_gate_proj = tensors["model.layers.0.mlp.experts.gate_proj.weight"].clone()

    result = converter.process(tensors)

    # 3D expert tensors should be sliced to k=2 on dim 0
    assert result["model.layers.0.mlp.experts.gate_proj.weight"].shape[0] == 2
    retained = converter.retained_experts["model.layers.0.mlp.gate.weight"]
    expected = original_gate_proj[torch.tensor(retained)]
    assert torch.equal(result["model.layers.0.mlp.experts.gate_proj.weight"], expected)

    # router sliced
    assert result["model.layers.0.mlp.gate.weight"].shape[0] == 2


@pytest.mark.unit
def test_update_model_config(mock_2d_checkpoint):
    converter = MagnitudeExpertPruner.from_pretrained(
        model_name_or_path=str(mock_2d_checkpoint),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\d+\.",
        sparsity=0.5,
    )

    with open(mock_2d_checkpoint / "config.json") as f:
        config = json.load(f)

    config = converter.update_model_config(config)

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

    converter = MagnitudeExpertPruner.from_pretrained(
        model_name_or_path=str(tmp_path),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\w+\.weight$",
        sparsity=0.5,
    )
    with open(tmp_path / "config.json") as f:
        config = json.load(f)

    config = converter.update_model_config(config)
    assert config["text_config"]["num_local_experts"] == 2


@pytest.mark.unit
def test_validate_returns_meta_output(mock_2d_checkpoint):
    """
    The pruner inherits the base default ``validate`` (== ``process``). Under
    the chained-converter interface, ``validate`` must return the converted
    tensor dict so downstream converters see the output format, and it must be
    meta-safe (``validate_file`` loads tensors on the meta device).
    """
    converter = MagnitudeExpertPruner.from_pretrained(
        model_name_or_path=str(mock_2d_checkpoint),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\d+\.",
        sparsity=0.5,
    )

    from safetensors import safe_open

    with safe_open(str(mock_2d_checkpoint / "model.safetensors"), framework="pt") as f:
        tensors = {name: f.get_tensor(name) for name in f.keys()}

    # feed meta tensors, exactly as validate_file does under the 805 interface
    meta_tensors = {
        name: torch.empty(t.shape, dtype=t.dtype, device="meta")
        for name, t in tensors.items()
    }
    result = converter.validate(meta_tensors)

    # router sliced to k
    assert result["model.layers.0.mlp.gate.weight"].shape[0] == 2
    # retained experts re-indexed to 0..k-1, pruned experts dropped
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in result
    assert "model.layers.0.mlp.experts.1.gate_proj.weight" in result
    assert "model.layers.0.mlp.experts.2.gate_proj.weight" not in result
    assert "model.layers.0.mlp.experts.3.gate_proj.weight" not in result
    # non-expert tensors pass through
    assert "model.embed_tokens.weight" in result
    # meta-safe: output stays on the meta device
    assert all(t.device.type == "meta" for t in result.values())


@pytest.mark.unit
def test_update_config_passthrough(mock_2d_checkpoint):
    """
    Pruning is not (de)quantization, so ``update_config`` must return the
    incoming config unchanged rather than stripping it. This preserves any
    existing quantization config when the pruner is chained with other
    converters.
    """
    converter = MagnitudeExpertPruner.from_pretrained(
        model_name_or_path=str(mock_2d_checkpoint),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\d+\.",
        sparsity=0.5,
    )

    sentinel = object()
    assert converter.update_config(sentinel) is sentinel
    assert converter.update_config(None) is None


@pytest.fixture
def mock_8expert_checkpoint(tmp_path):
    """2D MoE checkpoint with 8 experts whose scores increase with index."""
    num_experts = 8
    hidden = 8

    # expert i has row-sum == i * hidden, so top-k are the highest indices
    router_weight = torch.stack(
        [torch.full((hidden,), float(i)) for i in range(num_experts)]
    )

    tensors = {"model.layers.0.mlp.gate.weight": router_weight}
    for i in range(num_experts):
        tensors[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.randn(
            4, hidden
        )

    save_file(tensors, str(tmp_path / "model.safetensors"))
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"num_experts": num_experts, "model_type": "test"}, f)

    return tmp_path


@pytest.mark.unit
@pytest.mark.parametrize(
    "sparsity,expected_kept",
    [
        (0.0, [0, 1, 2, 3, 4, 5, 6, 7]),  # prune none
        (0.25, [2, 3, 4, 5, 6, 7]),  # round(0.25*8)=2 pruned -> keep 6
        (0.5, [4, 5, 6, 7]),  # keep 4
        (1.0, [7]),  # clamp: never prune all, keep 1 (highest score)
    ],
)
def test_sparsity_resolution(mock_8expert_checkpoint, sparsity, expected_kept):
    converter = MagnitudeExpertPruner.from_pretrained(
        model_name_or_path=str(mock_8expert_checkpoint),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\d+\.",
        sparsity=sparsity,
    )
    retained = converter.retained_experts["model.layers.0.mlp.gate.weight"]
    assert retained == expected_kept


@pytest.mark.unit
@pytest.mark.parametrize("sparsity", [-0.1, 1.5, 2.0])
def test_from_pretrained_sparsity_out_of_range(mock_2d_checkpoint, sparsity):
    with pytest.raises(ValueError, match="sparsity must be in"):
        MagnitudeExpertPruner.from_pretrained(
            model_name_or_path=str(mock_2d_checkpoint),
            router_pattern=r"\.gate\.weight$",
            expert_pattern=r"\.experts\.\d+\.",
            sparsity=sparsity,
        )


@pytest.fixture
def mock_mixed_sign_checkpoint(tmp_path):
    """
    2D MoE checkpoint whose router rows have mixed signs, so ranking by signed
    row-sum and by L1 magnitude disagree. Rows (hidden=4):

        expert 0: [ 10, -10,  10, -10]  signed=0    |sum|=40
        expert 1: [  1,   1,   1,   1]  signed=4    |sum|=4
        expert 2: [  2,   2,   2,   2]  signed=8    |sum|=8
        expert 3: [ -3,  -3,  -3,  -3]  signed=-12  |sum|=12

    L1 magnitude ranks experts 0 and 3 highest; signed sum ranks 1 and 2.
    """
    router_weight = torch.tensor(
        [
            [10.0, -10.0, 10.0, -10.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [-3.0, -3.0, -3.0, -3.0],
        ]
    )
    num_experts, hidden = router_weight.shape

    tensors = {"model.layers.0.mlp.gate.weight": router_weight}
    for i in range(num_experts):
        tensors[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.randn(
            4, hidden
        )

    save_file(tensors, str(tmp_path / "model.safetensors"))
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"num_experts": num_experts, "model_type": "test"}, f)

    return tmp_path


@pytest.mark.unit
def test_scoring_uses_absolute_magnitude(mock_mixed_sign_checkpoint):
    """
    Experts must be ranked by L1 magnitude, not signed row-sum, so that positive
    and negative router weights do not cancel. With sparsity=0.5 (keep 2), L1
    retains experts {0, 3}; a signed-sum bug would instead keep {1, 2}.
    """
    converter = MagnitudeExpertPruner.from_pretrained(
        model_name_or_path=str(mock_mixed_sign_checkpoint),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\d+\.",
        sparsity=0.5,
    )
    retained = converter.retained_experts["model.layers.0.mlp.gate.weight"]
    assert retained == [0, 3]


@pytest.fixture
def mock_2d_checkpoint_with_per_tok(tmp_path):
    """2D MoE checkpoint with num_experts=4 and num_experts_per_tok=2 in config."""
    num_experts = 4
    hidden = 8
    router_weight = torch.ones(num_experts, hidden)
    tensors = {"model.layers.0.mlp.gate.weight": router_weight}
    for i in range(num_experts):
        tensors[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.randn(
            16, hidden
        )
    tensors["model.embed_tokens.weight"] = torch.randn(100, hidden)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"num_local_experts": num_experts, "num_experts_per_tok": 2}, f)
    return tmp_path


@pytest.mark.unit
def test_sparsity_respects_num_experts_per_tok(mock_2d_checkpoint_with_per_tok):
    """
    from_pretrained must raise when the requested sparsity would retain fewer
    experts per layer than num_experts_per_tok, since the model cannot route
    each token to more experts than exist.
    """
    with pytest.raises(ValueError, match="per-token routing floor"):
        MagnitudeExpertPruner.from_pretrained(
            model_name_or_path=str(mock_2d_checkpoint_with_per_tok),
            router_pattern=r"\.gate\.weight$",
            expert_pattern=r"\.experts\.\d+\.",
            sparsity=0.75,  # retains max(1, 4 - round(0.75*4)) = max(1,1) = 1 < 2
        )


@pytest.mark.unit
def test_sparsity_at_num_experts_per_tok_floor_passes(mock_2d_checkpoint_with_per_tok):
    """Retaining exactly num_experts_per_tok experts should not raise."""
    converter = MagnitudeExpertPruner.from_pretrained(
        model_name_or_path=str(mock_2d_checkpoint_with_per_tok),
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\d+\.",
        sparsity=0.5,  # retains max(1, 4 - round(0.5*4)) = 2 == num_experts_per_tok
    )
    k = len(converter.retained_experts["model.layers.0.mlp.gate.weight"])
    assert k == 2
    assert converter.num_experts_per_tok == 2


@pytest.mark.unit
def test_update_model_config_non_uniform(tmp_path):
    """
    ``config.json`` holds a single expert count, so pruning must resolve to the
    same keep-count across every router. Mismatched counts should raise.
    """
    converter = MagnitudeExpertPruner(
        router_pattern=r"\.gate\.weight$",
        expert_pattern=r"\.experts\.\d+\.",
        sparsity=0.5,
        num_experts_config_key="num_experts",
        retained_experts={
            "model.layers.0.mlp.gate.weight": [0, 1],
            "model.layers.1.mlp.gate.weight": [0, 1, 2],
        },
        expert_to_router={},
        expert_indices={},
        is_3d=False,
    )

    with pytest.raises(ValueError, match="non-uniform"):
        converter.update_model_config({"num_experts": 4, "model_type": "test"})
