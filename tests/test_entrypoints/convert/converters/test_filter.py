# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.entrypoints.convert import FilterConverter


@pytest.mark.unit
def test_filter_removes_targeted_tensors():
    """
    Test that process removes tensors matching target patterns and leaves
    non-targeted tensors untouched.
    """
    converter = FilterConverter(targets=["model.layers.0.mlp.up_proj"])

    keep_weight = torch.randn(128, 128)
    tensors = {
        "model.layers.0.mlp.up_proj.weight": torch.randn(256, 256),
        "model.layers.0.mlp.up_proj.scale": torch.randn(2, 2),
        "model.layers.1.mlp.up_proj.weight": keep_weight,
    }

    converter.process(tensors)

    assert "model.layers.0.mlp.up_proj.weight" not in tensors
    assert "model.layers.0.mlp.up_proj.scale" not in tensors
    assert torch.equal(tensors["model.layers.1.mlp.up_proj.weight"], keep_weight)


@pytest.mark.unit
def test_filter_with_regex_targets():
    """
    Test that regex targets (re: prefix) correctly match tensor names.
    """
    converter = FilterConverter(targets=[r"re:.*\.mlp\..*proj$"])

    tensors = {
        "model.layers.0.mlp.up_proj.weight": torch.randn(256, 256),
        "model.layers.1.mlp.down_proj.scale": torch.randn(2, 2),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
        "model.embed_tokens.weight": torch.randn(1000, 128),
    }

    converter.process(tensors)

    assert "model.layers.0.mlp.up_proj.weight" not in tensors
    assert "model.layers.1.mlp.down_proj.scale" not in tensors
    assert "model.layers.0.self_attn.q_proj.weight" in tensors
    assert "model.embed_tokens.weight" in tensors


@pytest.mark.unit
def test_filter_with_ignore():
    """
    Test that tensors matching ignore patterns are preserved even when they
    also match targets.
    """
    converter = FilterConverter(
        targets=[r"re:.*\.mlp\..*"],
        ignore=["model.layers.0.mlp.up_proj"],
    )

    kept_weight = torch.randn(256, 256)
    tensors = {
        "model.layers.0.mlp.up_proj.weight": kept_weight,
        "model.layers.0.mlp.down_proj.weight": torch.randn(256, 256),
        "model.layers.1.mlp.up_proj.weight": torch.randn(256, 256),
    }

    converter.process(tensors)

    assert torch.equal(tensors["model.layers.0.mlp.up_proj.weight"], kept_weight)
    assert "model.layers.0.mlp.down_proj.weight" not in tensors
    assert "model.layers.1.mlp.up_proj.weight" not in tensors


@pytest.mark.unit
def test_filter_empty_targets_removes_all():
    """
    Test that an empty targets list matches all tensors, consistent with
    match_quantizable_tensors treating empty targets as all-inclusive.
    """
    converter = FilterConverter(targets=[])

    tensors = {
        "model.layers.0.mlp.up_proj.weight": torch.randn(256, 256),
        "model.embed_tokens.weight": torch.randn(1000, 128),
    }

    converter.process(tensors)

    assert len(tensors) == 0


@pytest.mark.unit
def test_filter_create_config_returns_none():
    converter = FilterConverter(targets=["model.layers.0.mlp.up_proj"])
    assert converter.create_config() is None


@pytest.mark.unit
def test_filter_get_dependencies_returns_empty():
    converter = FilterConverter(targets=["model.layers.0.mlp.up_proj"])
    assert converter.get_dependencies("model.layers.0.mlp.up_proj.weight") == set()
