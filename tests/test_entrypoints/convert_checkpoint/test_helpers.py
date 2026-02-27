# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.entrypoints.convert_checkpoint.helpers import (
    match_quantizable_tensors,
)


@pytest.fixture
def sample_tensors():
    """Create a sample set of tensors mimicking a model's state dict."""
    return {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(128, 128),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(128, 128),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(256, 128),
        "model.layers.0.mlp.up_proj.weight": torch.randn(256, 128),
        "model.layers.0.mlp.down_proj.weight": torch.randn(128, 256),
        "model.layers.0.input_layernorm.weight": torch.randn(128),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(128),
        "model.embed_tokens.weight": torch.randn(32000, 128),
        "lm_head.weight": torch.randn(32000, 128),
        "model.layers.0.self_attn.q_proj.bias": torch.randn(128),
    }


@pytest.mark.parametrize(
    "ignore,targets,allow_nonquantizable,expected_names",
    [
        # Test case: basic matching without ignore or targets
        (
            [],
            [],
            False,
            {
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
                "model.embed_tokens.weight",
                "lm_head.weight",
            },
        ),
        # Test case: ignore attention layers
        (
            ["re:.*self_attn.*"],
            [],
            False,
            {
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
                "model.embed_tokens.weight",
                "lm_head.weight",
            },
        ),
        # Test case: ignore attention and mlp layers
        (
            ["re:.*self_attn.*", "re:.*mlp.*"],
            [],
            False,
            {
                "model.embed_tokens.weight",
                "lm_head.weight",
            },
        ),
        # Test case: target only mlp gate_proj and up_proj
        (
            [],
            ["re:.*mlp.*gate_proj", "re:.*mlp.*up_proj"],
            False,
            {
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
            },
        ),
        # Test case: empty targets (all-inclusive)
        (
            [],
            [],
            False,
            {
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
                "model.embed_tokens.weight",
                "lm_head.weight",
            },
        ),
        # Test case: Linear targets (all-inclusive)
        (
            [],
            ["Linear"],
            False,
            {
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
                "model.embed_tokens.weight",
                "lm_head.weight",
            },
        ),
        # Test case: allow_nonquantizable includes bias and layernorm
        (
            [],
            [],
            True,
            {
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.embed_tokens.weight",
                "lm_head.weight",
                "model.layers.0.self_attn.q_proj.bias",
            },
        ),
        # Test case: ignore takes precedence over targets
        (
            ["re:.*self_attn.*"],
            ["re:.*self_attn.*q_proj"],
            False,
            set(),
        ),
        # Test case: regex pattern matching all proj layers
        (
            [],
            ["re:.*proj$"],
            False,
            {
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
            },
        ),
    ],
    ids=[
        "basic_matching",
        "ignore_attention",
        "ignore_attention_and_mlp",
        "target_mlp_gate_up",
        "empty_targets",
        "linear_targets",
        "allow_nonquantizable",
        "ignore_precedence",
        "regex_all_proj",
    ],
)
def test_match_quantizable_tensors(
    sample_tensors, ignore, targets, allow_nonquantizable, expected_names
):
    """
    Parameterized test for match_quantizable_tensors function.

    Tests various combinations of ignore patterns, target patterns, and flags
    to verify that the function returns the expected set of tensor names.
    """
    matches = list(
        match_quantizable_tensors(
            sample_tensors,
            ignore=ignore,
            targets=targets,
            allow_nonquantizable=allow_nonquantizable,
        )
    )

    # Extract full names from results
    result_names = {full_name for _, full_name in matches}

    # Assert the result matches expected
    assert result_names == expected_names

    # Additionally verify all results are tuples with correct format
    for module_name, full_name in matches:
        assert isinstance(module_name, str)
        assert isinstance(full_name, str)
        # module_name should be full_name without the last component
        assert full_name.startswith(module_name)
        assert full_name.rsplit(".", 1)[0] == module_name
