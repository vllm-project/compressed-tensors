# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import transformers
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    apply_quantization_config,
)


def _gemma4_config(num_global_key_value_heads, num_kv_shared_layers=0):
    """
    Tiny gemma-4 whose full-attention layers have a larger head_dim (and, when
    `num_global_key_value_heads` is set, fewer key/value heads) than its sliding
    layers, like gemma-4 E2B/E4B (head_dim only) and 26B/31B (both)
    """
    if not hasattr(transformers, "Gemma4TextConfig"):
        pytest.skip("gemma-4 requires a newer transformers")

    config = transformers.Gemma4TextConfig(
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        global_head_dim=32,
        num_global_key_value_heads=num_global_key_value_heads,
        attention_k_eq_v=num_global_key_value_heads is not None,
        num_hidden_layers=4,
        layer_types=["sliding_attention", "full_attention"] * 2,
        intermediate_size=64,
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=num_kv_shared_layers,
    )
    if not getattr(config, "is_heterogeneous", False):
        pytest.skip("gemma-4 per-layer configs require transformers>=5.15")
    return config


def _kv_cache_scheme(strategy):
    return QuantizationArgs(
        num_bits=8, type="float", strategy=strategy, symmetric=True, dynamic=False
    )


def _kv_attention_modules(model):
    return [layer.self_attn for layer in model.model.layers]


@pytest.mark.parametrize(
    "num_global_key_value_heads,num_kv_shared_layers",
    [(None, 0), (None, 2), (2, 0)],
)
@pytest.mark.parametrize("strategy", ["tensor", "attn_head"])
def test_kv_cache_scales_follow_each_layer(
    num_global_key_value_heads, num_kv_shared_layers, strategy
):
    config = _gemma4_config(num_global_key_value_heads, num_kv_shared_layers)
    model = transformers.Gemma4ForCausalLM(config).eval()
    apply_quantization_config(
        model,
        QuantizationConfig(
            config_groups={}, kv_cache_scheme=_kv_cache_scheme(strategy)
        ),
    )

    for attention in _kv_attention_modules(model):
        if attention.is_kv_shared_layer:
            # reuses the kv cache of an earlier layer, so has no scales of its own
            assert not hasattr(attention, "k_scale")
            continue

        num_kv_heads = attention.k_proj.out_features // attention.head_dim
        expected_shape = (1,) if strategy == "tensor" else (num_kv_heads, 1, 1)
        assert attention.k_scale.shape == expected_shape
        assert attention.v_scale.shape == expected_shape

        attention.k_scale.data.fill_(1.0)
        attention.v_scale.data.fill_(1.0)

    # quantized kv cache states keep the head_dim and kv heads of each layer
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize(
    "num_global_key_value_heads,num_kv_shared_layers",
    [(None, 0), (None, 2), (2, 0)],
)
@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
def test_attention_scales_follow_each_layer(
    num_global_key_value_heads, num_kv_shared_layers, attn_implementation
):
    config = _gemma4_config(num_global_key_value_heads, num_kv_shared_layers)
    model = transformers.Gemma4ForCausalLM(config).eval()
    model.set_attn_implementation(attn_implementation)
    scheme = QuantizationScheme(
        targets=["Gemma4TextAttention"], input_activations=_kv_cache_scheme("attn_head")
    )
    apply_quantization_config(
        model, QuantizationConfig(config_groups={"attention": scheme})
    )

    for attention in _kv_attention_modules(model):
        if attention.is_kv_shared_layer:
            # has no key/value projections of its own, so is not quantized
            assert not hasattr(attention, "q_scale")
            continue

        num_kv_heads = attention.k_proj.out_features // attention.head_dim
        assert attention.q_scale.shape == (config.num_attention_heads, 1, 1)
        assert attention.k_scale.shape == (num_kv_heads, 1, 1)
        assert attention.v_scale.shape == (num_kv_heads, 1, 1)

        for name in ("q", "k", "v"):
            getattr(attention, f"{name}_scale").data.fill_(1.0)

    # kv-shared layers keep the original attention implementation
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    assert torch.isfinite(logits).all()
