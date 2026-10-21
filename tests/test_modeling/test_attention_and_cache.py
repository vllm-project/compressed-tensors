# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch
from compressed_tensors.modeling import (
    IMPL_ATTR,
    KV_CACHE_ATTR,
    QuantizedAttentionImpl,
    QuantizedKVCache,
    initialize_hooked_attention,
    initialize_hooked_kv_cache,
    register_key_hook,
    register_query_hook,
    register_value_hook,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    apply_quantization_config,
)
from tests.testing_utils import requires_gpu
from transformers import AutoConfig, AutoModelForCausalLM


@requires_gpu
def test_attention_cache():
    model = AutoModelForCausalLM.from_pretrained(
        "nm-testing/llama2.c-stories15M", device_map="cuda"
    )
    inputs = {key: value.to("cuda") for key, value in model.dummy_inputs.items()}
    true_outputs = model(**inputs)
    layers = model.model.layers

    # check if hooks work
    k_called = [False for _ in range(len(layers))]
    v_called = [False for _ in range(len(layers))]

    # apply kv cache quantization
    _apply_kv_cache(model, layers, k_called, v_called)

    # check kv cache quantization
    outputs = model(**inputs)
    assert torch.equal(outputs.logits, true_outputs.logits)
    assert all(k_called) and all(v_called)

    """ apply attention quantization after kv cache quantization """

    # check if hooks work
    q_called = [False for _ in range(len(layers))]
    k_called = [False for _ in range(len(layers))]
    v_called = [False for _ in range(len(layers))]

    # apply attention quantization
    _apply_attention(model, layers, q_called, k_called, v_called)

    # check attention quantization
    outputs = model(**inputs)
    assert torch.equal(outputs.logits, true_outputs.logits)
    assert all(q_called) and all(k_called) and all(v_called)


@pytest.mark.parametrize(
    "model_type,attn_implementation",
    [("llama", "eager"), ("llama", "sdpa"), ("gpt_oss", "eager")],
)
def test_attention_cache_attn_implementation(model_type, attn_implementation):
    # tiny model built from config; gpt_oss eager attention adds attention sinks
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
    )
    if model_type == "gpt_oss":
        kwargs.update(num_local_experts=2, num_experts_per_tok=1)
    try:
        config = AutoConfig.for_model(model_type, **kwargs)
    except ValueError:
        pytest.skip(f"{model_type} is not supported by this transformers version")

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation=attn_implementation
    )
    model.eval()
    inputs = {"input_ids": torch.randint(0, 128, (2, 8))}
    true_outputs = model(**inputs)
    layers = model.model.layers

    q_called = [False for _ in range(len(layers))]
    k_called = [False for _ in range(len(layers))]
    v_called = [False for _ in range(len(layers))]
    _apply_attention(model, layers, q_called, k_called, v_called)

    outputs = model(**inputs)
    assert torch.equal(outputs.logits, true_outputs.logits)
    assert all(q_called) and all(k_called) and all(v_called)


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_attention_quantization_attn_implementation(attn_implementation):
    config = AutoConfig.for_model(
        "llama",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation=attn_implementation
    )
    model.eval()
    inputs = {"input_ids": torch.randint(0, 128, (2, 8))}
    true_outputs = model(**inputs)

    scheme = QuantizationScheme(
        targets=["LlamaAttention"],
        input_activations=QuantizationArgs(
            num_bits=8, type="int", strategy="tensor", dynamic=True
        ),
    )
    apply_quantization_config(model, QuantizationConfig(config_groups={"a": scheme}))

    outputs = model(**inputs)
    assert not torch.equal(outputs.logits, true_outputs.logits)
    assert torch.allclose(outputs.logits, true_outputs.logits, atol=1e-2)


def test_eager_attention_without_eager_function():
    class Attention(torch.nn.Module):
        pass

    impl = QuantizedAttentionImpl(None)
    query = key = value = torch.zeros(1, 1, 1, 1)
    with patch.object(QuantizedAttentionImpl, "_original_impl", "eager"):
        with pytest.raises(ValueError, match="eager_attention_forward"):
            impl(Attention(), query, key, value, None)


def _apply_kv_cache(model, layers, k_called, v_called):
    for layer_index, layer in enumerate(layers):
        module = layer.self_attn
        initialize_hooked_kv_cache(model, module)
        assert isinstance(getattr(module, KV_CACHE_ATTR), QuantizedKVCache)

        # reapply is no-op
        initialize_hooked_kv_cache(model, module)

        def k_hook(_module, _states, layer_index=layer_index):  # NOTE: capture by value
            k_called[layer_index] = True

        def v_hook(_module, _states, layer_index=layer_index):
            my_index = layer_index
            v_called[my_index] = True

        register_key_hook(module, k_hook)
        register_value_hook(module, v_hook)


def _apply_attention(model, layers, q_called, k_called, v_called):
    for layer_index, layer in enumerate(layers):
        module = layer.self_attn
        initialize_hooked_attention(model, module)
        assert isinstance(getattr(module, IMPL_ATTR), QuantizedAttentionImpl)

        # reapply is no-op
        initialize_hooked_attention(model, module)

        def q_hook(_module, _states, layer_index=layer_index):
            q_called[layer_index] = True

        def k_hook(_module, _states, layer_index=layer_index):
            k_called[layer_index] = True

        def v_hook(_module, _states, layer_index=layer_index):
            v_called[layer_index] = True

        register_query_hook(module, q_hook)
        register_key_hook(module, k_hook)
        register_value_hook(module, v_hook)
