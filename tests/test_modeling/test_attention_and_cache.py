# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM


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


def test_attention_quantization_of_some_layers():
    # layers whose attention is not quantized keep the original implementation
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
    )
    model = LlamaForCausalLM(config).eval()
    model.set_attn_implementation("sdpa")
    args = QuantizationArgs(
        num_bits=8, type="float", strategy="tensor", symmetric=True, dynamic=False
    )
    scheme = QuantizationScheme(
        targets=["re:.*layers\\.0\\.self_attn$"], input_activations=args
    )
    apply_quantization_config(
        model, QuantizationConfig(config_groups={"attention": scheme})
    )

    quantized, unquantized = (layer.self_attn for layer in model.model.layers)
    assert hasattr(quantized, IMPL_ATTR)
    assert not hasattr(unquantized, IMPL_ATTR)
    for name in ("q", "k", "v"):
        getattr(quantized, f"{name}_scale").data.fill_(1.0)

    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    assert torch.isfinite(logits).all()
