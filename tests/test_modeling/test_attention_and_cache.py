# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from functools import wraps

import pytest
import torch
import transformers
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
from compressed_tensors.offload import offload_module
from compressed_tensors.quantization import apply_quantization_config
from tests.test_modeling.test_per_layer_attention import (
    attention_modules,
    logits,
    quantization,
    snapshot,
    tiny_model,
    unchanged,
)
from tests.testing_utils import requires_gpu
from transformers import AutoModelForCausalLM
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import LlamaAttention


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


@pytest.mark.parametrize(
    "dynamic,partial", [(False, False), (True, False), (True, True)]
)
def test_missing_intended_attention_impl_raises(dynamic, partial):
    model = tiny_model(heterogeneous=False)
    config = quantization(model, dynamic=dynamic)
    if partial:
        config.config_groups["attention"].targets = ["model.layers.0.self_attn"]
    apply_quantization_config(model, config)
    if not partial:
        delattr(attention_modules(model)[-1], IMPL_ATTR)
    with pytest.raises(ValueError, match="impl"):
        logits(model)


class InheritedAttention(LlamaAttention):
    pass


class UnwrappedAttention(LlamaAttention):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class DelegatingAttention(LlamaAttention):
    @wraps(LlamaAttention.forward)
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize(
    "attention_class,offload",
    [
        (LlamaAttention, False),
        (InheritedAttention, False),
        (UnwrappedAttention, False),
        (DelegatingAttention, False),
        (DelegatingAttention, True),
    ],
)
def test_native_callback_and_reinitialization(backend, attention_class, offload):
    model = tiny_model(heterogeneous=False, backend=backend)
    reference = logits(model)
    for index, layer in enumerate(model.model.layers):
        replacement = attention_class(model.config, layer_idx=index)
        replacement.load_state_dict(layer.self_attn.state_dict())
        layer.self_attn = replacement
        if offload:
            offload_module(replacement, "cpu", "cpu")
    calls = []
    for module in attention_modules(model):
        initialize_hooked_attention(model, module)
        register_query_hook(module, lambda *_: calls.append(1))
        impl = module.impl
        initialize_hooked_attention(model, module)
        assert module.impl is impl
    torch.testing.assert_close(logits(model), reference, atol=0, rtol=0)
    assert len(calls) == 4


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_replacement_and_interleaved_models_preserve_native_attention(backend):
    model = tiny_model(heterogeneous=False, backend=backend)
    reference = logits(model)
    for module in attention_modules(model):
        initialize_hooked_attention(model, module)
    # A second model must not overwrite the first model's native callback/mask.
    other = tiny_model(
        heterogeneous=False, backend="sdpa" if backend == "eager" else "eager"
    )
    other_reference = logits(other)
    for module in attention_modules(other):
        initialize_hooked_attention(other, module)
    replacement = LlamaAttention(model.config, layer_idx=0)
    replacement.load_state_dict(attention_modules(model)[0].state_dict())
    model.model.layers[0].self_attn = replacement
    initialize_hooked_attention(model, replacement)
    torch.testing.assert_close(logits(model), reference, atol=0, rtol=0)
    torch.testing.assert_close(logits(other), other_reference, atol=0, rtol=0)


@pytest.mark.parametrize(
    "text_backend,vision_backend", [("eager", "sdpa"), ("sdpa", "eager")]
)
def test_nonquantizable_vision_retains_native_attention_and_mask(
    text_backend, vision_backend
):
    torch.manual_seed(1234)
    model = transformers.LlavaForConditionalGeneration(
        transformers.LlavaConfig(
            vision_config=transformers.CLIPVisionConfig(
                hidden_size=32,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                image_size=8,
                patch_size=4,
            ),
            text_config=transformers.LlamaConfig(
                hidden_size=32,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=32,
            ),
            image_token_index=30,
            vision_feature_layer=-1,
        )
    ).eval()
    model.set_attn_implementation(
        {"text_config": text_backend, "vision_config": vision_backend}
    )
    inputs = dict(
        input_ids=torch.tensor([[30, 30, 30, 30, 1, 3, 7]]),
        pixel_values=torch.randn(1, 3, 8, 8),
    )
    with torch.no_grad():
        reference = model(**inputs, use_cache=False).logits
    configs = [model.config.text_config, model.config.vision_config]
    masks = [ALL_MASK_ATTENTION_FUNCTIONS.get(c._attn_implementation) for c in configs]
    calls = []
    for layer in model.model.language_model.layers:
        initialize_hooked_attention(model, layer.self_attn)
        register_query_hook(layer.self_attn, lambda *_: calls.append(1))
    assert [
        ALL_MASK_ATTENTION_FUNCTIONS.get(c._attn_implementation) for c in configs
    ] == masks
    with torch.no_grad():
        actual = model(**inputs, use_cache=False).logits
        changed = deepcopy(inputs)
        changed["input_ids"][0, -1] = 2
        causal = model(**changed, use_cache=False).logits
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(causal[:, :-1], actual[:, :-1], atol=0, rtol=0)
    assert len(calls) == 4


def test_unresolvable_selected_callback_does_not_mutate_model():
    model = tiny_model(heterogeneous=False)

    class InlineAttention(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    model.model.layers[-1].self_attn = InlineAttention()
    before = snapshot(model)
    with pytest.raises(ValueError, match="Cannot resolve original attention"):
        initialize_hooked_attention(model, model.model.layers[-1].self_attn)
    unchanged(model, before)
