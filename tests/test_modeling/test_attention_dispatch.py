# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from functools import wraps

import pytest
import torch
from compressed_tensors.modeling import initialize_hooked_attention, register_query_hook
from compressed_tensors.modeling.attention import IMPL_ATTR
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    apply_quantization_config,
    disable_quantization,
)
from tests.test_modeling.test_attention_and_cache import _tiny_llava
from transformers import LlamaConfig, LlamaForCausalLM, PretrainedConfig
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import LlamaAttention


class InheritedAttention(LlamaAttention):
    pass


class DelegatingAttention(LlamaAttention):
    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        return super().forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )


class WrappedAttention(LlamaAttention):
    @wraps(LlamaAttention.forward)
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


def _llama(backend="sdpa", attention_cls=LlamaAttention):
    torch.manual_seed(1234)
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
    )
    model = LlamaForCausalLM(config).eval()
    model.set_attn_implementation(backend)
    if attention_cls is not LlamaAttention:
        for index, layer in enumerate(model.model.layers):
            replacement = attention_cls(config, index)
            replacement.load_state_dict(layer.self_attn.state_dict())
            layer.self_attn = replacement.eval()
    return model


def _hook(model):
    for layer in model.model.layers:
        initialize_hooked_attention(model, layer.self_attn)


def _logits(model, **inputs):
    with torch.no_grad():
        return model(**inputs, use_cache=False).logits


@pytest.mark.parametrize("backend", ["sdpa", "eager"])
@pytest.mark.parametrize(
    "attention_cls",
    [LlamaAttention, InheritedAttention, DelegatingAttention, WrappedAttention],
)
def test_inherited_attention_callback_preserves_outputs(backend, attention_cls):
    model = _llama(backend, attention_cls)
    inputs = {"input_ids": torch.tensor([[1, 3, 7, 2]])}
    expected = _logits(model, **inputs)
    _hook(model)
    calls = []
    for layer in model.model.layers:
        register_query_hook(layer.self_attn, lambda module, query: calls.append(module))
    torch.testing.assert_close(_logits(model, **inputs), expected, rtol=0, atol=0)
    assert calls == [layer.self_attn for layer in model.model.layers]


@pytest.mark.parametrize("backend", ["sdpa", "eager"])
@pytest.mark.parametrize("mode", ["repeat", "replace", "second_model"])
def test_attention_dispatch_initialization_lifecycle(backend, mode):
    model = _llama(backend)
    inputs = {"input_ids": torch.tensor([[1, 3, 7, 2]])}
    expected = _logits(model, **inputs)
    _hook(model)
    if mode == "replace":
        old = model.model.layers[0].self_attn
        replacement = LlamaAttention(model.config, 0).eval()
        replacement.load_state_dict(old.state_dict(), strict=True)
        model.model.layers[0].self_attn = replacement
    elif mode == "second_model":
        other = LlamaForCausalLM(model.config).eval()
        other.load_state_dict(model.state_dict(), strict=True)
        model = other
    _hook(model)
    counts = [len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers]
    _hook(model)
    assert counts == [
        len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers
    ]
    torch.testing.assert_close(_logits(model, **inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize(
    "text_backend,vision_backend",
    [("eager", "sdpa"), ("sdpa", "eager"), ("sdpa", "sdpa")],
)
def test_component_attention_keeps_its_mask(text_backend, vision_backend, with_image):
    torch.manual_seed(1234)
    model, inputs = _tiny_llava()
    model.set_attn_implementation(
        {"text_config": text_backend, "vision_config": vision_backend}
    )
    if not with_image:
        inputs = {"input_ids": torch.tensor([[1, 3, 7, 2]])}
    expected = _logits(model, **inputs)
    original_masks = {
        name: ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]
        for name, config in [
            ("text_config", model.config.text_config),
            ("vision_config", model.config.vision_config),
        ]
    }
    args = QuantizationArgs(num_bits=8, type="int", strategy="tensor", dynamic=True)
    scheme = QuantizationScheme(
        targets=["re:.*language_model.*self_attn$"], input_activations=args
    )
    apply_quantization_config(
        model, QuantizationConfig(config_groups={"attention": scheme})
    )
    model.apply(disable_quantization)
    for name, original in original_masks.items():
        assert (
            ALL_MASK_ATTENTION_FUNCTIONS[
                getattr(model.config, name)._attn_implementation
            ]
            is original
        )
    actual = _logits(model, **inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    changed = dict(inputs, input_ids=inputs["input_ids"].clone())
    changed["input_ids"][0, -1] = 11
    changed_logits = _logits(model, **changed)
    torch.testing.assert_close(changed_logits[:, :-1], actual[:, :-1], rtol=0, atol=0)


def test_interleaved_models_keep_original_mask_and_attention():
    models = [_llama("sdpa"), _llama("eager")]
    inputs = {"input_ids": torch.tensor([[1, 3, 7, 2]])}
    expected = [_logits(model, **inputs) for model in models]
    masks = [
        ALL_MASK_ATTENTION_FUNCTIONS[model.config._attn_implementation]
        for model in models
    ]
    for model in models:
        _hook(model)
    for index in [0, 1, 0]:
        model = models[index]
        assert (
            ALL_MASK_ATTENTION_FUNCTIONS[model.config._attn_implementation]
            is masks[index]
        )
        torch.testing.assert_close(
            _logits(model, **inputs), expected[index], rtol=0, atol=0
        )


def test_delegating_attention_preserves_local_registry_override(monkeypatch):
    registry = type(modeling_llama.ALL_ATTENTION_FUNCTIONS)()
    original = registry["sdpa"]
    calls = []

    def custom_attention(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    registry["sdpa"] = custom_attention
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    monkeypatch.setattr(modeling_llama, "ALL_ATTENTION_FUNCTIONS", registry)
    # The override delegates to its parent; an unrelated import in its globals
    # must not replace the registry actually used by that parent's forward.
    monkeypatch.setitem(
        DelegatingAttention.forward.__globals__,
        "ALL_ATTENTION_FUNCTIONS",
        ALL_ATTENTION_FUNCTIONS,
    )
    model = _llama("sdpa", DelegatingAttention)
    inputs = {"input_ids": torch.tensor([[1, 3, 7, 2]])}
    expected = _logits(model, **inputs)
    calls.clear()
    _hook(model)
    torch.testing.assert_close(_logits(model, **inputs), expected, rtol=0, atol=0)
    assert calls == [layer.self_attn for layer in model.model.layers]


class InlineAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.config._attn_implementation = "eager"
        self.q_proj = torch.nn.Linear(4, 4)

    def forward(self, hidden_states):
        return self.q_proj(hidden_states)


def test_unselected_inline_attention_remains_usable():
    model = _llama()
    model.inline = InlineAttention()
    inputs = torch.ones(1, 4)
    expected = model.inline(inputs)
    _hook(model)
    assert not hasattr(model.inline, IMPL_ATTR)
    torch.testing.assert_close(model.inline(inputs), expected, rtol=0, atol=0)


def test_unresolved_selected_callback_fails_before_model_mutation():
    model = _llama("eager")
    selected = InlineAttention()
    model.inline = selected
    before = model.config._attn_implementation
    with pytest.raises(ValueError, match="Cannot resolve.*attention"):
        initialize_hooked_attention(model, selected)
    assert model.config._attn_implementation == before
    assert not hasattr(selected, IMPL_ATTR)
    assert not hasattr(model.model.layers[0].self_attn, "_ct_original_attention")


@pytest.mark.parametrize("delegating", [False, True])
def test_model_specific_eager_sinks_are_preserved(delegating):
    gpt = pytest.importorskip("transformers.models.gpt_oss.modeling_gpt_oss")
    from transformers import GptOssConfig, GptOssForCausalLM

    class DelegatingGptOssAttention(gpt.GptOssAttention):
        def forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            **kwargs,
        ):
            return super().forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

    torch.manual_seed(1234)
    config = GptOssConfig(
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=["full_attention"],
    )
    config._attn_implementation = "eager"
    model = GptOssForCausalLM(config).eval()
    if delegating:
        old = model.model.layers[0].self_attn
        replacement = DelegatingGptOssAttention(config, 0).eval()
        replacement.load_state_dict(old.state_dict())
        model.model.layers[0].self_attn = replacement
    with torch.no_grad():
        model.model.layers[0].self_attn.sinks.fill_(2.0)
    inputs = {"input_ids": torch.tensor([[1, 3, 7, 2]])}
    expected = _logits(model, **inputs)
    _hook(model)
    torch.testing.assert_close(_logits(model, **inputs), expected, rtol=0, atol=0)


def test_transform_dispatch_requires_only_selected_impl():
    model = _llama()
    inputs = {"input_ids": torch.tensor([[1, 3, 7, 2]])}
    expected = _logits(model, **inputs)
    selected = model.model.layers[0].self_attn
    initialize_hooked_attention(model, selected, allow_unselected=True)
    calls = []
    register_query_hook(selected, lambda module, query: calls.append(module))
    torch.testing.assert_close(_logits(model, **inputs), expected, rtol=0, atol=0)
    assert calls == [selected]
    assert not hasattr(model.model.layers[1].self_attn, IMPL_ATTR)
    delattr(selected, IMPL_ATTR)
    with pytest.raises(ValueError, match="has no impl"):
        _logits(model, **inputs)


def test_strict_dispatch_requirement_survives_later_transform_setup():
    model = _llama()
    selected = model.model.layers[0].self_attn
    initialize_hooked_attention(model, selected, allow_unselected=True)
    initialize_hooked_attention(model, selected)
    initialize_hooked_attention(model, selected, allow_unselected=True)
    with pytest.raises(ValueError, match="has no impl"):
        _logits(model, input_ids=torch.tensor([[1, 3, 7, 2]]))


@pytest.mark.parametrize("offload_device", ["cpu", "disk"])
def test_offloaded_delegating_eager_forward_is_restored(tmp_path, offload_device):
    from compressed_tensors.offload import offload_module

    model = _llama("eager", DelegatingAttention)
    inputs = {"input_ids": torch.tensor([[1, 3, 7, 2]])}
    expected = _logits(model, **inputs)
    wrappers = []
    for layer in model.model.layers:
        module = layer.self_attn
        options = {"offload_dir": str(tmp_path)} if offload_device == "disk" else {}
        offload_module(module, "cpu", offload_device, **options)
        wrappers.append(module.forward.__func__)
    _hook(model)
    assert wrappers == [
        layer.self_attn.forward.__func__ for layer in model.model.layers
    ]
    torch.testing.assert_close(_logits(model, **inputs), expected, rtol=0, atol=0)


def test_new_mask_override_does_not_change_existing_hooked_model(monkeypatch):
    first = _llama()
    original_mask = ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    _hook(first)
    first_backend = first.config._attn_implementation

    def custom_mask(*args, **kwargs):
        return original_mask(*args, **kwargs)

    monkeypatch.setitem(ALL_MASK_ATTENTION_FUNCTIONS, "sdpa", custom_mask)
    second = _llama()
    _hook(second)
    assert ALL_MASK_ATTENTION_FUNCTIONS[first_backend] is original_mask
    assert (
        ALL_MASK_ATTENTION_FUNCTIONS[second.config._attn_implementation] is custom_mask
    )
    assert first_backend != second.config._attn_implementation


@pytest.mark.parametrize("hook_before", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_backend_override_keeps_hooks_and_restores_nested_context(hook_before, fail):
    from compressed_tensors.modeling.attention import override_hooked_attention_backend

    model = _llama()
    inputs = {"input_ids": torch.tensor([[1, 3, 7, 2]])}
    expected_sdpa = _logits(model, **inputs)
    eager_control = _llama("eager")
    expected_eager = _logits(eager_control, **inputs)
    if hook_before:
        _hook(model)
    saved_backend = model.config._attn_implementation
    callbacks = [
        getattr(layer.self_attn, "_ct_original_attention", None)
        for layer in model.model.layers
    ]
    calls = []

    def observe(module, query):
        calls.append(module.layer_idx)

    with pytest.raises(
        RuntimeError, match="context failure"
    ) if fail else nullcontext():
        with override_hooked_attention_backend(model, "eager"):
            if not hook_before:
                _hook(model)
            for layer in model.model.layers:
                register_query_hook(layer.self_attn, observe)
            torch.testing.assert_close(
                _logits(model, **inputs), expected_eager, rtol=0, atol=0
            )
            assert calls == [0, 1]
            eager_backend = model.config._attn_implementation
            with override_hooked_attention_backend(model, "sdpa"):
                torch.testing.assert_close(
                    _logits(model, **inputs), expected_sdpa, rtol=0, atol=0
                )
            assert model.config._attn_implementation == eager_backend
            if fail:
                raise RuntimeError("context failure")
    calls.clear()
    torch.testing.assert_close(_logits(model, **inputs), expected_sdpa, rtol=0, atol=0)
    assert calls == [0, 1]
    if hook_before:
        assert model.config._attn_implementation == saved_backend
        assert callbacks == [
            getattr(layer.self_attn, "_ct_original_attention")
            for layer in model.model.layers
        ]
    else:
        assert model.config._attn_implementation != saved_backend
        assert all(hasattr(layer.self_attn, IMPL_ATTR) for layer in model.model.layers)


def test_backend_override_without_hooks_restores_component_configs():
    from compressed_tensors.modeling.attention import override_hooked_attention_backend

    model, _ = _tiny_llava()
    model.set_attn_implementation({"text_config": "eager", "vision_config": "sdpa"})
    configs = [model.config, model.config.text_config, model.config.vision_config]
    original = [config._attn_implementation for config in configs]
    with override_hooked_attention_backend(model, "eager"):
        assert all(config._attn_implementation == "eager" for config in configs)
    assert [config._attn_implementation for config in configs] == original


def test_unsupported_backend_override_does_not_mutate_hooked_model():
    from compressed_tensors.modeling.attention import override_hooked_attention_backend

    model = _llama()
    _hook(model)
    backend = model.config._attn_implementation
    callbacks = [layer.self_attn._ct_original_attention for layer in model.model.layers]
    with pytest.raises(ValueError, match="Cannot resolve"):
        with override_hooked_attention_backend(model, "not_a_registered_backend"):
            pytest.fail("unsupported backend was accepted")
    assert model.config._attn_implementation == backend
    assert callbacks == [
        layer.self_attn._ct_original_attention for layer in model.model.layers
    ]


@pytest.mark.parametrize("text_backend", ["eager", "sdpa"])
def test_component_mask_preserves_padded_prefill_and_cached_decode(text_backend):
    torch.manual_seed(1234)
    model, _ = _tiny_llava()
    model.set_attn_implementation(
        {"text_config": text_backend, "vision_config": "sdpa"}
    )
    inputs = {
        "input_ids": torch.tensor([[0, 0, 1, 3, 7, 2], [0, 4, 5, 6, 7, 8]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]]),
    }
    decode_inputs = {
        "input_ids": torch.tensor([[9], [10]]),
        "attention_mask": torch.cat(
            [inputs["attention_mask"], torch.ones(2, 1, dtype=torch.long)], dim=1
        ),
    }
    with torch.no_grad():
        expected_prefill = model(**inputs, use_cache=True)
        expected_decode = model(
            **decode_inputs,
            past_key_values=expected_prefill.past_key_values,
            use_cache=True,
        )
    args = QuantizationArgs(num_bits=8, strategy="tensor", dynamic=True)
    scheme = QuantizationScheme(
        targets=["re:.*language_model.*self_attn$"], input_activations=args
    )
    apply_quantization_config(
        model, QuantizationConfig(config_groups={"attention": scheme})
    )
    model.apply(disable_quantization)
    with torch.no_grad():
        actual_prefill = model(**inputs, use_cache=True)
        actual_decode = model(
            **decode_inputs,
            past_key_values=actual_prefill.past_key_values,
            use_cache=True,
        )
    torch.testing.assert_close(
        actual_prefill.logits, expected_prefill.logits, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_decode.logits, expected_decode.logits, rtol=0, atol=0
    )
