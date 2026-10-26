# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

import numpy as np
import pytest
import torch
import transformers
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.modeling.attention import (
    initialize_hooked_attention,
    register_query_hook,
)
from compressed_tensors.modeling.kvcache import register_key_hook, register_value_hook
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    apply_quantization_config,
)
from compressed_tensors.quantization.lifecycle.initialize import _get_layer_config
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)
from pydantic import ValidationError
from transformers import DynamicCache, LlamaConfig


pytestmark = pytest.mark.skipif(
    not hasattr(transformers.PretrainedConfig, "per_layer_config"),
    reason="Per-layer configurations require a recent Transformers",
)


def _model(shared=0, backend="sdpa", dtype=torch.float32):
    if not hasattr(transformers, "Gemma4TextConfig"):
        pytest.skip("Gemma4 requires a recent Transformers")
    torch.manual_seed(1234)
    config = transformers.Gemma4TextConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=16,
        num_hidden_layers=4,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=4,
        intermediate_size=32,
        vocab_size=32,
        vocab_size_per_layer_input=32,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=shared,
    )
    if not getattr(config, "is_heterogeneous", False):
        pytest.skip("Gemma4 per-layer configurations unavailable")
    model = transformers.Gemma4ForCausalLM(config).to(dtype=dtype).eval()
    model.set_attn_implementation(backend)
    return model


def _attention(model):
    return [layer.self_attn for layer in model.model.layers]


def _args(strategy="tensor", dynamic=False, **kwargs):
    return QuantizationArgs(num_bits=8, strategy=strategy, dynamic=dynamic, **kwargs)


def _quantization(args=None, **kwargs):
    return QuantizationConfig(
        config_groups={
            "attention": QuantizationScheme(
                targets=["Gemma4TextAttention"],
                input_activations=_args() if args is None else args,
            )
        },
        **kwargs,
    )


def _snapshot(model):
    return (
        {name: value.clone() for name, value in model.state_dict().items()},
        [
            (
                name,
                tuple(module._modules),
                tuple(module._forward_pre_hooks),
                tuple(module._forward_hooks),
            )
            for name, module in model.named_modules()
        ],
        model.config._attn_implementation,
    )


def _unchanged(model, before):
    state, modules, backend = before
    assert list(model.state_dict()) == list(state)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)
    assert [
        (
            name,
            tuple(module._modules),
            tuple(module._forward_pre_hooks),
            tuple(module._forward_hooks),
        )
        for name, module in model.named_modules()
    ] == modules
    assert model.config._attn_implementation == backend
    assert all(not hasattr(module, "quantization_scheme") for module in model.modules())


def _rotation(model, block=8, transform_type="hadamard"):
    queries = [f"model.layers.{i}.self_attn" for i in range(4)]
    keys = [
        f"model.layers.{i}.self_attn"
        for i, module in enumerate(_attention(model))
        if not module.is_kv_shared_layer
    ]
    return TransformConfig(
        config_groups={
            "rotation": TransformScheme(
                type=transform_type,
                head_dim=block,
                apply=[
                    TransformArgs(targets=queries, location="q_attn"),
                    TransformArgs(targets=keys, location="k_cache"),
                ],
            )
        }
    )


@pytest.mark.parametrize("index", [0, 3, np.int64(1)])
def test_layer_config_accepts_boundary_and_integral_indices(index):
    model = _model()
    attention = _attention(model)[int(index)]
    attention.layer_idx = index
    layer = _get_layer_config(model.config, attention)
    assert layer.head_dim == (8 if int(index) % 2 == 0 else 16)
    assert layer.num_key_value_heads == 2


@pytest.mark.parametrize("index", [-1, True, 1.0, "1", None, 10**100])
def test_layer_config_rejects_invalid_or_huge_indices_before_mutation(index):
    model = _model()
    _attention(model)[0].layer_idx = index
    before = _snapshot(model)
    with pytest.raises(ValueError, match="layer_idx.*does not select"):
        apply_quantization_config(model, _quantization())
    _unchanged(model, before)


def test_empty_heterogeneous_layer_view_has_no_valid_index():
    config = LlamaConfig(num_hidden_layers=0, per_layer_config={})
    assert config.is_heterogeneous
    assert len(config.per_layer_config) == 0
    module = torch.nn.Module()
    module.layer_idx = 0
    with pytest.raises(ValueError, match="0 layers"):
        _get_layer_config(config, module)


@pytest.mark.parametrize("overrides", [{-1: {}}, {10**100: {}}, {"not-an-index": {}}])
def test_transformers_rejects_malformed_layer_indices(overrides):
    # The public HF setter rejects these before CT can receive a layer view.
    with pytest.raises(ValueError):
        LlamaConfig(num_hidden_layers=2, per_layer_config=overrides)


@pytest.mark.parametrize("overrides", [[], [None], "invalid"])
def test_transformers_layer_overrides_require_a_mapping(overrides):
    with pytest.raises(AttributeError, match="items"):
        LlamaConfig(num_hidden_layers=2, per_layer_config=overrides)


def test_empty_layer_overrides_preserve_homogeneous_dimensions():
    config = LlamaConfig(num_hidden_layers=2, head_dim=8, per_layer_config={})
    module = torch.nn.Module()
    for index in (0, 1):
        module.layer_idx = index
        assert _get_layer_config(config, module).head_dim == 8
    config.per_layer_config = None
    module.layer_idx = 10**100
    assert _get_layer_config(config, module) is config


@pytest.mark.parametrize(
    "operation", ["weights", "output_activations", "missing_input"]
)
def test_invalid_attention_operations_rejected_before_mutation(operation):
    model = _model()
    before = _snapshot(model)
    arguments = {"input_activations": _args()}
    if operation == "missing_input":
        arguments = {}
    else:
        arguments[operation] = _args()
    scheme = QuantizationScheme(targets=["Gemma4TextAttention"], **arguments)
    with pytest.raises(ValueError, match="Cannot apply"):
        apply_quantization_config(
            model, QuantizationConfig(config_groups={"bad": scheme})
        )
    _unchanged(model, before)


@pytest.mark.parametrize(
    "field,value",
    [("location", "unknown"), ("inverse", "invalid"), ("operation", "inverse")],
)
def test_transform_operation_fields_are_validated(field, value):
    arguments = {"targets": ["Gemma4TextAttention"], "location": "q_attn"}
    arguments[field] = value
    with pytest.raises(ValidationError):
        TransformArgs(**arguments)


@pytest.mark.parametrize("block", [None, 0, -8, 3, 32])
def test_attention_rotation_dimension_rejected_before_mutation(block):
    model = _model()
    before = _snapshot(model)
    with pytest.raises(ValueError, match="head_dim"):
        apply_transform_config(model, _rotation(model, block))
    _unchanged(model, before)


@pytest.mark.parametrize("strategy", ["group", "tensor_group"])
@pytest.mark.parametrize("group_size", [4, 16])
def test_dynamic_attention_group_dimensions(strategy, group_size):
    model = _model()
    before = _snapshot(model)
    config = _quantization(_args(strategy, True, group_size=group_size))
    if group_size == 16:
        with pytest.raises(ValueError, match="group_size.*16.*head_dim.*8"):
            apply_quantization_config(model, config)
        _unchanged(model, before)
    else:
        apply_quantization_config(model, config)
        assert all(hasattr(module, "impl") for module in _attention(model))
        assert all(not hasattr(module, "q_scale") for module in _attention(model))
        with torch.no_grad():
            assert torch.isfinite(
                model(torch.tensor([[1, 2, 3]]), use_cache=False).logits
            ).all()


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_and_regex_targets_override_class_target(reverse):
    model = _model()
    entries = [
        (
            "class",
            QuantizationScheme(
                targets=["Gemma4TextAttention"], input_activations=_args()
            ),
        ),
        (
            "regex",
            QuantizationScheme(
                targets=[r"re:model\.layers\.[01]\.self_attn$"],
                input_activations=_args("attn_head"),
            ),
        ),
        (
            "exact",
            QuantizationScheme(
                targets=["model.layers.0.self_attn"],
                input_activations=_args(dynamic=True),
            ),
        ),
    ]
    apply_quantization_config(
        model,
        QuantizationConfig(
            config_groups=dict(reversed(entries) if reverse else entries)
        ),
    )
    modules = _attention(model)
    assert modules[0].quantization_scheme.input_activations.dynamic is True
    assert not hasattr(modules[0], "q_scale")
    assert modules[1].q_scale.shape == (4, 1, 1)
    assert modules[2].q_scale.shape == modules[3].q_scale.shape == (1,)


@pytest.mark.parametrize("kv", [False, True])
def test_empty_allowed_modules_is_noop_even_for_unsupported_selected_attention(kv):
    model = _model(shared=2)
    before = _snapshot(model)
    config = _quantization(kv_cache_scheme=_args() if kv else None)
    apply_quantization_config(model, config, allowed_modules=[])
    _unchanged(model, before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_attention_qparam_device_dtype_and_shape(dtype, device):
    with torch.device(device):
        model = _model(dtype=dtype)
    args = _args("attn_head")
    apply_quantization_config(model, _quantization(args))
    for module in _attention(model):
        for name, heads in [("q", 4), ("k", 2), ("v", 2)]:
            scale = getattr(module, f"{name}_scale")
            zero = getattr(module, f"{name}_zero_point")
            assert scale.shape == zero.shape == (heads, 1, 1)
            assert scale.dtype == dtype
            assert zero.dtype == args.zp_dtype
            assert scale.device.type == zero.device.type == device
            assert not scale.requires_grad and not zero.requires_grad


@pytest.mark.parametrize("scope", ["attention", "kv"])
@pytest.mark.parametrize("strategy", ["attn_head", "tensor"])
@pytest.mark.parametrize("mismatch", [False, True])
def test_heterogeneous_kv_head_validation_precedes_all_mutation(
    scope, strategy, mismatch
):
    model = _model()
    reference = deepcopy(model)
    if mismatch:
        config = model.config
        overrides = {
            index: {
                name: deepcopy(getattr(layer, name))
                for name in config.per_layer_attributes
            }
            for index, layer in enumerate(config.per_layer_config)
        }
        overrides[3]["num_key_value_heads"] = 1
        config.per_layer_config = overrides
        assert config.per_layer_config[3].num_key_value_heads == 1
        module = _attention(model)[3]
        assert module.k_proj.out_features // module.head_dim == 2
    before = _snapshot(model)
    args = _args(strategy)
    quantization = (
        _quantization(args)
        if scope == "attention"
        else QuantizationConfig(config_groups={}, kv_cache_scheme=args)
    )
    if mismatch and strategy == "attn_head":
        with pytest.raises(ValueError, match="1 key/value heads.*k_proj.*2 heads"):
            apply_quantization_config(model, quantization)
        _unchanged(model, before)
    else:
        apply_quantization_config(model, quantization)
        for module in _attention(model):
            expected = (2, 1, 1) if strategy == "attn_head" else (1,)
            assert module.k_scale.shape == module.v_scale.shape == expected
            module.quantization_enabled = False
        inputs = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            expected = reference(inputs, use_cache=False).logits
            actual = model(inputs, use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("transform_type", ["hadamard", "random-hadamard"])
@pytest.mark.parametrize("valid", [False, True])
def test_shared_attention_validation_preserves_meta_device_context(
    transform_type, valid
):
    original_device = torch.get_default_device()
    with torch.device("meta"):
        model = _model(shared=2)
        config = _rotation(model, 8 if valid else 3, transform_type)
        if valid:
            apply_transform_config(model, config)
            for module in _attention(model):
                assert module.rotation_q_attn.weight.device.type == "meta"
                if not module.is_kv_shared_layer:
                    assert module.rotation_k_cache.weight.device.type == "meta"
        else:
            with pytest.raises(ValueError, match="head_dim"):
                apply_transform_config(model, config)
            assert all(not hasattr(module, "impl") for module in _attention(model))
        assert torch.empty(0).device.type == "meta"
    assert torch.get_default_device() == original_device


def _assert_cache_equal(reference, actual, rotated):
    assert type(reference) is type(actual)
    assert len(reference.layers) == len(actual.layers)
    for first, second in zip(reference.layers, actual.layers):
        assert type(first) is type(second)
        for name in ("keys", "values"):
            expected, value = getattr(first, name), getattr(second, name)
            if expected is None:
                assert value is None
            else:
                assert expected.shape == value.shape
                if name == "values" or not rotated:
                    torch.testing.assert_close(
                        value,
                        expected,
                        atol=2e-6 if rotated else 0,
                        rtol=2e-5 if rotated else 0,
                    )


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("shared", [0, 2])
@pytest.mark.parametrize("rotated", [False, True])
def test_sliding_cache_prefill_decode_reorder_expand_and_reset(
    backend, shared, rotated
):
    model = _model(shared=shared, backend=backend)
    reference = deepcopy(model)
    if rotated:
        apply_transform_config(model, _rotation(model))
    else:
        for module in _attention(model):
            initialize_hooked_attention(model, module, allow_unselected=True)
    original_cache = DynamicCache(config=reference.config)
    actual_cache = DynamicCache(config=model.config)
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])

    def compare(tokens):
        with torch.no_grad():
            expected = reference(
                tokens, past_key_values=original_cache, use_cache=True
            ).logits
            actual = model(tokens, past_key_values=actual_cache, use_cache=True).logits
        torch.testing.assert_close(
            actual, expected, atol=2e-6 if rotated else 0, rtol=2e-5 if rotated else 0
        )
        _assert_cache_equal(original_cache, actual_cache, rotated)

    compare(inputs)
    assert original_cache.layers[0].keys.shape[-2] < inputs.shape[-1]
    compare(torch.tensor([[7], [8]]))
    for cache in (original_cache, actual_cache):
        cache.reorder_cache(torch.tensor([1, 0]))
    compare(torch.tensor([[9], [10]]))
    for cache in (original_cache, actual_cache):
        cache.batch_repeat_interleave(2)
    compare(torch.tensor([[11], [12], [13], [14]]))
    for cache in (original_cache, actual_cache):
        cache.reset()
        assert cache.get_seq_length() == 0
    compare(inputs)


@pytest.mark.parametrize(
    "register", [register_query_hook, register_key_hook, register_value_hook]
)
@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_user_hook_exception_and_removal_preserve_next_forward(register, exception):
    model = _model()
    reference = deepcopy(model)
    for module in _attention(model):
        initialize_hooked_attention(model, module)
    called = []

    def fail(module, states):
        called.append(module)
        raise exception("user hook failed")

    handle = register(_attention(model)[0], fail)
    inputs = torch.tensor([[1, 2, 3]])
    failed_cache = DynamicCache(config=model.config)
    with pytest.raises(exception, match="user hook failed"), torch.no_grad():
        model(inputs, past_key_values=failed_cache, use_cache=True)
    assert called == [_attention(model)[0]]
    handle.remove()
    handle.remove()
    # A query failure can follow a completed cache update; use the public reset
    # rather than claiming a failed model forward rolls its cache back atomically.
    failed_cache.reset()
    with torch.no_grad():
        expected = reference(inputs, use_cache=False).logits
        actual = model(inputs, past_key_values=failed_cache, use_cache=True).logits
        fresh = model(inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(fresh, expected, atol=0, rtol=0)
    assert called == [_attention(model)[0]]


@pytest.mark.parametrize("shared", [0, 2])
@pytest.mark.parametrize("transform_type", ["hadamard", "random-hadamard"])
def test_dynamic_attention_rotation_save_before_first_forward(
    tmp_path, shared, transform_type
):
    model = _model(shared=shared)
    reference = deepcopy(model)
    apply_quantization_config(
        model, QuantizationConfig(config_groups={}, kv_cache_scheme=_args(dynamic=True))
    )
    apply_transform_config(model, _rotation(model, transform_type=transform_type))
    calls = []
    handle = model.register_forward_pre_hook(lambda *args: calls.append(True))
    compressor = ModelCompressor.from_pretrained_model(
        model, quantization_format="dense"
    )
    model.save_pretrained(tmp_path)
    compressor.update_config(str(tmp_path))
    assert calls == []
    handle.remove()
    device = torch.get_default_device()
    restored, info = transformers.Gemma4ForCausalLM.from_pretrained(
        tmp_path, attn_implementation="sdpa", output_loading_info=True
    )
    assert torch.get_default_device() == device
    assert not any(
        info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    )
    inputs = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        expected = model(inputs, use_cache=False).logits
        actual = restored(inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for module in _attention(restored):
        module.quantization_enabled = False
    with torch.no_grad():
        original = reference(inputs, use_cache=False).logits
        unquantized = restored(inputs, use_cache=False).logits
    torch.testing.assert_close(unquantized, original, atol=2e-6, rtol=2e-5)
