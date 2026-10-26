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
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)


def _model(shared=2):
    if not hasattr(transformers, "Gemma4TextConfig"):
        pytest.skip("Gemma4 needs a recent Transformers")
    torch.manual_seed(1234)
    config = transformers.Gemma4TextConfig(
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        global_head_dim=32,
        num_hidden_layers=4,
        layer_types=["sliding_attention", "full_attention"] * 2,
        intermediate_size=64,
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=shared,
    )
    if not getattr(config, "is_heterogeneous", False):
        pytest.skip("Gemma4 per-layer configurations unavailable")
    model = transformers.Gemma4ForCausalLM(config).eval()
    model.set_attn_implementation("sdpa")
    return model


def _snapshot(model):
    return (
        {name: value.clone() for name, value in model.state_dict().items()},
        [
            (name, tuple(m._modules), len(m._forward_pre_hooks), len(m._forward_hooks))
            for name, m in model.named_modules()
        ],
        model.config._attn_implementation,
    )


def _assert_unchanged(model, snapshot):
    weights, modules, backend = snapshot
    assert list(model.state_dict()) == list(weights)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
    assert [
        (name, tuple(m._modules), len(m._forward_pre_hooks), len(m._forward_hooks))
        for name, m in model.named_modules()
    ] == modules
    assert model.config._attn_implementation == backend
    assert all(not hasattr(m, "quantization_scheme") for m in model.modules())


def _args(strategy="tensor", dynamic=False, group_size=None):
    return QuantizationArgs(
        num_bits=8, strategy=strategy, dynamic=dynamic, group_size=group_size
    )


@pytest.mark.parametrize("with_kv", [False, True])
def test_explicit_unsupported_attention_rejected_before_mutation(with_kv):
    model = _model()
    before = _snapshot(model)
    scheme = QuantizationScheme(
        targets=["Gemma4TextAttention"], input_activations=_args()
    )
    config = QuantizationConfig(
        config_groups={"attention": scheme},
        kv_cache_scheme=_args() if with_kv else None,
    )
    with pytest.raises(ValueError, match="requested.*attention|attention.*requested"):
        apply_quantization_config(model, config)
    _assert_unchanged(model, before)


def test_explicit_custom_attention_does_not_silently_skip_name_heuristic():
    torch.manual_seed(1234)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=32,
        )
    ).eval()
    model.set_attn_implementation("sdpa")
    original = model.model.layers[0].self_attn

    class LlamaSelfMHA(type(original)):
        pass

    replacement = LlamaSelfMHA(model.config, layer_idx=0)
    replacement.load_state_dict(original.state_dict())
    model.model.layers[0].self_attn = replacement
    before = _snapshot(model)
    scheme = QuantizationScheme(targets=["LlamaSelfMHA"], input_activations=_args())
    with pytest.raises(ValueError, match="requested.*attention|attention.*requested"):
        apply_quantization_config(
            model, QuantizationConfig(config_groups={"attn": scheme})
        )
    _assert_unchanged(model, before)


@pytest.mark.parametrize("selection", ["ignore", "allowed_modules"])
def test_explicit_supported_attention_scope_preserves_original(selection):
    model = _model()
    inputs = torch.tensor([[1, 3, 7, 2]])
    with torch.no_grad():
        reference = model(inputs, use_cache=False).logits
    ignored = [f"model.layers.{i}.self_attn" for i in (2, 3)]
    scheme = QuantizationScheme(
        targets=["Gemma4TextAttention"], input_activations=_args()
    )
    config = QuantizationConfig(
        config_groups={"attention": scheme},
        ignore=ignored if selection == "ignore" else [],
    )
    kwargs = {"allowed_modules": [layer.self_attn for layer in model.model.layers[:2]]}
    apply_quantization_config(
        model, config, **(kwargs if selection == "allowed_modules" else {})
    )
    for index, layer in enumerate(model.model.layers):
        assert hasattr(layer.self_attn, "q_scale") == (index < 2)
        layer.self_attn.quantization_enabled = False
    with torch.no_grad():
        actual = model(inputs, use_cache=False).logits
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("strategy", ["group", "tensor_group"])
def test_group_size_rejected_before_attention_mutation(strategy):
    model = _model(shared=0)
    before = _snapshot(model)
    with pytest.raises(ValueError, match="group_size.*32.*head_dim.*16"):
        apply_quantization_config(
            model,
            QuantizationConfig(
                config_groups={}, kv_cache_scheme=_args(strategy, True, 32)
            ),
        )
    _assert_unchanged(model, before)


def _rotation(q, k, block=16, name="rotation", transform_type="hadamard"):
    def targets(indices):
        return [f"model.layers.{i}.self_attn" for i in indices]

    return name, TransformScheme(
        type=transform_type,
        head_dim=block,
        apply=[
            TransformArgs(targets=targets(q), location="q_attn"),
            TransformArgs(targets=targets(k), location="k_cache"),
        ],
    )


@pytest.mark.parametrize("q,k", [([0], [0]), ([2], []), ([0, 2, 3], [0]), ([0, 2], [])])
def test_shared_cache_mismatched_basis_rejected_before_mutation(q, k):
    model = _model()
    before = _snapshot(model)
    with pytest.raises(ValueError, match="query.*key|key.*query"):
        apply_transform_config(
            model, TransformConfig(config_groups=dict([_rotation(q, k)]))
        )
    _assert_unchanged(model, before)


@pytest.mark.parametrize("block", [4, 8, 16])
@pytest.mark.parametrize("transform_type", ["hadamard", "random-hadamard"])
def test_valid_partial_shared_rotation_preserves_original(block, transform_type):
    model = _model()
    inputs = torch.tensor([[1, 3, 7, 2]])
    with torch.no_grad():
        reference = model(inputs, use_cache=False).logits
    apply_transform_config(
        model,
        TransformConfig(
            config_groups=dict(
                [_rotation([0, 2], [0], block=block, transform_type=transform_type)]
            )
        ),
    )
    with torch.no_grad():
        actual = model(inputs, use_cache=False).logits
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)


def test_same_basis_from_distinct_factories_preserves_shared_attention():
    model = _model()
    inputs = torch.tensor([[1, 3, 7, 2]])
    with torch.no_grad():
        reference = model(inputs, use_cache=False).logits
    config = TransformConfig(
        config_groups=dict(
            [
                _rotation([0, 2], [], name="queries"),
                _rotation([], [0], name="keys"),
            ]
        )
    )
    apply_transform_config(model, config)
    with torch.no_grad():
        actual = model(inputs, use_cache=False).logits
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)


def test_wrong_present_basis_rejected_before_mutation():
    model = _model()
    before = _snapshot(model)
    config = TransformConfig(
        config_groups=dict(
            [
                _rotation([0, 2], [], name="queries", transform_type="random-hadamard"),
                _rotation([], [0], name="keys"),
            ]
        )
    )
    with pytest.raises(ValueError, match="query.*key|key.*query"):
        apply_transform_config(model, config)
    _assert_unchanged(model, before)


def test_unselected_decoder_does_not_change_transform_validation_scope():
    model = _model()
    model.unrelated = _model()
    # This component does not expose a resolvable sharing relationship, and is
    # neither selected by the transform nor used by this decoder's forward.
    for layer in model.unrelated.model.layers:
        if layer.self_attn.is_kv_shared_layer:
            layer.self_attn.layer_type = None
    inputs = torch.tensor([[1, 3, 7, 2]])
    with torch.no_grad():
        reference = model(inputs, use_cache=False).logits
    apply_transform_config(
        model, TransformConfig(config_groups=dict([_rotation([0, 1, 2, 3], [0, 1])]))
    )
    with torch.no_grad():
        actual = model(inputs, use_cache=False).logits
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
