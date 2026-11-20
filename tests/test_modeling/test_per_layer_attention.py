# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
import transformers
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    apply_quantization_config,
)


def tiny_model(heterogeneous=True, shared=0, global_heads=None, backend="sdpa"):
    torch.manual_seed(1234)
    kwargs = dict(
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
    )
    if heterogeneous:
        if not hasattr(transformers, "Gemma4TextConfig"):
            pytest.skip("Gemma4 needs a recent Transformers")
        config = transformers.Gemma4TextConfig(
            **kwargs,
            head_dim=8,
            global_head_dim=16,
            layer_types=["sliding_attention", "full_attention"] * 2,
            sliding_window=4,
            vocab_size_per_layer_input=32,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=shared,
            num_global_key_value_heads=global_heads,
            attention_k_eq_v=global_heads is not None,
        )
        if not getattr(config, "is_heterogeneous", False):
            pytest.skip("Per-layer configurations unavailable")
        model = transformers.Gemma4ForCausalLM(config)
    else:
        model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**kwargs))
    model.set_attn_implementation(backend)
    return model.eval()


def attention_modules(model):
    return [layer.self_attn for layer in model.model.layers]


def quantization(model, strategy="tensor", dynamic=False, kv=False, **kwargs):
    args = QuantizationArgs(num_bits=8, strategy=strategy, dynamic=dynamic, **kwargs)
    return QuantizationConfig(
        config_groups={}
        if kv
        else {
            "attention": QuantizationScheme(
                targets=[type(attention_modules(model)[0]).__name__],
                input_activations=args,
            )
        },
        kv_cache_scheme=args if kv else None,
        ignore=[
            f"model.layers.{i}.self_attn"
            for i, m in enumerate(attention_modules(model))
            if not hasattr(m, "k_proj")
        ],
    )


def logits(model, inputs=None):
    with torch.no_grad():
        return model(
            torch.tensor([[1, 3, 7, 2]]) if inputs is None else inputs, use_cache=False
        ).logits


def snapshot(model):
    return (
        {n: t.clone() for n, t in model.state_dict().items()},
        [
            (n, tuple(m._modules), tuple(m._forward_pre_hooks), tuple(m._forward_hooks))
            for n, m in model.named_modules()
        ],
        model.config._attn_implementation,
    )


def unchanged(model, before):
    state, modules, backend = before
    after = snapshot(model)
    assert after[1:] == (modules, backend)
    assert state.keys() == after[0].keys()
    for name in state:
        torch.testing.assert_close(state[name], after[0][name], rtol=0, atol=0)
    assert not any(hasattr(m, "quantization_scheme") for m in model.modules())


@pytest.mark.parametrize(
    "heterogeneous,shared,global_heads",
    [(False, 0, None), (True, 0, None), (True, 2, None), (True, 0, 1)],
)
@pytest.mark.parametrize("strategy", ["tensor", "attn_head"])
@pytest.mark.parametrize("kv", [False, True])
def test_owning_layer_shapes_and_original_output(
    heterogeneous, shared, global_heads, strategy, kv
):
    model = tiny_model(heterogeneous, shared, global_heads)
    reference = logits(model)
    apply_quantization_config(model, quantization(model, strategy, kv=kv))
    for module in attention_modules(model):
        if hasattr(module, "k_proj"):
            heads = module.k_proj.out_features // module.head_dim
            expected = (1,) if strategy == "tensor" else (heads, 1, 1)
            assert module.k_scale.shape == module.v_scale.shape == expected
            assert hasattr(module, "q_scale") == (not kv)
            if not kv:
                assert module.q_scale.shape == (
                    (1,)
                    if strategy == "tensor"
                    else (module.q_proj.out_features // module.head_dim, 1, 1)
                )
            for name in ("q_scale", "k_scale", "v_scale"):
                if hasattr(module, name):
                    getattr(module, name).data.fill_(0.1)
        else:
            assert not hasattr(module, "k_scale")
        module.quantization_enabled = False
    torch.testing.assert_close(logits(model), reference, atol=0, rtol=0)
    for module in attention_modules(model):
        module.quantization_enabled = True
    assert torch.isfinite(logits(model)).all()


@pytest.mark.parametrize("strategy", ["token", "group", "tensor_group"])
@pytest.mark.parametrize("kv", [False, True])
def test_dynamic_attention(strategy, kv):
    model = tiny_model()
    kwargs = {} if strategy == "token" else {"group_size": 8}
    apply_quantization_config(model, quantization(model, strategy, True, kv, **kwargs))
    assert all(hasattr(m, "quantization_scheme") for m in attention_modules(model))
    assert torch.isfinite(logits(model)).all()


@pytest.mark.parametrize("index", [None, True, -1, 4, 10**20, 1.0, "1", np.int64(3)])
def test_layer_index_validated_before_mutation(index):
    model = tiny_model()
    attention_modules(model)[-1].layer_idx = index
    before = snapshot(model)
    if isinstance(index, np.integer):
        apply_quantization_config(model, quantization(model, "attn_head"))
        assert attention_modules(model)[-1].k_scale.shape == (2, 1, 1)
    else:
        with pytest.raises(ValueError, match="layer_idx"):
            apply_quantization_config(model, quantization(model, "attn_head"))
        unchanged(model, before)


@pytest.mark.parametrize("kv", [False, True])
@pytest.mark.parametrize(
    "strategy,dynamic,size",
    [
        ("group", False, 8),
        ("tensor_group", False, 8),
        ("group", True, 16),
        ("tensor_group", True, 16),
    ],
)
def test_unsupported_group_geometry_is_atomic(kv, strategy, dynamic, size):
    model = tiny_model()
    before = snapshot(model)
    with pytest.raises(ValueError, match="group"):
        apply_quantization_config(
            model, quantization(model, strategy, dynamic, kv, group_size=size)
        )
    unchanged(model, before)


@pytest.mark.parametrize("kv", [False, True])
@pytest.mark.parametrize("strategy", ["tensor", "attn_head"])
def test_declared_kv_heads_match_projection(kv, strategy):
    model = tiny_model()
    config = model.config
    config.per_layer_config = {
        i: {
            **{key: getattr(layer, key) for key in config.per_layer_attributes},
            "num_key_value_heads": 1 if i == 3 else 2,
        }
        for i, layer in enumerate(config.per_layer_config)
    }
    before = snapshot(model)
    if strategy == "attn_head":
        with pytest.raises(ValueError, match="KV heads.*k_proj"):
            apply_quantization_config(model, quantization(model, strategy, kv=kv))
        unchanged(model, before)
    else:
        apply_quantization_config(model, quantization(model, strategy, kv=kv))
        assert attention_modules(model)[-1].k_scale.shape == (1,)


def test_explicit_unsupported_reader_rejected_before_mutation():
    model = tiny_model(shared=2)
    config = quantization(model)
    config.ignore = []
    before = snapshot(model)
    with pytest.raises(ValueError, match="requested for unsupported attention"):
        apply_quantization_config(model, config)
    unchanged(model, before)


def test_heterogeneous_kv_checkpoint(tmp_path):
    model = tiny_model()
    apply_quantization_config(model, quantization(model, dynamic=True, kv=True))
    reference = logits(model)
    compressor = ModelCompressor.from_pretrained_model(
        model, quantization_format="dense"
    )
    model.save_pretrained(tmp_path)
    compressor.update_config(str(tmp_path))
    restored, info = type(model).from_pretrained(
        tmp_path, attn_implementation="sdpa", output_loading_info=True
    )
    assert not any(
        info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    )
    assert all(hasattr(m, "kv_cache") for m in attention_modules(restored))
    torch.testing.assert_close(logits(restored), reference, atol=0, rtol=0)
