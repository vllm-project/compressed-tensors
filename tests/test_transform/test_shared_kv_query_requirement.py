# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.modeling import (
    initialize_hooked_attention,
    register_key_hook,
    register_query_hook,
)
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformFactory,
    TransformScheme,
    apply_transform_config,
)
from tests.test_modeling.test_per_layer_attention import (
    attention_modules,
    logits,
    tiny_model,
)


def _args(indices, location):
    return TransformArgs(
        targets=[f"model.layers.{index}.self_attn" for index in indices],
        location=location,
    )


def _apply(model, groups, direct):
    schemes = {
        name: TransformScheme(type="hadamard", head_dim=8, apply=args)
        for name, args in groups.items()
    }
    if direct:
        for name, scheme in schemes.items():
            TransformFactory.from_scheme(scheme, name=name).apply_to_model(model)
    else:
        apply_transform_config(model, TransformConfig(config_groups=schemes))


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("key_first", [False, True])
def test_transformed_shared_keys_require_reader_impl(direct, key_first):
    model = tiny_model(shared=2)
    reference = logits(model)
    args = [_args([0, 1], "q_attn"), _args([0, 1], "k_cache")]
    _apply(model, {"rotation": args[::-1] if key_first else args}, direct)
    assert all(hasattr(module, "impl") for module in attention_modules(model)[:2])
    assert all(not hasattr(module, "impl") for module in attention_modules(model)[2:])
    assert torch.isfinite(reference).all()
    # Main already rejected this at forward. Native fallback must not turn it into
    # silently incorrect attention over rotated keys with an unrotated query.
    with pytest.raises((AssertionError, ValueError), match="impl"):
        logits(model)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("order", ["together", "key_first", "query_first"])
def test_shared_keys_with_all_query_hooks_preserve_output(direct, order):
    model = tiny_model(shared=2)
    reference = logits(model)
    query = _args(range(4), "q_attn")
    key = _args([0, 1], "k_cache")
    groups = (
        {"rotation": [query, key]}
        if order == "together"
        else {"key": [key], "query": [query]}
        if order == "key_first"
        else {"query": [query], "key": [key]}
    )
    _apply(model, groups, direct)
    calls = {"q": 0, "k": 0}

    def query_hook(*_):
        calls["q"] += 1

    def key_hook(*_):
        calls["k"] += 1

    for index, module in enumerate(attention_modules(model)):
        register_query_hook(module, query_hook)
        if index < 2:
            register_key_hook(module, key_hook)
    torch.testing.assert_close(logits(model), reference, atol=1e-5, rtol=0)
    assert calls == {"q": 4, "k": 2}


@pytest.mark.parametrize("direct", [False, True])
def test_untransformed_shared_key_type_keeps_native_fallback(direct):
    model = tiny_model(shared=2)
    reference = logits(model)
    # Full-attention layer1 is not transformed. Its reader3 must retain native
    # attention while sliding-attention producer0 and reader2 rotate together.
    initialize_hooked_attention(model, attention_modules(model)[1])
    _apply(
        model,
        {"rotation": [_args([0, 2], "q_attn"), _args([0], "k_cache")]},
        direct,
    )
    assert not hasattr(attention_modules(model)[3], "impl")
    torch.testing.assert_close(logits(model), reference, atol=1e-5, rtol=0)
