# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

import pytest
import torch
from compressed_tensors.modeling import register_key_hook, register_query_hook
from compressed_tensors.transform import TransformConfig, apply_transform_config
from compressed_tensors.transform.factory.base import TransformBase, TransformFactory
from tests.test_modeling.test_attention_application import (
    _assert_unchanged,
    _model,
    _rotation,
    _snapshot,
)


def _scheme(
    name, kind, precision, *, inverse=False, q=(0, 1, 2, 3), k=(0, 1), randomize=False
):
    name, scheme = _rotation(q, k, block=8, name=name, transform_type=kind)
    scheme.precision = precision
    scheme.randomize = randomize
    for arg in scheme.apply:
        arg.inverse = inverse
    return name, scheme


def _apply_with_reference(model, reference, config):
    # The factory path is the original execution contract, without the preflight.
    for name, scheme in config.config_groups.items():
        TransformFactory.from_scheme(scheme, name=name).apply_to_model(reference)
    apply_transform_config(model, config)


def _assert_outputs(model, reference, original, precision, *, random_matrix=False):
    calls = {"query": 0, "key": 0}

    def query_hook(_, states):
        calls["query"] += 1

    def key_hook(_, states):
        calls["key"] += 1

    handles = []
    for layer in model.model.layers:
        handles.append(register_query_hook(layer.self_attn, query_hook))
        if not layer.self_attn.is_kv_shared_layer:
            handles.append(register_key_hook(layer.self_attn, key_hook))
    try:
        with torch.no_grad():
            inputs = torch.tensor([[1, 3, 7, 2]])
            expected = reference(inputs, use_cache=False).logits
            output = model(inputs, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
    assert calls == {"query": 4, "key": 2}
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    if random_matrix and precision == torch.bfloat16:
        # This configuration already amplifies rounding in the original factory
        # path. Preserve that execution exactly without claiming FP32 accuracy.
        return
    # These bounds cover the original factory arithmetic at each precision.
    # Random-matrix inverses amplify BF16 rounding substantially; the exact
    # reference comparison above separately forbids any new execution drift.
    tolerances = (
        {torch.float32: 1e-5, torch.float16: 6e-3}
        if random_matrix
        else {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1e-2}
    )
    torch.testing.assert_close(output, original, atol=tolerances[precision], rtol=0)


@pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["hadamard", "random-hadamard"])
@pytest.mark.parametrize("existing", [False, True])
def test_matching_hadamard_precision_and_existing_composition(
    precision, kind, existing
):
    model = _model()
    reference = deepcopy(model)
    with torch.no_grad():
        original = model(torch.tensor([[1, 3, 7, 2]]), use_cache=False).logits
    if existing:
        _apply_with_reference(
            model,
            reference,
            TransformConfig(
                config_groups=dict([_scheme("first", kind, precision, randomize=True)])
            ),
        )
    _apply_with_reference(
        model,
        reference,
        TransformConfig(
            config_groups=dict([_scheme("next", kind, precision, inverse=True)])
        ),
    )
    _assert_outputs(model, reference, original, precision)


@pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "mismatch", ["permutation", "different_kind", "unpaired_inverse"]
)
def test_low_precision_wrong_basis_rejected(precision, mismatch):
    model = _model()
    before = _snapshot(model)
    if mismatch == "permutation":
        groups = [
            _scheme("q", "hadamard", precision, k=(), randomize=True),
            _scheme("k", "hadamard", precision, q=()),
        ]
    elif mismatch == "different_kind":
        groups = [
            _scheme("q", "random-hadamard", precision, k=()),
            _scheme("k", "hadamard", precision, q=()),
        ]
    else:
        groups = [
            _scheme("q", "random-hadamard", precision, k=(), inverse=True),
            _scheme("k", "random-hadamard", precision, q=()),
        ]
    with pytest.raises(ValueError, match="query.*key"):
        apply_transform_config(model, TransformConfig(config_groups=dict(groups)))
    _assert_unchanged(model, before)


@pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("existing", [False, True])
def test_random_matrix_inverse_composition(precision, existing):
    model = _model()
    reference = deepcopy(model)
    with torch.no_grad():
        original = model(torch.tensor([[1, 3, 7, 2]]), use_cache=False).logits
    groups = [
        _scheme("matrix", "random-matrix", precision),
        _scheme("inverse", "random-matrix", precision, inverse=True),
    ]
    _apply_with_reference(model, reference, TransformConfig(config_groups=dict(groups)))
    if existing:
        _apply_with_reference(
            model,
            reference,
            TransformConfig(
                config_groups=dict([_scheme("next", "hadamard", precision)])
            ),
        )
    _assert_outputs(model, reference, original, precision, random_matrix=True)


@pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inverse", [False, True])
def test_nonorthogonal_random_matrix_is_not_valid_qk_basis(precision, inverse):
    model = _model()
    before = _snapshot(model)
    with pytest.raises(ValueError, match="query.*key"):
        apply_transform_config(
            model,
            TransformConfig(
                config_groups=dict(
                    [_scheme("wrong", "random-matrix", precision, inverse=inverse)]
                )
            ),
        )
    _assert_unchanged(model, before)


def test_nonfinite_composition_bound_cannot_accept_an_invalid_basis():
    model = _model()
    groups = [
        _scheme("matrix", "random-matrix", torch.float64),
        _scheme("inverse", "random-matrix", torch.float64, inverse=True),
    ]
    apply_transform_config(model, TransformConfig(config_groups=dict(groups)))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, TransformBase):
                module.weight.fill_(1e200)
    before = _snapshot(model)
    with pytest.raises(ValueError, match="nonfinite|query.*key"):
        apply_transform_config(
            model,
            TransformConfig(
                config_groups=dict([_scheme("next", "hadamard", torch.float64)])
            ),
        )
    _assert_unchanged(model, before)
