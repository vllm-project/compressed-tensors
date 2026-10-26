# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.transform import TransformConfig, apply_transform_config
from compressed_tensors.transform.factory.base import TransformBase
from tests.test_modeling.test_attention_application import (
    _assert_unchanged,
    _model,
    _rotation,
    _snapshot,
)
from tests.test_modeling.test_attention_transform_precision import _scheme


@pytest.mark.parametrize("invalid", [False, True])
def test_attention_basis_validation_materializes_only_metadata_on_cpu(invalid):
    from compressed_tensors.transform.attention import validate_attention_transforms

    model = _model()
    name, scheme = _rotation(q=(0, 1, 2, 3), k=() if invalid else (0, 1))
    config = TransformConfig(config_groups={name: scheme})
    before = _snapshot(model)
    rng = torch.get_rng_state().clone()
    with torch.device("meta"):
        if invalid:
            with pytest.raises(ValueError, match="query.*key"):
                validate_attention_transforms(model, config)
        else:
            validate_attention_transforms(model, config)
        assert torch.empty(0).device.type == "meta"
    assert torch.empty(0).device.type == "cpu"
    assert torch.equal(rng, torch.get_rng_state())
    _assert_unchanged(model, before)


@pytest.mark.parametrize("precision", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("corruption", ["underflow", "nonfinite"])
def test_existing_inverse_basis_rejects_stored_underflow_and_nonfinite_values(
    precision, corruption
):
    model = _model()
    config = TransformConfig(
        config_groups=dict(
            [
                _scheme("matrix", "random-matrix", precision),
                _scheme("inverse", "random-matrix", precision, inverse=True),
            ]
        )
    )
    apply_transform_config(model, config)
    replacement = (
        torch.finfo(precision).tiny * torch.finfo(precision).eps / 4
        if corruption == "underflow"
        else float("inf")
    )
    if corruption == "underflow":
        assert torch.tensor(replacement, dtype=precision).item() == 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, TransformBase) and module.args.inverse:
                module.weight.fill_(replacement)
    before = _snapshot(model)
    with pytest.raises(ValueError, match="nonfinite|query.*key"):
        apply_transform_config(
            model,
            TransformConfig(
                config_groups=dict([_scheme("next", "hadamard", precision)])
            ),
        )
    _assert_unchanged(model, before)
