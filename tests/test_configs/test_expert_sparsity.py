# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from compressed_tensors.config import ExpertSparsityConfig, MoESparsityMethod
from pydantic import ValidationError


def test_moe_method_values():
    assert MoESparsityMethod("prune") == MoESparsityMethod.PRUNE
    assert MoESparsityMethod("novice") == MoESparsityMethod.NOVICE


def test_moe_method_invalid():
    with pytest.raises(ValueError):
        MoESparsityMethod("invalid")


@pytest.mark.parametrize("method", ["prune", "novice"])
def test_expert_sparsity_basic(method):
    cfg = ExpertSparsityConfig(
        method=method,
        num_experts=[None, None, 129, 253, 128, 128],
    )
    assert cfg.method == MoESparsityMethod(method)
    assert cfg.num_experts == [None, None, 129, 253, 128, 128]


def test_expert_sparsity_accepts_enum_method():
    cfg = ExpertSparsityConfig(
        method=MoESparsityMethod.PRUNE,
        num_experts=[None, 128],
    )
    assert cfg.method == MoESparsityMethod.PRUNE


def test_expert_sparsity_json_roundtrip():
    cfg = ExpertSparsityConfig(
        method="novice",
        num_experts=[None, 129, 253, 128],
    )
    dumped = cfg.model_dump(mode="json")
    assert dumped["method"] == "novice"
    assert dumped["num_experts"] == [None, 129, 253, 128]

    restored = ExpertSparsityConfig.model_validate(dumped)
    assert restored.method == MoESparsityMethod.NOVICE
    assert restored.num_experts == [None, 129, 253, 128]


def test_expert_sparsity_invalid_method():
    with pytest.raises(ValidationError):
        ExpertSparsityConfig(method="bad", num_experts=[None, 1])


def test_expert_sparsity_bool_rejected():
    # bool is a subclass of int but should not be a valid expert count
    with pytest.raises(ValidationError):
        ExpertSparsityConfig(method="prune", num_experts=[None, True])
