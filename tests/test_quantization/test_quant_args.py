# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from pydantic import ValidationError


def _args(**kwargs):
    values = {
        "num_bits": 8,
        "type": QuantizationType.INT,
        "symmetric": True,
        "strategy": QuantizationStrategy.TENSOR,
        "dynamic": False,
        "zp_dtype": torch.int8,
    }
    values.update(kwargs)
    return QuantizationArgs(**values)


def test_defaults():
    default = QuantizationArgs()

    assert default.num_bits == 8
    assert default.type == QuantizationType.INT
    assert default.symmetric
    assert default.strategy is None
    assert default.group_size is None
    assert default.block_structure is None


def test_group():
    kwargs = {"strategy": "group", "group_size": 128}

    group = _args(**kwargs)
    assert group.strategy == QuantizationStrategy.GROUP
    assert group.group_size == kwargs["group_size"]

    with pytest.raises(ValueError):
        QuantizationArgs(strategy=QuantizationStrategy.GROUP, group_size=-1)

    args = _args(group_size=128, strategy="group")
    assert args.group_size == 128
    assert args.strategy == "group"

    with pytest.raises(ValueError):
        QuantizationArgs(strategy=QuantizationStrategy.GROUP)

    with pytest.raises(ValueError):
        QuantizationArgs(strategy="tensor", group_size=128)


def test_block():
    kwargs = {"strategy": "block", "block_structure": "2x4"}

    block = _args(**kwargs)
    assert block.strategy == QuantizationStrategy.BLOCK
    assert block.block_structure == [2, 4]
    assert block.block_structure != kwargs["block_structure"]  # "2x4" != [2, 4]


def test_block_structure_string_length_validation():
    # string and list forms must enforce the same [rows, cols] contract
    with pytest.raises(ValidationError):
        QuantizationArgs(strategy="block", block_structure="2x4x8")
    with pytest.raises(ValidationError):
        QuantizationArgs(strategy="block", block_structure=[2, 4, 8])


def test_block_structure_string_non_int():
    with pytest.raises(ValidationError):
        QuantizationArgs(strategy="block", block_structure="2xfoo")


@pytest.mark.parametrize(
    "block_structure",
    ([0, 4], [-1, 4], [4, 0], [4, -1], "0x4", "-1x4", "4x0", "4x-1"),
)
def test_block_structure_requires_positive_dimensions(block_structure):
    with pytest.raises(ValidationError, match="positive"):
        QuantizationArgs(strategy="block", block_structure=block_structure)


def test_strategy_is_not_inferred():
    with pytest.raises(ValidationError, match="group_size requires strategy"):
        QuantizationArgs(group_size=128)
    assert QuantizationArgs(group_size=-1).strategy is None


def test_observer_is_not_defaulted_by_format_schema():
    assert (
        QuantizationArgs(
            strategy="tensor_group", group_size=16, dynamic="local"
        ).observer
        is None
    )
    args = QuantizationArgs(
        strategy="tensor", dynamic=True, observer="static_minmax"
    )
    assert args.observer == "static_minmax"


def test_enums():
    assert _args(
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        actorder=ActivationOrdering.WEIGHT,
        group_size=1,
    ) == _args(type="InT", strategy="GROUP", actorder="weight", group_size=1)


def test_actorder():
    # test group inference with actorder
    args = _args(
        strategy="group", group_size=128, actorder=ActivationOrdering.GROUP
    )
    assert args.strategy == QuantizationStrategy.GROUP
    args = _args(
        strategy="group", group_size=128, actorder=ActivationOrdering.DYNAMIC
    )
    assert args.strategy == QuantizationStrategy.GROUP

    # test invalid pairings
    with pytest.raises(ValueError):
        QuantizationArgs(group_size=None, actorder="group")
    with pytest.raises(ValueError):
        QuantizationArgs(group_size=-1, actorder="group")
    with pytest.raises(ValueError):
        QuantizationArgs(strategy="tensor", actorder="group")

    # test boolean and none defaulting
    assert (
        _args(strategy="group", group_size=1, actorder=True).actorder
        == ActivationOrdering.GROUP
    )
    assert _args(strategy="group", group_size=1, actorder=False).actorder is None
    assert _args(strategy="group", group_size=1, actorder=None).actorder is None


def test_actorder_aliases():
    assert (
        ActivationOrdering.GROUP
        == ActivationOrdering.DYNAMIC
        == ActivationOrdering.GROUP
    )
    assert (
        ActivationOrdering.WEIGHT
        == ActivationOrdering.STATIC
        == ActivationOrdering.WEIGHT
    )

    assert ActivationOrdering.GROUP == "dynamic" == ActivationOrdering.GROUP
    assert ActivationOrdering.DYNAMIC == "dynamic" == ActivationOrdering.DYNAMIC
    assert ActivationOrdering.GROUP == "group" == ActivationOrdering.GROUP
    assert ActivationOrdering.DYNAMIC == "group" == ActivationOrdering.DYNAMIC

    assert ActivationOrdering.WEIGHT == "static" == ActivationOrdering.WEIGHT
    assert ActivationOrdering.STATIC == "static" == ActivationOrdering.STATIC
    assert ActivationOrdering.WEIGHT == "weight" == ActivationOrdering.WEIGHT
    assert ActivationOrdering.STATIC == "weight" == ActivationOrdering.STATIC

    assert ActivationOrdering.WEIGHT != "dynamic" != ActivationOrdering.WEIGHT
    assert ActivationOrdering.STATIC != "dynamic" != ActivationOrdering.STATIC
    assert ActivationOrdering.WEIGHT != "group" != ActivationOrdering.WEIGHT
    assert ActivationOrdering.STATIC != "group" != ActivationOrdering.STATIC
    assert ActivationOrdering.GROUP != "static" != ActivationOrdering.GROUP
    assert ActivationOrdering.DYNAMIC != "static" != ActivationOrdering.DYNAMIC
    assert ActivationOrdering.GROUP != "weight" != ActivationOrdering.GROUP
    assert ActivationOrdering.DYNAMIC != "weight" != ActivationOrdering.DYNAMIC


def test_invalid():
    with pytest.raises(ValidationError):
        QuantizationArgs(type="invalid")
    with pytest.raises(ValidationError):
        QuantizationArgs(strategy="invalid")
    with pytest.raises(ValidationError):
        QuantizationArgs(strategy=QuantizationStrategy.GROUP)


def test_serialize_args():
    """Test serialization of QuantizationArgs"""
    args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        symmetric=True,
        group_size=128,
        strategy=QuantizationStrategy.GROUP,
        dynamic=False,
        zp_dtype=torch.int8,
        actorder=ActivationOrdering.GROUP,
    )

    # Serialize to dict
    args_dict = args.model_dump()
    assert args_dict["num_bits"] == 4
    assert args_dict["type"] == "int"
    assert args_dict["symmetric"] is True
    assert args_dict["group_size"] == 128
    assert args_dict["strategy"] == "group"
    assert args_dict["actorder"] == "group"

    # Deserialize from dict
    reloaded = QuantizationArgs.model_validate(args_dict)
    assert reloaded == args
