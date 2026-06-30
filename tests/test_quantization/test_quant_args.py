# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch._dynamo
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA
from pydantic import ValidationError
from torch._dynamo.utils import counters


def test_defaults():
    default = QuantizationArgs()

    assert default.num_bits == 8
    assert default.type == QuantizationType.INT
    assert default.symmetric
    assert default.strategy == QuantizationStrategy.TENSOR
    assert default.group_size is None
    assert default.block_structure is None


def test_group():
    kwargs = {"strategy": "group", "group_size": 128}

    group = QuantizationArgs(**kwargs)
    assert group.strategy == QuantizationStrategy.GROUP
    assert group.group_size == kwargs["group_size"]

    with pytest.raises(ValueError):
        QuantizationArgs(strategy=QuantizationStrategy.GROUP, group_size=-1)

    args = QuantizationArgs(group_size=128, strategy="group")
    assert args.group_size == 128
    assert args.strategy == "group"

    with pytest.raises(ValueError):
        QuantizationArgs(strategy=QuantizationStrategy.GROUP)

    with pytest.raises(ValueError):
        QuantizationArgs(strategy="tensor", group_size=128)


def test_block():
    kwargs = {"strategy": "block", "block_structure": "2x4"}

    block = QuantizationArgs(**kwargs)
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


def test_infer_strategy():
    args = QuantizationArgs(group_size=128)
    assert args.strategy == QuantizationStrategy.GROUP

    args = QuantizationArgs(group_size=-1)
    assert args.strategy == QuantizationStrategy.CHANNEL


def test_enums():
    assert QuantizationArgs(
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        actorder=ActivationOrdering.WEIGHT,
        group_size=1,
    ) == QuantizationArgs(type="InT", strategy="GROUP", actorder="weight", group_size=1)


def test_actorder():
    # test group inference with actorder
    args = QuantizationArgs(group_size=128, actorder=ActivationOrdering.GROUP)
    assert args.strategy == QuantizationStrategy.GROUP
    args = QuantizationArgs(group_size=128, actorder=ActivationOrdering.DYNAMIC)
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
        QuantizationArgs(group_size=1, actorder=True).actorder
        == ActivationOrdering.GROUP
    )
    assert QuantizationArgs(group_size=1, actorder=False).actorder is None
    assert QuantizationArgs(group_size=1, actorder=None).actorder is None


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


def test_cast_to_fp4_no_recompile_across_ranks():
    # https://github.com/vllm-project/compressed-tensors/issues/734
    # rank-varying inputs must not recompile the compiled fp4 rounding core
    torch._dynamo.reset()
    counters.clear()

    for shape in [(16,), (4, 16), (2, 4, 16), (2, 2, 4, 16), (2, 2, 2, 4, 16)]:
        FP4_E2M1_DATA.cast_to_fp4(torch.randn(*shape))

    # must not grow one graph per rank: pre-fix this hit 5 (one per distinct
    # rank) and exhausted recompile_limit; the fix keeps it bounded. Lower
    # bound guards against a vacuous pass if the compiled core never runs.
    assert 1 <= counters["stats"]["unique_graphs"] < 5


def test_cast_to_fp4_boundary_values():
    x = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, -0.75, -3.5]
    )
    expected = torch.tensor(
        [0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -1.0, -4.0]
    )
    assert torch.equal(FP4_E2M1_DATA.cast_to_fp4(x), expected)

    x = torch.randn(2, 3, 4, 5)
    assert FP4_E2M1_DATA.cast_to_fp4(x).shape == x.shape


def test_cast_to_fp4_degenerate_shapes():
    # MoE experts can receive zero routed tokens -> numel 0 activations
    for t in [torch.randn(0), torch.randn(1), torch.tensor(3.0), torch.randn(1, 0, 4)]:
        out = FP4_E2M1_DATA.cast_to_fp4(t)
        assert out.shape == t.shape
