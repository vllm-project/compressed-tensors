# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
    round_to_quantized_type_dtype,
)
from compressed_tensors.quantization.utils import calculate_qparams
from pydantic import ValidationError


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


@pytest.mark.parametrize(
    "block_structure",
    ([0, 4], [-1, 4], [4, 0], [4, -1], "0x4", "-1x4", "4x0", "4x-1"),
)
def test_block_structure_requires_positive_dimensions(block_structure):
    with pytest.raises(ValidationError, match="positive"):
        QuantizationArgs(strategy="block", block_structure=block_structure)


def test_infer_strategy():
    args = QuantizationArgs(group_size=128)
    assert args.strategy == QuantizationStrategy.GROUP

    args = QuantizationArgs(group_size=-1)
    assert args.strategy == QuantizationStrategy.CHANNEL


def test_observer_is_not_resolved():
    assert QuantizationArgs().observer is None
    assert QuantizationArgs(observer="static_minmax").observer == "static_minmax"


def test_enums():
    assert QuantizationArgs(
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        actorder=ActivationOrdering.WEIGHT,
        group_size=1,
    ) == QuantizationArgs(type="InT", strategy="GROUP", actorder="weight", group_size=1)


def test_actorder():
    # test that GROUP/DYNAMIC/True now raise ValueError
    with pytest.raises(ValueError, match="removed"):
        QuantizationArgs(group_size=128, actorder="group")
    with pytest.raises(ValueError, match="removed"):
        QuantizationArgs(group_size=128, actorder="dynamic")
    with pytest.raises(ValueError, match="removed"):
        QuantizationArgs(group_size=1, actorder=True)

    # test WEIGHT still works
    args = QuantizationArgs(group_size=128, actorder=ActivationOrdering.WEIGHT)
    assert args.strategy == QuantizationStrategy.GROUP
    assert args.actorder == ActivationOrdering.WEIGHT

    # test STATIC alias still works
    args = QuantizationArgs(group_size=128, actorder="static")
    assert args.actorder == ActivationOrdering.WEIGHT

    # test boolean False and None defaulting
    assert QuantizationArgs(group_size=1, actorder=False).actorder is None
    assert QuantizationArgs(group_size=1, actorder=None).actorder is None


def test_actorder_aliases():
    assert (
        ActivationOrdering.WEIGHT
        == ActivationOrdering.STATIC
        == ActivationOrdering.WEIGHT
    )

    assert ActivationOrdering.WEIGHT == "static" == ActivationOrdering.WEIGHT
    assert ActivationOrdering.STATIC == "static" == ActivationOrdering.STATIC
    assert ActivationOrdering.WEIGHT == "weight" == ActivationOrdering.WEIGHT
    assert ActivationOrdering.STATIC == "weight" == ActivationOrdering.STATIC


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
        actorder=ActivationOrdering.WEIGHT,
    )

    # Serialize to dict
    args_dict = args.model_dump()
    assert args_dict["num_bits"] == 4
    assert args_dict["type"] == "int"
    assert args_dict["symmetric"] is True
    assert args_dict["group_size"] == 128
    assert args_dict["strategy"] == "group"
    assert args_dict["actorder"] == "weight"

    # Deserialize from dict
    reloaded = QuantizationArgs.model_validate(args_dict)
    assert reloaded == args


def test_default_fields_not_serialized():
    """
    Fields actorder, block_structure, observer, observer_kwargs, scale_dtype,
    and zp_dtype are omitted from model_dump() when not explicitly set.

    Before this change, model_dump() on QuantizationArgs() would produce:
        {
            "num_bits": 8, "type": "int", "symmetric": True,
            "group_size": None, "strategy": "tensor", "dynamic": False,
            "actorder": None,
            "block_structure": None,
            "observer": None,
            "observer_kwargs": {},
            "scale_dtype": None,
            "zp_dtype": None,  # None because symmetric=True
        }

    After this change, model_dump() on QuantizationArgs() produces:
        {
            "num_bits": 8, "type": "int", "symmetric": True,
            "group_size": None, "strategy": "tensor", "dynamic": False,
        }
    """
    args = QuantizationArgs()
    serialized = args.model_dump()

    assert "actorder" not in serialized
    assert "block_structure" not in serialized
    assert "observer" not in serialized
    assert "observer_kwargs" not in serialized
    assert "scale_dtype" not in serialized
    assert "zp_dtype" not in serialized

    # Core fields are still present
    assert serialized["num_bits"] == 8
    assert serialized["type"] == "int"
    assert serialized["strategy"] == "tensor"


def test_explicit_fields_are_serialized():
    """Explicitly set optional fields appear in model_dump()."""
    args = QuantizationArgs(
        group_size=128,
        actorder=ActivationOrdering.WEIGHT,
        observer="minmax",
        observer_kwargs={"percentile": 99.9},
        scale_dtype=torch.float16,
    )
    serialized = args.model_dump()

    assert serialized["actorder"] == "weight"
    assert serialized["observer"] == "minmax"
    assert serialized["observer_kwargs"] == {"percentile": 99.9}
    assert "scale_dtype" in serialized


def test_explicit_zp_dtype_serialized_when_asymmetric():
    """Explicitly set zp_dtype is serialized for asymmetric quantization."""
    args = QuantizationArgs(symmetric=False, zp_dtype=torch.int16)
    serialized = args.model_dump()
    assert "zp_dtype" in serialized
    assert serialized["zp_dtype"] == "torch.int16"


def test_zp_dtype_not_serialized_when_symmetric():
    """zp_dtype is omitted when symmetric=True even if explicitly set."""
    args = QuantizationArgs(symmetric=True, zp_dtype=torch.int8)
    serialized = args.model_dump()
    assert "zp_dtype" not in serialized


def test_round_trip_with_optional_fields_omitted():
    """Serializing and deserializing default args produces equivalent args."""
    for args in (
        QuantizationArgs(),
        QuantizationArgs(group_size=128, actorder=ActivationOrdering.WEIGHT),
        QuantizationArgs(num_bits=4, type=QuantizationType.INT, group_size=128),
    ):
        reloaded = QuantizationArgs.model_validate(args.model_dump())
        assert args == reloaded


@pytest.mark.parametrize(
    "tensor_dtype,target_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
        (torch.float16, torch.int32),
    ],
)
def test_round_to_wider_dtype_does_not_overflow(tensor_dtype, target_dtype):
    # the target dtype's bounds are not representable in the input dtype, which
    # used to make torch.clamp raise an overflow error on CPU
    tensor = torch.tensor([-3.0, 0.0, 1.25, 7.0], dtype=tensor_dtype)
    rounded = round_to_quantized_type_dtype(tensor, dtype=target_dtype)
    if torch.is_floating_point(torch.tensor([], dtype=target_dtype)):
        expected = tensor
    else:
        expected = torch.tensor([-3.0, 0.0, 1.0, 7.0], dtype=tensor_dtype)
    assert rounded.dtype == tensor_dtype
    assert torch.equal(rounded, expected)


def test_round_to_dtype_still_clamps_representable_bounds():
    # bounds that fit in the input dtype must still be applied
    tensor = torch.tensor([-float("inf"), 2.0, float("inf")], dtype=torch.bfloat16)
    rounded = round_to_quantized_type_dtype(tensor, dtype=torch.bfloat16)
    bf16 = torch.finfo(torch.bfloat16)
    assert torch.equal(
        rounded, torch.tensor([bf16.min, 2.0, bf16.max], dtype=torch.bfloat16)
    )

    ints = round_to_quantized_type_dtype(
        torch.tensor([-300.0, 2.25, 300.0], dtype=torch.bfloat16), dtype=torch.int8
    )
    assert torch.equal(ints, torch.tensor([-128, 2, 127], dtype=torch.bfloat16))


def test_calculate_qparams_bfloat16_with_float32_scale_dtype():
    args = QuantizationArgs(
        num_bits=8, strategy="token", dynamic=True, scale_dtype=torch.float32
    )
    min_vals = torch.tensor([[-2.0], [-0.5]], dtype=torch.bfloat16)
    max_vals = torch.tensor([[1.0], [4.0]], dtype=torch.bfloat16)
    scales, zero_points = calculate_qparams(min_vals, max_vals, args)
    assert scales.dtype == torch.bfloat16
    expected = (torch.tensor([[2.0], [4.0]]) / 127.5).to(torch.bfloat16)
    assert torch.equal(scales, expected)
    assert torch.equal(zero_points, torch.zeros_like(zero_points, dtype=torch.int8))
