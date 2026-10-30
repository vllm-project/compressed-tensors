# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict, namedtuple
from dataclasses import dataclass

import pytest
import torch
from compressed_tensors.offload.utils import send_tensors
from tests.testing_utils import requires_gpu


Point = namedtuple("Point", ["x", "y"])


@dataclass
class Batch:
    tensor: torch.Tensor
    name: str


class MyDict(dict):
    pass


def _empty_tensors(device) -> list[torch.Tensor]:
    param = torch.nn.Parameter(torch.empty(0, device=device))
    param.attr = "value"
    return [
        torch.empty(0, device=device),
        torch.empty(1024, device=device)[:0],  # view of a larger allocation
        param,
    ]


@pytest.mark.unit
def test_send_tensors_keeps_tensors_already_on_device():
    tensors = [torch.zeros(2), torch.nn.Parameter(torch.zeros(2))]
    for tensor in tensors + _empty_tensors("cpu"):
        assert send_tensors(tensor, device="cpu") is tensor


@pytest.mark.unit
def test_send_tensors_moves_empty_tensors_to_meta():
    # tensors without elements have a null data pointer on every device, so they
    # must not be mistaken for tensors which are already on the target device
    for tensor in _empty_tensors("cpu"):
        moved = send_tensors(tensor, device="meta")
        assert moved is not tensor
        assert moved.is_meta and moved.shape == tensor.shape
        assert moved.__class__ is tensor.__class__
        assert moved.__dict__ == tensor.__dict__


@pytest.mark.unit
@requires_gpu
def test_send_tensors_moves_empty_tensors_between_devices():
    accel_device = torch.accelerator.current_accelerator()
    for source, target in ((torch.device("cpu"), accel_device), (accel_device, "cpu")):
        for tensor in _empty_tensors(source):
            moved = send_tensors(tensor, device=target)
            assert moved is not tensor
            assert moved.device.type == torch.device(target).type
            assert moved.__class__ is tensor.__class__


@pytest.mark.unit
def test_send_tensors_nested_containers():
    value = {"a": [torch.zeros(2), (torch.ones(2), 3)], "b": {"c": torch.zeros(1)}}
    moved = send_tensors(value, device="meta")

    assert moved["a"][0].is_meta
    assert moved["a"][1][0].is_meta
    assert moved["a"][1][1] == 3
    assert moved["b"]["c"].is_meta
    assert isinstance(moved["a"], list) and isinstance(moved["a"][1], tuple)


@pytest.mark.unit
def test_send_tensors_preserves_namedtuple():
    moved = send_tensors(Point(torch.zeros(2), torch.ones(2)), device="meta")

    assert isinstance(moved, Point)
    assert moved.x.is_meta and moved.y.is_meta


@pytest.mark.unit
def test_send_tensors_preserves_ordered_dict():
    moved = send_tensors(OrderedDict(b=torch.zeros(1), a=torch.ones(1)), device="meta")

    assert isinstance(moved, OrderedDict)
    assert list(moved.keys()) == ["b", "a"]
    assert all(tensor.is_meta for tensor in moved.values())


@pytest.mark.unit
def test_send_tensors_moves_tensors_inside_dataclass():
    moved = send_tensors({"batch": Batch(torch.zeros(2), "hi")}, device="meta")

    assert isinstance(moved["batch"], Batch)
    assert moved["batch"].tensor.is_meta
    assert moved["batch"].name == "hi"


@pytest.mark.unit
def test_send_tensors_moves_tensors_inside_dict_subclass():
    # pytree treats unregistered dict subclasses as leaves
    moved = send_tensors(MyDict(weight=torch.zeros(2)), device="meta")

    assert isinstance(moved, MyDict)
    assert moved["weight"].is_meta


@pytest.mark.unit
def test_send_tensors_keeps_non_tensor_leaves():
    value = {"none": None, "int": 1, "str": "hi", "float": 1.5}
    assert send_tensors(value, device="meta") == value


@pytest.mark.unit
def test_send_tensors_keeps_parameter_class_and_attrs():
    param = torch.nn.Parameter(torch.zeros(2))
    param.attr = "value"
    moved = send_tensors({"param": param}, device="meta")["param"]

    assert moved is not param
    assert moved.is_meta
    assert isinstance(moved, torch.nn.Parameter)
    assert moved.attr == "value"


@pytest.mark.unit
def test_send_tensors_keeps_dataclass_types():
    # `is_dataclass` is also true for dataclass types, which have no field values
    assert send_tensors({"cls": Batch}, device="meta") == {"cls": Batch}
