# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from compressed_tensors import ParameterizedDefaultDict, patch_attr, patch_attrs


def test_patch_attr():
    # patch, original value
    obj = SimpleNamespace()
    obj.attribute = "original"
    with patch_attr(obj, "attribute", "patched"):
        assert obj.attribute == "patched"
        obj.attribute = "modified"
    assert obj.attribute == "original"

    # patch, no original attribute
    obj = SimpleNamespace()
    with patch_attr(obj, "attribute", "patched"):
        assert obj.attribute == "patched"
        obj.attribute = "modified"
    assert not hasattr(obj, "attribute")


def test_patch_attrs():
    num_objs = 4
    objs = [SimpleNamespace() for _ in range(num_objs)]
    for idx, obj in enumerate(objs):
        if idx % 2 == 0:
            obj.attribute = f"original_{idx}"
    with patch_attrs(objs, "attribute", [f"patched_{idx}" for idx in range(num_objs)]):
        for idx, obj in enumerate(objs):
            assert obj.attribute == f"patched_{idx}"
            obj.attribute = "modified"
    for idx, obj in enumerate(objs):
        if idx % 2 == 0:
            assert obj.attribute == f"original_{idx}"
        else:
            assert not hasattr(obj, "attribute")


def test_parameterized_default_dict():
    def add_one(value):
        return value + 1

    add_dict = ParameterizedDefaultDict(add_one)
    assert add_dict[0] == 1
    assert add_dict[1] == 2

    def sum_vals(a, b):
        return a + b

    sum_dict = ParameterizedDefaultDict(sum_vals)
    assert sum_dict[0, 1] == 1
    assert sum_dict[5, 7] == 12
