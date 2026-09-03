# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from compressed_tensors import (
    ParameterizedDefaultDict,
    get_nested_value,
    patch_attr,
    patch_attrs,
)
from compressed_tensors.utils.helpers import get_head_dim


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


@pytest.mark.parametrize(
    "key,default,expected_value",
    [
        ("c.d", -4, 4),
        ("c.e", -4, -4),
        ("b", -4, 1),
        ("d", -4, -4),
    ],
)
def test_get_nested_value(key, default, expected_value):
    a = {"b": 1, "c": {"d": 4}}

    assert get_nested_value(a, key, default) == expected_value


def test_get_head_dim():
    assert get_head_dim(SimpleNamespace(head_dim=256)) == 256
    assert get_head_dim(SimpleNamespace(hidden_size=1536, num_attention_heads=8)) == 192
    with pytest.raises(ValueError, match="Cannot determine head_dim"):
        get_head_dim(SimpleNamespace())


def test_get_head_dim_heterogeneous_config():
    # transformers>=5 heterogeneous (per-layer) configs raise
    # AmbiguousGlobalPerLayerAttributeError (a RuntimeError) when a per-layer
    # attribute such as head_dim is read on the global config, which broke the
    # previous hasattr-based lookup.
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers.PretrainedConfig, "per_layer_config"):
        pytest.skip("heterogeneous configs require transformers>=5")

    config = transformers.PretrainedConfig(
        hidden_size=1536,
        num_attention_heads=8,
        head_dim=256,
        num_hidden_layers=4,
    )
    config.per_layer_config = {1: {"head_dim": 128}}

    assert get_head_dim(config) == 256
