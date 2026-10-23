# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import patch

import pytest
from compressed_tensors.utils.safetensors_load import (
    get_nested_weight_mappings,
    get_quantization_config,
)


mock_weight_mappings = {
    "layer1.weight": "file1",
    "layer1.bias": "file2",
    "layer2.weight": "file3",
    "layer2.bias": "file4",
    "layer3.weight": "file5",
}


@pytest.fixture
def mock_get_weight_mappings():
    with patch(
        "compressed_tensors.utils.safetensors_load.get_weight_mappings",
        return_value=mock_weight_mappings,
    ):
        yield


@pytest.mark.usefixtures("mock_get_weight_mappings")
class TestGetNestedWeightMappings:
    """
    Tests for the get_nested_weight_mappings function
    in different scenarios, such as single and multiple
    parameters to nest, and returning other parameters
    """

    def test_single_param(self):
        params_to_nest = ["weight"]
        result = get_nested_weight_mappings("dummy_path", params_to_nest)
        expected = {
            "layer1": {"weight": "file1"},
            "layer2": {"weight": "file3"},
            "layer3": {"weight": "file5"},
        }
        assert result == expected

    def test_multiple_params(self):
        params_to_nest = ["weight", "bias"]
        result = get_nested_weight_mappings("dummy_path", params_to_nest)
        expected = {
            "layer1": {"weight": "file1", "bias": "file2"},
            "layer2": {"weight": "file3", "bias": "file4"},
            "layer3": {"weight": "file5"},
        }
        assert result == expected

    def test_return_other_params(self):
        params_to_nest = ["weight"]
        result, other_params = get_nested_weight_mappings(
            "dummy_path", params_to_nest, return_unmatched_params=True
        )
        expected_nested = {
            "layer1": {"weight": "file1"},
            "layer2": {"weight": "file3"},
            "layer3": {"weight": "file5"},
        }
        expected_other = {
            "layer1.bias": "file2",
            "layer2.bias": "file4",
        }
        assert result == expected_nested
        assert other_params == expected_other


@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, None),
        ({"text_config": None}, None),
        ({"quantization_config": {"source": "top"}}, {"source": "top"}),
        (
            {
                "quantization_config": {"source": "top"},
                "text_config": {"quantization_config": {"source": "text"}},
                "compression_config": {"source": "legacy"},
            },
            {"source": "top"},
        ),
        (
            {
                "quantization_config": None,
                "text_config": {"quantization_config": {"source": "text"}},
                "compression_config": {"source": "legacy"},
            },
            {"source": "text"},
        ),
        (
            {"text_config": {"quantization_config": {"source": "text"}}},
            {"source": "text"},
        ),
        (
            {
                "quantization_config": None,
                "compression_config": {"source": "legacy"},
            },
            {"source": "legacy"},
        ),
        (
            {
                "text_config": {"quantization_config": None},
                "compression_config": {"source": "legacy"},
            },
            {"source": "legacy"},
        ),
        (
            {"text_config": None, "compression_config": {"source": "legacy"}},
            {"source": "legacy"},
        ),
        ({"compression_config": {"source": "legacy"}}, {"source": "legacy"}),
        (
            {"quantization_config": {}, "compression_config": {"source": "legacy"}},
            {},
        ),
        (
            {
                "text_config": {"quantization_config": {}},
                "compression_config": {"source": "legacy"},
            },
            {},
        ),
    ],
)
def test_get_quantization_config_falls_back_only_for_null_values(
    tmp_path, config, expected
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    assert get_quantization_config(config_path) == expected
