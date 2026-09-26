# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import patch

import pytest
import torch
from compressed_tensors.utils.safetensors_load import (
    get_nested_weight_mappings,
    get_tensor_metadata,
)
from safetensors.torch import save_file


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


class TestGetTensorMetadata:
    """
    Tests for the get_tensor_metadata function, covering single-file and
    multi-shard checkpoints.
    """

    def test_single_file(self, tmp_path):
        tensors = {
            "layer1.weight": torch.zeros(4, 8, dtype=torch.float16),
            "layer1.bias": torch.zeros(4, dtype=torch.bfloat16),
        }
        save_file(tensors, str(tmp_path / "model.safetensors"))
        (tmp_path / "config.json").write_text("{}")

        metadata = get_tensor_metadata(tmp_path)

        assert metadata == {
            "layer1.weight": {"dtype": "F16", "shape": [4, 8]},
            "layer1.bias": {"dtype": "BF16", "shape": [4]},
        }

    def test_multi_shard(self, tmp_path):
        shard1 = {
            "layer1.weight": torch.zeros(2, 3, dtype=torch.float32),
        }
        shard2 = {
            "layer2.weight": torch.zeros(5, dtype=torch.int8),
            "layer2.scale": torch.ones(1, dtype=torch.float8_e4m3fn),
        }
        save_file(shard1, str(tmp_path / "model-00001-of-00002.safetensors"))
        save_file(shard2, str(tmp_path / "model-00002-of-00002.safetensors"))
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {
                "layer1.weight": "model-00001-of-00002.safetensors",
                "layer2.weight": "model-00002-of-00002.safetensors",
                "layer2.scale": "model-00002-of-00002.safetensors",
            },
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        (tmp_path / "config.json").write_text("{}")

        metadata = get_tensor_metadata(tmp_path)

        assert metadata == {
            "layer1.weight": {"dtype": "F32", "shape": [2, 3]},
            "layer2.weight": {"dtype": "I8", "shape": [5]},
            "layer2.scale": {"dtype": "F8_E4M3", "shape": [1]},
        }

    def test_missing_shard_raises(self, tmp_path):
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {"layer1.weight": "missing.safetensors"},
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        (tmp_path / "config.json").write_text("{}")

        with pytest.raises(ValueError):
            get_tensor_metadata(tmp_path)
