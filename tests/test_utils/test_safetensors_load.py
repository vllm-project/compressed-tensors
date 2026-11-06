# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from unittest.mock import patch

import pytest
from compressed_tensors.utils.safetensors_load import get_nested_weight_mappings


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


@patch("compressed_tensors.utils.safetensors_load.cached_file")
@patch("compressed_tensors.utils.safetensors_load.list_repo_files")
def test_get_checkpoint_files_forwards_model_kwargs(list_repo_files, cached_file):
    """Hub stub lookups must forward model_kwargs to Hub file APIs."""
    from compressed_tensors.utils.safetensors_load import get_checkpoint_files

    list_repo_files.return_value = ["config.json", "model.safetensors"]
    cached_file.side_effect = lambda stub, file_path, **kwargs: f"/cache/{file_path}"
    model_kwargs = {"revision": "abc123", "token": "hf_token"}

    result = get_checkpoint_files("org/model", model_kwargs=model_kwargs)

    list_repo_files.assert_called_once_with(
        "org/model", revision="abc123", token="hf_token"
    )
    assert cached_file.call_count == 2
    for call in cached_file.call_args_list:
        assert call.kwargs.get("revision") == "abc123"
        assert call.kwargs.get("token") == "hf_token"
    assert result == {
        "config.json": "/cache/config.json",
        "model.safetensors": "/cache/model.safetensors",
    }


@patch("compressed_tensors.utils.safetensors_load.cached_file")
@patch("compressed_tensors.utils.safetensors_load.list_repo_files")
def test_get_checkpoint_files_default_model_kwargs_is_none(
    list_repo_files, cached_file
):
    """Omitting model_kwargs preserves today's default-branch Hub behaviour."""
    from compressed_tensors.utils.safetensors_load import get_checkpoint_files

    list_repo_files.return_value = ["config.json"]
    cached_file.return_value = "/cache/config.json"

    get_checkpoint_files("org/model")

    list_repo_files.assert_called_once_with("org/model")
    cached_file.assert_called_once_with("org/model", "config.json")


@patch("compressed_tensors.utils.safetensors_load.cached_file")
@patch("compressed_tensors.utils.safetensors_load.list_repo_files")
def test_get_checkpoint_files_local_path_ignores_model_kwargs(
    list_repo_files, cached_file, tmp_path
):
    """Local directories are walked on disk; model_kwargs must not hit the Hub."""
    from compressed_tensors.utils.safetensors_load import get_checkpoint_files

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")
    cached_file.side_effect = lambda stub, file_path, **kwargs: str(
        tmp_path / file_path
    )

    result = get_checkpoint_files(
        tmp_path, model_kwargs={"revision": "should-be-ignored", "token": "hf_token"}
    )

    list_repo_files.assert_not_called()
    for call in cached_file.call_args_list:
        assert call.kwargs == {}
    assert set(result) == {"config.json", "model.safetensors"}
    assert all(Path(path).is_file() for path in result.values())
