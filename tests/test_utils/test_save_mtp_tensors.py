# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

import pytest
import torch
from compressed_tensors.utils.safetensors_load import save_mtp_tensors_to_checkpoint
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_safetensors(path: str, tensors: dict) -> None:
    """Save a dict of tensors as a safetensors file."""
    save_file({k: v.contiguous() for k, v in tensors.items()}, path)


def _read_safetensors(path: str) -> dict:
    """Load all tensors from a safetensors file into a plain dict."""
    result = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            result[key] = f.get_tensor(key)
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def source_dir(tmp_path):
    """
    Multi-shard source checkpoint containing both regular model tensors and MTP
    tensors (prefixed with "mtp"), along with a model.safetensors.index.json
    that maps each key to its shard file.
    """
    src = tmp_path / "source"
    src.mkdir()

    shard1 = {
        "model.layer0.weight": torch.randn(4, 4),
        "mtp.layer0.weight": torch.randn(3, 3),
    }
    shard2 = {
        "model.layer1.weight": torch.randn(4, 4),
        "mtp.layer1.weight": torch.randn(3, 3),
    }

    _make_safetensors(str(src / "model-00001-of-00002.safetensors"), shard1)
    _make_safetensors(str(src / "model-00002-of-00002.safetensors"), shard2)

    index = {
        "metadata": {},
        "weight_map": {
            "model.layer0.weight": "model-00001-of-00002.safetensors",
            "mtp.layer0.weight": "model-00001-of-00002.safetensors",
            "model.layer1.weight": "model-00002-of-00002.safetensors",
            "mtp.layer1.weight": "model-00002-of-00002.safetensors",
        },
    }
    with open(src / SAFE_WEIGHTS_INDEX_NAME, "w") as f:
        json.dump(index, f)

    return src


@pytest.fixture()
def dest_dir_with_index(tmp_path):
    """
    Destination checkpoint directory pre-populated with a single model shard
    and a model.safetensors.index.json. Represents the common quantized-model
    output where an index already exists and should be updated in place.
    """
    dest = tmp_path / "dest_index"
    dest.mkdir()

    tensors = {"model.layer0.weight": torch.randn(4, 4)}
    shard_name = "model-00001-of-00001.safetensors"
    _make_safetensors(str(dest / shard_name), tensors)

    index = {
        "metadata": {},
        "weight_map": {"model.layer0.weight": shard_name},
    }
    with open(dest / SAFE_WEIGHTS_INDEX_NAME, "w") as f:
        json.dump(index, f)

    return dest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveMtpTensorsToCheckpoint:
    """
    Tests for save_mtp_tensors_to_checkpoint, which extracts tensors whose
    keys start with a given prefix (default "mtp") from a source checkpoint and
    appends them as a new shard to a destination checkpoint directory, updating
    the destination's model.safetensors.index.json accordingly.
    """

    def test_mtp_tensors_saved_correctly(self, source_dir, dest_dir_with_index):
        """
        Verify that the MTP shard is created in the destination directory and
        that its tensor values are numerically identical to those in the source.
        Also checks that only MTP-prefixed keys are included in the shard (no
        regular model weights leak through).
        """
        # Collect expected MTP tensors directly from source shards
        expected = {}
        for shard in (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ):
            with safe_open(str(source_dir / shard), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("mtp"):
                        expected[key] = f.get_tensor(key)

        save_mtp_tensors_to_checkpoint(str(source_dir), str(dest_dir_with_index))

        mtp_shard_path = str(dest_dir_with_index / "model_mtp.safetensors")
        assert os.path.exists(mtp_shard_path)

        saved = _read_safetensors(mtp_shard_path)
        assert set(saved.keys()) == set(expected.keys())
        for key in expected:
            assert torch.equal(saved[key], expected[key])
        assert all(k.startswith("mtp") for k in saved)

    def test_index_updated(self, source_dir, dest_dir_with_index):
        """
        Verify that after the call the destination index file contains MTP keys
        pointing to the new shard, existing keys are preserved, and
        metadata.total_size reflects the sizes of all shards in the weight_map.
        """
        save_mtp_tensors_to_checkpoint(str(source_dir), str(dest_dir_with_index))

        with open(dest_dir_with_index / SAFE_WEIGHTS_INDEX_NAME) as f:
            index = json.load(f)

        weight_map = index["weight_map"]

        # MTP keys added, pointing to the new shard
        assert weight_map.get("mtp.layer0.weight") == "model_mtp.safetensors"
        assert weight_map.get("mtp.layer1.weight") == "model_mtp.safetensors"

        # Pre-existing keys must not be removed
        assert "model.layer0.weight" in weight_map

        # total_size must equal the sum of all referenced shard sizes on disk
        expected_size = sum(
            os.path.getsize(dest_dir_with_index / s) for s in set(weight_map.values())
        )
        assert index["metadata"]["total_size"] == expected_size

    def test_single_shard_dest_creates_index(self, source_dir, tmp_path):
        """
        Verify that when the destination has no index file (only a
        model.safetensors), the function synthesises one that includes both the
        original model keys and the newly added MTP keys.
        """
        dest = tmp_path / "dest_single"
        dest.mkdir()
        _make_safetensors(
            str(dest / SAFE_WEIGHTS_NAME), {"model.layer0.weight": torch.randn(4, 4)}
        )

        save_mtp_tensors_to_checkpoint(str(source_dir), str(dest))

        index_path = dest / SAFE_WEIGHTS_INDEX_NAME
        assert index_path.exists()

        with open(index_path) as f:
            index = json.load(f)

        assert index["weight_map"].get("model.layer0.weight") == SAFE_WEIGHTS_NAME
        assert index["weight_map"].get("mtp.layer0.weight") == "model_mtp.safetensors"

    def test_no_mtp_tensors_raises(self, dest_dir_with_index, tmp_path):
        """
        Verify that a ValueError is raised when the source checkpoint has no
        tensors matching the MTP prefix, preventing a silent empty-shard write.
        """
        src = tmp_path / "src_no_mtp"
        src.mkdir()
        _make_safetensors(
            str(src / SAFE_WEIGHTS_NAME), {"model.weight": torch.randn(4, 4)}
        )

        with pytest.raises(ValueError, match="No tensors with prefix"):
            save_mtp_tensors_to_checkpoint(str(src), str(dest_dir_with_index))

    def test_missing_dest_files_raises(self, source_dir, tmp_path):
        """
        Verify that a FileNotFoundError is raised when the destination directory
        contains neither model.safetensors.index.json nor model.safetensors.
        """
        empty_dest = tmp_path / "dest_empty"
        empty_dest.mkdir()

        with pytest.raises(FileNotFoundError):
            save_mtp_tensors_to_checkpoint(str(source_dir), str(empty_dest))

    def test_custom_mtp_prefix(self, dest_dir_with_index, tmp_path):
        """
        Verify that when a non-default mtp_prefix is supplied, only tensors
        whose keys start with that prefix are extracted. Tensors matching the
        default "mtp" prefix but not the custom one must be excluded.
        """
        src = tmp_path / "src_custom"
        src.mkdir()
        _make_safetensors(
            str(src / SAFE_WEIGHTS_NAME),
            {
                "model.weight": torch.randn(4, 4),
                "speculative.layer0.weight": torch.randn(3, 3),
                "mtp.layer0.weight": torch.randn(3, 3),  # should be ignored
            },
        )

        save_mtp_tensors_to_checkpoint(
            str(src), str(dest_dir_with_index), mtp_prefix="speculative"
        )

        saved = _read_safetensors(str(dest_dir_with_index / "model_mtp.safetensors"))
        assert set(saved.keys()) == {"speculative.layer0.weight"}
