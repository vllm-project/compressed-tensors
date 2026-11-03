# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import torch
from compressed_tensors.offload import offload_module, print_offload_folder
from compressed_tensors.offload.cache import disk
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.cache.dist_disk import DistributedDiskCache
from compressed_tensors.offload.utils import as_single_threaded
from safetensors.torch import save_file


@pytest.mark.unit
@pytest.mark.parametrize("checkpoint_symlink", [False, True])
def test_deferred_deletion_preserves_replacement(
    tmp_path, monkeypatch, checkpoint_symlink
):
    monkeypatch.setattr(DiskCache, "index", {})
    monkeypatch.setattr(DistributedDiskCache, "_deferred_file_deletions", set())
    # Simulate reuse of a retired tensor's ID without depending on the allocator.
    monkeypatch.setattr(disk, "id", lambda tensor: 42, raising=False)
    offload_dir = tmp_path / "offload"
    offload_dir.mkdir()
    cache = DistributedDiskCache("cpu", offload_dir=str(offload_dir))

    with as_single_threaded():
        if checkpoint_symlink:
            checkpoint = tmp_path / "checkpoint.safetensors"
            save_file({"weight": torch.zeros(8)}, checkpoint)
            offloaded = torch.empty(8, device="meta")
            DiskCache.create_checkpoint_symlink(
                offloaded,
                {
                    "safetensors_file": str(checkpoint),
                    "weight_name": "weight",
                    "dtype": "float32",
                },
                str(offload_dir),
            )
            cache["old"] = offloaded
            del offloaded
        else:
            cache["old"] = torch.zeros(8)

        old_path = Path(cache.index[cache.offloaded_values["old"]]["safetensors_file"])
        del cache["old"]
        assert old_path.exists()

        expected = torch.ones(8)
        cache["replacement"] = expected
        offloaded = cache.offloaded_values["replacement"]
        assert torch.equal(cache.stage(offloaded), expected)

        DistributedDiskCache.flush_deferred_file_deletions()

        assert torch.equal(cache.stage(offloaded), expected)
        assert not old_path.exists()
        if checkpoint_symlink:
            assert checkpoint.exists()


@pytest.mark.unit
def test_explicit_offload_reuses_recorded_path(tmp_path, monkeypatch):
    monkeypatch.setattr(DiskCache, "index", {})
    cache = DiskCache("cpu", offload_dir=str(tmp_path))
    offloaded = cache.offload(torch.zeros(8))
    original_path = cache.index[offloaded]["safetensors_file"]

    assert cache.offload(torch.ones(8), offloaded=offloaded) is offloaded

    assert cache.index[offloaded]["safetensors_file"] == original_path
    assert len(list(tmp_path.iterdir())) == 1
    assert torch.equal(cache.stage(offloaded), torch.ones(8))


@pytest.mark.unit
@pytest.mark.parametrize("folder_path", ["absolute", "relative", "symlink"])
def test_print_offload_folder_uses_recorded_paths(
    tmp_path, monkeypatch, capsys, folder_path
):
    monkeypatch.setattr(DiskCache, "index", {})
    module = torch.nn.Linear(2, 2)
    offload_module(module, "cpu", "disk", offload_dir=str(tmp_path))

    monkeypatch.chdir(tmp_path.parent)
    if folder_path == "symlink":
        offload_folder = tmp_path.parent / f"{tmp_path.name}-alias"
        offload_folder.symlink_to(tmp_path, target_is_directory=True)
    else:
        offload_folder = tmp_path.name if folder_path == "relative" else str(tmp_path)
    print_offload_folder(module, offload_folder)

    output = capsys.readouterr().out
    assert "weight" in output
    assert "bias" in output
    assert "Unknown files" not in output
