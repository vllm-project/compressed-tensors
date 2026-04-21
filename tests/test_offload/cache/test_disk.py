# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
from compressed_tensors.offload.cache.disk import DiskCache
from safetensors import safe_open
from tests.test_offload.cache.helpers import (
    _test_delete,
    _test_disable_offloading,
    _test_disable_onloading,
    _test_garbage_collect,
    _test_offload,
    _test_onload,
    _test_onloading,
    _test_shared_attributes,
    _test_tensor_subclass,
    _test_update_offload,
)
from tests.test_offload.conftest import assert_tensor_equal
from tests.testing_utils import requires_gpu


@pytest.fixture()
def onload_device():
    return torch.device("cuda")


@pytest.fixture()
def offload_device():
    return "disk"


@pytest.mark.unit
@requires_gpu
def test_delete(offload_device, onload_device, offload_cache):
    _test_delete(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_disable_offloading(offload_device, onload_device, offload_cache):
    _test_disable_offloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_disable_onloading(offload_device, onload_device, offload_cache):
    _test_disable_onloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_garbage_collect(offload_device, onload_device, offload_cache):
    _test_garbage_collect(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_offload(offload_device, onload_device, offload_cache):
    _test_offload(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
@requires_gpu
def test_onload(offload_device, onload_device, offload_cache):
    _test_onload(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_onloading(offload_device, onload_device, offload_cache):
    _test_onloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_shared_attributes(offload_device, onload_device, offload_cache):
    _test_shared_attributes(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_tensor_subclass(offload_device, onload_device, offload_cache):
    _test_tensor_subclass(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_update_offload(offload_device, onload_device, offload_cache):
    _test_update_offload(offload_device, onload_device, offload_cache)


@pytest.mark.unit
def test_files(tmp_path):
    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    # initial write
    DiskCache.index = {}
    cache = DiskCache("cpu", offload_dir=str(offload_dir))
    tensor = torch.zeros(10)
    cache["weight"] = tensor

    files = os.listdir(offload_dir)
    assert len(DiskCache.index) == 1
    assert len(files) == 1
    with safe_open(offload_dir / files[0], framework="pt", device="cpu") as file:
        read_tensor = file.get_tensor("weight")
        assert_tensor_equal(read_tensor, tensor)

    # modify
    tensor = torch.ones(10)
    cache["weight"] = tensor

    files = os.listdir(offload_dir)
    assert len(DiskCache.index) == 1
    assert len(files) == 1
    with safe_open(offload_dir / files[0], framework="pt", device="cpu") as file:
        read_tensor = file.get_tensor("weight")
        assert_tensor_equal(read_tensor, tensor)

    # delete
    del cache["weight"]
    files = os.listdir(offload_dir)
    assert len(DiskCache.index) == 0
    assert len(files) == 0


@pytest.mark.unit
def test_clean_offload_dir(tmp_path):
    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    # Create multiple cache entries
    DiskCache.index = {}
    cache = DiskCache("cpu", offload_dir=str(offload_dir))
    cache["weight1"] = torch.zeros(10)
    cache["weight2"] = torch.ones(10)
    cache["weight3"] = torch.randn(10)

    # Verify files were created
    files = os.listdir(offload_dir)
    assert len(DiskCache.index) == 3
    assert len(files) == 3

    # Clean up all files
    files_cleaned = DiskCache.clean_offload_dir()
    assert files_cleaned == 3

    # Verify cleanup
    files = os.listdir(offload_dir)
    assert len(DiskCache.index) == 0
    assert len(files) == 0


@pytest.mark.unit
def test_clean_offload_dir_with_symlinks(tmp_path):
    offload_dir = tmp_path / "offload_dir"
    checkpoint_dir = tmp_path / "checkpoint"
    os.mkdir(offload_dir)
    os.mkdir(checkpoint_dir)

    # Create a checkpoint file
    checkpoint_file = checkpoint_dir / "model.safetensors"
    from safetensors.torch import save_file

    save_file({"weight": torch.zeros(10)}, str(checkpoint_file))

    # Create symlink via create_checkpoint_symlink
    DiskCache.index = {}
    offloaded = torch.empty(10, device="meta")
    weight_info = {
        "safetensors_file": str(checkpoint_file),
        "weight_name": "weight",
        "dtype": "float32",
    }
    DiskCache.create_checkpoint_symlink(offloaded, weight_info, str(offload_dir))

    # Verify symlink was created
    files = os.listdir(offload_dir)
    assert len(files) == 1
    assert os.path.islink(offload_dir / files[0])

    # Clean up should remove symlink but not target
    files_cleaned = DiskCache.clean_offload_dir()
    assert files_cleaned == 1

    # Verify symlink removed but target still exists
    files = os.listdir(offload_dir)
    assert len(files) == 0
    assert checkpoint_file.exists()
    assert len(DiskCache.index) == 0
