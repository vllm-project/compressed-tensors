# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload import disable_onloading
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.cache.dist_disk import DistributedDiskCache
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
)
from tests.test_offload.conftest import assert_tensor_equal, torchrun
from tests.testing_utils import requires_gpu


ONLOAD_DEVICE = torch.device("cuda")
OFFLOAD_DEVICE = "disk"


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_delete():
    _test_delete(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_disable_offloading():
    _test_disable_offloading(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_disable_onloading():
    _test_disable_onloading(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_garbage_collect():
    _test_garbage_collect(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_offload():
    _test_offload(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_onload():
    _test_onload(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_onloading():
    _test_onloading(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_shared_attributes():
    _test_shared_attributes(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_offload(tmp_path):
    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    cache = DistributedDiskCache(ONLOAD_DEVICE, offload_dir=str(offload_dir))
    tensor = torch.zeros((5, 2))
    cache["tensor"] = tensor

    # check tensor construction
    assert torch.equal(cache["tensor"], tensor.to(ONLOAD_DEVICE))
    with disable_onloading():
        assert_tensor_equal(cache["tensor"], tensor.to("meta"))

    # update tensor
    tensor = torch.ones((5, 2))
    cache["tensor"] = tensor

    # check tensor construction
    assert torch.equal(cache["tensor"], tensor.to(ONLOAD_DEVICE))
    with disable_onloading():
        assert_tensor_equal(cache["tensor"], tensor.to("meta"))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_files(tmp_path):
    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    # initial write, broadcasted to all ranks
    DiskCache.index = {}
    cache = DistributedDiskCache("cpu", offload_dir=str(offload_dir))
    tensor = torch.zeros(10)
    cache["weight"] = tensor

    assert len(DiskCache.index) == 1
    if dist.get_rank() == 0:  # only rank0 bc `tmp_path` is not shared between ranks
        files = os.listdir(offload_dir)
        assert len(files) == 1
        with safe_open(offload_dir / files[0], framework="pt", device="cpu") as file:
            read_tensor = file.get_tensor("weight")
            assert_tensor_equal(read_tensor, tensor)

    # modify on one rank
    tensor = torch.ones(10)
    if dist.get_rank() == 0:
        cache["weight"] = tensor

    assert len(DiskCache.index) == 1
    if dist.get_rank() == 0:  # only rank0 bc `tmp_path` is not shared between ranks
        files = os.listdir(offload_dir)
        assert len(files) == 1
        with safe_open(offload_dir / files[0], framework="pt", device="cpu") as file:
            read_tensor = file.get_tensor("weight")
            assert_tensor_equal(read_tensor, tensor)

    # delete
    del cache["weight"]
    assert len(DiskCache.index) == 0
    if dist.get_rank() == 0:  # only rank0 bc `tmp_path` is not shared between ranks
        files = os.listdir(offload_dir)
        assert len(files) == 0
