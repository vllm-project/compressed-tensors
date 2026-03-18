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
    _test_tensor_subclass,
)
from tests.test_offload.conftest import assert_tensor_equal, torchrun
from tests.testing_utils import requires_gpu


@pytest.fixture()
def onload_device():
    return torch.device("cuda")


@pytest.fixture()
def offload_device():
    return "disk"


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_delete(offload_device, onload_device, offload_cache):
    _test_delete(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_disable_offloading(offload_device, onload_device, offload_cache):
    _test_disable_offloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_disable_onloading(offload_device, onload_device, offload_cache):
    _test_disable_onloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_garbage_collect(offload_device, onload_device, offload_cache):
    _test_garbage_collect(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_offload(offload_device, onload_device, offload_cache):
    _test_offload(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_onload(offload_device, onload_device, offload_cache):
    _test_onload(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_onloading(offload_device, onload_device, offload_cache):
    _test_onloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_shared_attributes(offload_device, onload_device, offload_cache):
    _test_shared_attributes(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_tensor_subclass(offload_device, onload_device, offload_cache):
    _test_tensor_subclass(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_offload(onload_device, tmp_path):
    # Broadcast directory path from rank 0 to all ranks
    if dist.get_rank() == 0:
        offload_dir = tmp_path / "offload_dir"
        os.mkdir(offload_dir)
        broadcast_obj = [str(offload_dir)]
    else:
        broadcast_obj = [None]

    dist.broadcast_object_list(broadcast_obj, src=0)
    offload_dir = broadcast_obj[0]

    # Ensure directory creation completes before other ranks proceed
    dist.barrier()

    cache = DistributedDiskCache(onload_device, offload_dir=offload_dir)
    tensor = torch.zeros((5, 2))
    cache["tensor"] = tensor

    # check tensor construction
    assert torch.equal(cache["tensor"], tensor.to(onload_device))
    with disable_onloading():
        assert_tensor_equal(cache["tensor"], tensor.to("meta"))

    # update tensor
    tensor = torch.ones((5, 2))
    cache["tensor"] = tensor

    # check tensor construction
    assert torch.equal(cache["tensor"], tensor.to(onload_device))
    with disable_onloading():
        assert_tensor_equal(cache["tensor"], tensor.to("meta"))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_files(tmp_path):
    # Broadcast directory path from rank 0 to all ranks
    if dist.get_rank() == 0:
        offload_dir = tmp_path / "offload_dir"
        os.mkdir(offload_dir)
        broadcast_obj = [str(offload_dir)]
    else:
        broadcast_obj = [None]

    dist.broadcast_object_list(broadcast_obj, src=0)
    offload_dir = broadcast_obj[0]

    # Ensure directory creation completes before other ranks proceed
    dist.barrier()

    # initial write, broadcasted to all ranks
    DiskCache.index = {}
    cache = DistributedDiskCache("cpu", offload_dir=offload_dir)
    tensor = torch.zeros(10)
    cache["weight"] = tensor

    assert len(DiskCache.index) == 1
    if dist.get_rank() == 0:  # only rank0 bc `tmp_path` is not shared between ranks
        files = os.listdir(offload_dir)
        assert len(files) == 1
        with safe_open(
            os.path.join(offload_dir, files[0]), framework="pt", device="cpu"
        ) as file:
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
        with safe_open(
            os.path.join(offload_dir, files[0]), framework="pt", device="cpu"
        ) as file:
            read_tensor = file.get_tensor("weight")
            assert_tensor_equal(read_tensor, tensor)

    # delete
    del cache["weight"]
    assert len(DiskCache.index) == 0
    if dist.get_rank() == 0:  # only rank0 bc `tmp_path` is not shared between ranks
        files = os.listdir(offload_dir)
        assert len(files) == 0


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_async_update(tmp_path):
    """
    Test that different ranks can update different tensors asynchronously,
    and that values are correct after a barrier.
    """
    offload_dir = tmp_path / "offload_dir"
    if dist.get_rank() == 0:
        os.mkdir(offload_dir)

    # Ensure directory creation completes before other ranks proceed
    dist.barrier()

    onload_device = torch.device("cuda")
    cache = DistributedDiskCache(onload_device, offload_dir=str(offload_dir))

    # Initialize two tensors in the cache
    cache["tensor_0"] = torch.zeros(10, device=onload_device)
    cache["tensor_1"] = torch.zeros(10, device=onload_device)

    # Each rank updates a different tensor
    rank = dist.get_rank()
    if rank == 0:
        # Rank 0 updates tensor_0
        cache[f"tensor_{rank}"] = torch.ones(10, device=onload_device) * 1.0
    elif rank == 1:
        # Rank 1 updates tensor_1
        cache[f"tensor_{rank}"] = torch.ones(10, device=onload_device) * 2.0

    # Synchronize to ensure all updates are complete
    dist.barrier()

    # Verify that both tensors have the correct values on all ranks
    tensor_0 = cache["tensor_0"]
    tensor_1 = cache["tensor_1"]

    assert torch.allclose(tensor_0.cpu(), torch.ones(10) * 1.0)
    assert torch.allclose(tensor_1.cpu(), torch.ones(10) * 2.0)

    # Verify offloaded values are also correct
    with disable_onloading():
        offloaded_0 = cache["tensor_0"]
        offloaded_1 = cache["tensor_1"]
        assert_tensor_equal(offloaded_0, torch.ones(10) * 1.0, "disk")
        assert_tensor_equal(offloaded_1, torch.ones(10) * 2.0, "disk")
