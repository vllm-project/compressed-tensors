# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload import disable_onloading
from compressed_tensors.offload.cache.dist_cpu import DistributedCPUCache
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
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


ONLOAD_DEVICE = torch.device("cuda")
OFFLOAD_DEVICE = torch.device("cpu")


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
def test_tensor_subclass():
    _test_tensor_subclass(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_offload():
    cache = DistributedCPUCache(ONLOAD_DEVICE)
    tensor = torch.zeros((5, 2))
    cache["tensor"] = tensor

    # check tensor construction
    assert torch.equal(cache["tensor"].cpu(), tensor)
    with disable_onloading():
        assert torch.equal(cache["tensor"].cpu(), tensor)

    # update tensor
    tensor = torch.ones((5, 2))
    cache["tensor"] = tensor

    # check tensor construction
    assert torch.equal(cache["tensor"].cpu(), tensor)
    with disable_onloading():
        assert torch.equal(cache["tensor"].cpu(), tensor)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_shared_cpu_offload():
    cache = DistributedCPUCache(ONLOAD_DEVICE)
    tensor = torch.zeros((5, 2))
    cache["tensor"] = tensor

    # modify the offloaded cpu tensor directly
    tensor = torch.ones((5, 2))
    if dist.get_rank() == 0:
        with disable_onloading():
            cache["tensor"].copy_(tensor)

    dist.barrier()

    # check that the value is affected on all ranks
    assert torch.equal(cache["tensor"].cpu(), tensor)
    with disable_onloading():
        assert torch.equal(cache["tensor"].cpu(), tensor)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_async_update():
    """
    Test that different ranks can update different tensors asynchronously,
    and that values are correct after a barrier.
    """
    cache = DistributedCPUCache(ONLOAD_DEVICE)

    # Initialize two tensors in the cache
    cache["tensor_0"] = torch.zeros(10, device=ONLOAD_DEVICE)
    cache["tensor_1"] = torch.zeros(10, device=ONLOAD_DEVICE)

    # Each rank updates a different tensor
    rank = dist.get_rank()
    if rank == 0:
        # Rank 0 updates tensor_0
        cache[f"tensor_{rank}"] = torch.ones(10, device=ONLOAD_DEVICE) * 1.0
    elif rank == 1:
        # Rank 1 updates tensor_1
        cache[f"tensor_{rank}"] = torch.ones(10, device=ONLOAD_DEVICE) * 2.0

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
        assert torch.allclose(offloaded_0.cpu(), torch.ones(10) * 1.0)
        assert torch.allclose(offloaded_1.cpu(), torch.ones(10) * 2.0)
