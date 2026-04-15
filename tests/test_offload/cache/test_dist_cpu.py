# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload import disable_onloading
from compressed_tensors.offload.cache.dist_cpu import DistributedCPUCache
from loguru import logger as loguru_logger
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


@pytest.fixture()
def onload_device():
    return torch.device("cuda")


@pytest.fixture()
def offload_device():
    return torch.device("cpu")


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
def test_distributed_offload(onload_device):
    cache = DistributedCPUCache(onload_device)
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
def test_shared_cpu_offload(onload_device):
    cache = DistributedCPUCache(onload_device)
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
def test_distributed_async_update(onload_device):
    """
    Test that different ranks can update different tensors asynchronously,
    and that values are correct after a barrier.
    """
    cache = DistributedCPUCache(onload_device)

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
        assert torch.allclose(offloaded_0.cpu(), torch.ones(10) * 1.0)
        assert torch.allclose(offloaded_1.cpu(), torch.ones(10) * 2.0)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_offload_logs_memory_hint(onload_device):
    cache = DistributedCPUCache(onload_device)

    original_share_memory = torch.Tensor.share_memory_

    def raise_memory_error(*args, **kwargs):
        raise RuntimeError("mmap failed: Cannot allocate memory")

    torch.Tensor.share_memory_ = raise_memory_error

    warnings = []
    handler_id = loguru_logger.add(
        lambda msg: warnings.append(msg.record["message"]), level="WARNING"
    )

    try:
        # Only Rank 0 calls offload(), which throws an error *before* the
        # dist.broadcast barrier. This cleanly avoids the hang since Rank 1
        # safely exits without waiting.
        if dist.get_rank() == 0:
            with pytest.raises(RuntimeError, match="Cannot allocate memory"):
                cache.offload(torch.zeros(1, device=onload_device))
    finally:
        torch.Tensor.share_memory_ = original_share_memory
        loguru_logger.remove(handler_id)

    if dist.get_rank() == 0:
        assert any(
            "CPU offloading ran out of host RAM or mmap descriptors." in w
            for w in warnings
        )
