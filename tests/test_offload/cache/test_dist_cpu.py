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
