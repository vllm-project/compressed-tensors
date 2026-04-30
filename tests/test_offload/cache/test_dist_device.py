# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from weakref import ref

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload import disable_onloading
from compressed_tensors.offload.cache.dist_device import DistributedDeviceCache
from tests.test_offload.cache.helpers import (
    _test_delete,
    _test_disable_onloading,
    _test_offload,
    _test_onload,
    _test_onloading,
    _test_shared_attributes,
    _test_tensor_subclass,
)
from tests.test_offload.conftest import assert_device_equal, torchrun
from tests.testing_utils import requires_gpu


# Note that tests only require at least 1 gpu
# b/c different ranks can share the same gpu


@pytest.fixture()
def onload_device():
    return torch.accelerator.current_accelerator()


@pytest.fixture()
def offload_device():
    return torch.accelerator.current_accelerator()


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_delete(offload_device, onload_device, offload_cache):
    _test_delete(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_disable_offloading(onload_device):
    # unlike other device caches, the onload is not garbage collected
    cache = DistributedDeviceCache(onload_device)
    cache["weight"] = torch.ones(10)

    outside_onloaded = cache["weight"]
    outside_onloaded_ref = ref(outside_onloaded)
    assert_device_equal(outside_onloaded.device, onload_device)

    with cache.disable_offloading():
        inside_onloaded = cache["weight"]
        inside_onloaded_ref = ref(inside_onloaded)
        assert_device_equal(inside_onloaded.device, onload_device)

        del outside_onloaded
        del inside_onloaded
        gc.collect()

        assert outside_onloaded_ref() is not None  # changed
        assert inside_onloaded_ref() is not None

    assert outside_onloaded_ref() is not None  # changed
    assert inside_onloaded_ref() is not None  # changed


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_disable_onloading(offload_device, onload_device, offload_cache):
    _test_disable_onloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_garbage_collect(onload_device):
    # unlike other device caches, the onload is not garbage collected
    cache = DistributedDeviceCache(onload_device)
    cache["weight"] = torch.ones(10)
    onloaded = cache["weight"]

    onloaded_ref = ref(onloaded)
    del onloaded
    gc.collect()
    assert onloaded_ref() is not None  # changed


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
    cache = DistributedDeviceCache(onload_device)
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
def test_distributed_offload_fp8(onload_device):
    """FP8 tensors should broadcast successfully and preserve their dtype"""
    float8_dtypes = [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    ]
    for dtype in float8_dtypes:
        cache = DistributedDeviceCache(onload_device)
        tensor = torch.zeros((5, 2), dtype=dtype)
        cache["tensor"] = tensor

        result = cache["tensor"].cpu()
        assert result.shape == tensor.shape
        assert result.dtype == dtype


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replicated_device_offload(onload_device):
    cache = DistributedDeviceCache(onload_device)
    tensor = torch.empty((5, 2))
    cache["tensor"] = tensor

    # modify the offloaded cpu tensor directly
    tensor = torch.full((5, 2), dist.get_rank())
    cache["tensor"].copy_(tensor)
    dist.barrier()

    # check that the value is affected on all ranks
    assert torch.equal(cache["tensor"].cpu(), tensor)
    with disable_onloading():
        assert torch.equal(cache["tensor"].cpu(), tensor)
