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


ONLOAD_DEVICE = torch.device("cuda")
OFFLOAD_DEVICE = torch.device("cuda")

# Note that tests only require at least 1 gpu
# b/c different ranks can share the same gpu


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_delete():
    _test_delete(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_disable_offloading():
    # unlike other device caches, the onload is not garbage collected
    cache = DistributedDeviceCache(ONLOAD_DEVICE)
    cache["weight"] = torch.ones(10)

    outside_onloaded = cache["weight"]
    outside_onloaded_ref = ref(outside_onloaded)
    assert_device_equal(outside_onloaded.device, ONLOAD_DEVICE)

    with cache.disable_offloading():
        inside_onloaded = cache["weight"]
        inside_onloaded_ref = ref(inside_onloaded)
        assert_device_equal(inside_onloaded.device, ONLOAD_DEVICE)

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
def test_disable_onloading():
    _test_disable_onloading(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_garbage_collect():
    # unlike other device caches, the onload is not garbage collected
    cache = DistributedDeviceCache(ONLOAD_DEVICE)
    cache["weight"] = torch.ones(10)
    onloaded = cache["weight"]

    onloaded_ref = ref(onloaded)
    del onloaded
    gc.collect()
    assert onloaded_ref() is not None  # changed


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
@requires_gpu
def test_tensor_subclass():
    _test_tensor_subclass(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_distributed_offload():
    cache = DistributedDeviceCache(ONLOAD_DEVICE)
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
def test_replicated_device_offload():
    cache = DistributedDeviceCache(ONLOAD_DEVICE)
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
