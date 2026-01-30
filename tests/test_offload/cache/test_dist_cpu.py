# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
)
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


ONLOAD_DEVICE = torch.device("cuda:0")
OFFLOAD_DEVICE = torch.device("cpu")

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
def test_distributed_offload():
    cache = DistributedCPUCache(ONLOAD_DEVICE)
    tensor = torch.zeros(5)
    cache["tensor"] = tensor

    # check tensor construction
    assert torch.equal(cache["tensor"].cpu(), tensor)
    with disable_onloading():
        assert torch.equal(cache["tensor"].cpu(), tensor)

    # update tensor
    tensor = torch.ones(5)
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
    tensor = torch.zeros(5)
    cache["tensor"] = tensor

    # modify the offloaded cpu tensor directly
    tensor = torch.ones(5)
    if dist.get_rank() == 0:
        with disable_onloading():
            cache["tensor"].copy_(tensor)

    dist.barrier()

    # check that the value is affected on all ranks
    assert torch.equal(cache["tensor"].cpu(), tensor)
    with disable_onloading():
        assert torch.equal(cache["tensor"].cpu(), tensor)
