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

import gc
from weakref import ref

import pytest
import torch
from compressed_tensors.offload.cache.dist_cpu import DistributedCPUCache
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


ONLOAD_DEVICE = torch.device("cuda:0")
OFFLOAD_DEVICE = torch.device("cpu")


@pytest.fixture(scope="function")
def cache():
    return DistributedCPUCache(ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
@torchrun(target="tests.test_offload.cache.test_dist_cpu:test_onloading", world_size=2)
def test_onloading():
    cache = DistributedCPUCache(ONLOAD_DEVICE)
    tensor = torch.ones(10)
    onloaded = cache[tensor]

    assert type(onloaded) is type(tensor)
    assert torch.equal(onloaded.to(tensor.device), tensor)


@pytest.mark.unit
@requires_gpu
@torchrun(
    target="tests.test_offload.cache.test_dist_cpu:test_garbage_collect", world_size=2
)
def test_garbage_collect():
    cache = DistributedCPUCache(ONLOAD_DEVICE)
    tensor = torch.ones(10)
    onloaded = cache[tensor]

    onloaded_ref = ref(onloaded)
    del onloaded
    gc.collect()
    assert onloaded_ref() is None


@pytest.mark.unit
@requires_gpu
@torchrun(target="tests.test_offload.cache.test_dist_cpu:test_offload", world_size=2)
def test_offload():
    cache = DistributedCPUCache(ONLOAD_DEVICE)
    tensor = torch.ones(10, device=ONLOAD_DEVICE)
    offloaded = cache.offload(tensor)
    assert offloaded.device == OFFLOAD_DEVICE


@pytest.mark.unit
@requires_gpu
@torchrun(
    target="tests.test_offload.cache.test_dist_cpu:test_disable_offloading",
    world_size=2,
)
def test_disable_offloading():
    cache = DistributedCPUCache(ONLOAD_DEVICE)
    tensor = torch.ones(10)

    outside_onloaded = cache[tensor]
    outside_onloaded_ref = ref(outside_onloaded)
    assert outside_onloaded.device == ONLOAD_DEVICE

    with cache.disable_offloading():
        inside_onloaded = cache[tensor]
        inside_onloaded_ref = ref(inside_onloaded)
        assert inside_onloaded.device == ONLOAD_DEVICE

        del outside_onloaded
        del inside_onloaded
        gc.collect()

        assert outside_onloaded_ref() is not None
        assert inside_onloaded_ref() is not None

    assert outside_onloaded_ref() is None
    assert inside_onloaded_ref() is None


@pytest.mark.unit
@requires_gpu
@torchrun(
    target="tests.test_offload.cache.test_dist_cpu:test_disable_onloading", world_size=2
)
def test_disable_onloading():
    cache = DistributedCPUCache(ONLOAD_DEVICE)
    tensor = torch.ones(10)

    with cache.disable_onloading():
        onloaded = cache[tensor]
        assert onloaded is tensor

    assert onloaded is tensor


@pytest.mark.unit
@requires_gpu
@torchrun(target="tests.test_offload.cache.test_dist_cpu:test_delete", world_size=2)
def test_delete():
    cache = DistributedCPUCache(ONLOAD_DEVICE)
    tensor = torch.ones(10)
    onloaded = cache[tensor]
    onloaded_ref = ref(onloaded)

    with cache.disable_offloading():
        del cache[tensor]
        del onloaded
        gc.collect()

        assert onloaded_ref() is None

    assert onloaded_ref() is None
