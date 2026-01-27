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
from compressed_tensors.offload.cache.cpu import CPUCache
from tests.testing_utils import requires_gpu


ONLOAD_DEVICE = torch.device("cuda:0")
OFFLOAD_DEVICE = torch.device("cpu")


@pytest.fixture(scope="function")
def cache():
    return CPUCache(ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_onloading(cache: CPUCache):
    tensor = torch.ones(10)
    cache["weight"] = tensor
    onloaded = cache["weight"]

    assert type(onloaded) is type(tensor)
    assert torch.equal(onloaded.to(tensor.device), tensor)


@pytest.mark.unit
@requires_gpu
def test_garbage_collect(cache: CPUCache):
    cache["weight"] = torch.ones(10)
    onloaded = cache["weight"]

    onloaded_ref = ref(onloaded)
    del onloaded
    gc.collect()
    assert onloaded_ref() is None


@pytest.mark.unit
@requires_gpu
def test_offload(cache: CPUCache):
    tensor = torch.ones(10, device=ONLOAD_DEVICE)
    offloaded = cache.offload(tensor)
    assert offloaded.device == OFFLOAD_DEVICE
    assert torch.equal(offloaded.to(ONLOAD_DEVICE), tensor)


@pytest.mark.unit
@requires_gpu
def test_onload(cache: CPUCache):
    tensor = torch.ones(10, device=ONLOAD_DEVICE)
    onloaded = cache.onload(cache.offload(tensor))
    assert onloaded.device == ONLOAD_DEVICE
    assert torch.equal(onloaded, onloaded)


@pytest.mark.unit
@requires_gpu
def test_disable_offloading(cache: CPUCache):
    cache["weight"] = torch.ones(10)

    outside_onloaded = cache["weight"]
    outside_onloaded_ref = ref(outside_onloaded)
    assert outside_onloaded.device == ONLOAD_DEVICE

    with cache.disable_offloading():
        inside_onloaded = cache["weight"]
        inside_onloaded_ref = ref(inside_onloaded)
        assert inside_onloaded.device == ONLOAD_DEVICE

        del outside_onloaded
        del inside_onloaded
        gc.collect()

        assert outside_onloaded_ref() is None
        assert inside_onloaded_ref() is not None

    assert outside_onloaded_ref() is None
    assert inside_onloaded_ref() is None


@pytest.mark.unit
@requires_gpu
def test_disable_onloading(cache: CPUCache):
    tensor = torch.ones(10)
    cache.offloaded_values["weight"] = tensor

    with cache.disable_onloading():
        onloaded = cache["weight"]
        assert onloaded is tensor

    assert onloaded is tensor


@pytest.mark.unit
@requires_gpu
def test_delete(cache: CPUCache):
    cache["weight"] = torch.ones(10)
    onloaded = cache["weight"]
    onloaded_ref = ref(onloaded)

    with cache.disable_offloading():
        del cache["weight"]
        del onloaded
        gc.collect()

        assert onloaded_ref() is None

    assert onloaded_ref() is None


@pytest.mark.unit
@requires_gpu
def test_shared_attributes(cache: CPUCache):
    assert cache.offload_device is CPUCache.offload_device
    assert cache.offloading_disabled is CPUCache.offloading_disabled
    assert cache.onloading_disabled is CPUCache.onloading_disabled
    assert cache.keep_onloaded_values is CPUCache.keep_onloaded_values

    assert not hasattr(CPUCache, "onload_device")
    assert not hasattr(CPUCache, "offloaded_values")
