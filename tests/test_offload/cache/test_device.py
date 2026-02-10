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
from compressed_tensors.offload.cache.device import DeviceCache
from tests.test_offload.cache.helpers import (
    _test_delete,
    _test_disable_onloading,
    _test_offload,
    _test_onload,
    _test_onloading,
    _test_shared_attributes,
    _test_tensor_subclass,
)
from tests.test_offload.conftest import assert_device_equal
from tests.testing_utils import requires_gpu


ONLOAD_DEVICE = torch.device("cuda")
OFFLOAD_DEVICE = torch.device("cuda")


@pytest.mark.unit
@requires_gpu
def test_delete():
    _test_delete(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_disable_offloading():
    # unlike other device caches, the onload is not garbage collected
    cache = DeviceCache(ONLOAD_DEVICE)
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
@requires_gpu
def test_disable_onloading():
    _test_disable_onloading(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_garbage_collect():
    # unlike other device caches, the onload is not garbage collected
    cache = DeviceCache(ONLOAD_DEVICE)
    cache["weight"] = torch.ones(10)
    onloaded = cache["weight"]

    onloaded_ref = ref(onloaded)
    del onloaded
    gc.collect()
    assert onloaded_ref() is not None  # changed


@pytest.mark.unit
@requires_gpu
def test_offload():
    _test_offload(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
@requires_gpu
def test_onload():
    _test_onload(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_onloading():
    _test_onloading(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_shared_attributes():
    _test_shared_attributes(OFFLOAD_DEVICE, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_tensor_subclass():
    _test_tensor_subclass(OFFLOAD_DEVICE, ONLOAD_DEVICE)
