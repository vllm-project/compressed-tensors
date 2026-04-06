# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


@pytest.fixture()
def onload_device():
    if (torch.mps.is_available()):
        return torch.device("mps")

    return torch.device("cuda")


@pytest.fixture()
def offload_device():
    if (torch.mps.is_available()):
        return torch.device("mps")

    return torch.device("cuda")


@pytest.mark.unit
@requires_gpu
def test_delete(offload_device, onload_device, offload_cache):
    _test_delete(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_disable_offloading(onload_device):
    # unlike other device caches, the onload is not garbage collected
    cache = DeviceCache(onload_device)
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
@requires_gpu
def test_disable_onloading(offload_device, onload_device, offload_cache):
    _test_disable_onloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_garbage_collect(onload_device):
    # unlike other device caches, the onload is not garbage collected
    cache = DeviceCache(onload_device)
    cache["weight"] = torch.ones(10)
    onloaded = cache["weight"]

    onloaded_ref = ref(onloaded)
    del onloaded
    gc.collect()
    assert onloaded_ref() is not None  # changed


@pytest.mark.unit
@requires_gpu
def test_offload(offload_device, onload_device, offload_cache):
    _test_offload(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
@requires_gpu
def test_onload(offload_device, onload_device, offload_cache):
    _test_onload(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_onloading(offload_device, onload_device, offload_cache):
    _test_onloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_shared_attributes(offload_device, onload_device, offload_cache):
    _test_shared_attributes(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_tensor_subclass(offload_device, onload_device, offload_cache):
    _test_tensor_subclass(offload_device, onload_device, offload_cache)
