# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from weakref import ref

import torch
from compressed_tensors.offload import OffloadCache
from tests.test_offload.conftest import assert_device_equal


def _test_onloading(offload_device: str, onload_device: str):
    cache = OffloadCache.cls_from_device(offload_device)(onload_device)
    tensor = torch.ones(10)
    cache["weight"] = tensor
    onloaded = cache["weight"]

    assert type(onloaded) is type(tensor)
    assert torch.equal(onloaded, tensor.to(onloaded))


def _test_garbage_collect(offload_device: str, onload_device: str):
    cache = OffloadCache.cls_from_device(offload_device)(onload_device)
    cache["weight"] = torch.ones(10)
    onloaded = cache["weight"]

    onloaded_ref = ref(onloaded)
    del onloaded
    gc.collect()
    assert onloaded_ref() is None


def _test_offload(offload_device: str, onload_device: str):
    cache = OffloadCache.cls_from_device(offload_device)(onload_device)
    tensor = torch.ones(10, device=onload_device)
    offloaded = cache.offload(tensor)
    assert_device_equal(offloaded.device, offload_device)
    assert torch.equal(offloaded, tensor.to(offloaded))


def _test_onload(offload_device: str, onload_device: str):
    cache = OffloadCache.cls_from_device(offload_device)(onload_device)
    tensor = torch.ones(10, device=onload_device)
    onloaded = cache.onload(cache.offload(tensor))
    assert_device_equal(onloaded.device, onload_device)
    assert torch.equal(onloaded, tensor.to(onloaded))


def _test_disable_offloading(offload_device: str, onload_device: str):
    cache = OffloadCache.cls_from_device(offload_device)(onload_device)
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

        assert outside_onloaded_ref() is None
        assert inside_onloaded_ref() is not None

    assert outside_onloaded_ref() is None
    assert inside_onloaded_ref() is None


def _test_disable_onloading(offload_device: str, onload_device: str):
    cache = OffloadCache.cls_from_device(offload_device)(onload_device)
    tensor = torch.ones(10)
    cache.offloaded_values["weight"] = tensor

    with cache.disable_onloading():
        onloaded = cache["weight"]
        assert onloaded is tensor

    assert onloaded is tensor


def _test_delete(offload_device: str, onload_device: str):
    cache = OffloadCache.cls_from_device(offload_device)(onload_device)
    cache["weight"] = torch.ones(10)
    onloaded = cache["weight"]
    onloaded_ref = ref(onloaded)

    with cache.disable_offloading():
        del cache["weight"]
        del onloaded
        gc.collect()

        assert onloaded_ref() is None

    assert onloaded_ref() is None


def _test_shared_attributes(offload_device: str, onload_device: str):
    cache = OffloadCache.cls_from_device(offload_device)(onload_device)
    assert cache.offload_device is cache.__class__.offload_device
    assert cache.offloading_disabled is cache.__class__.offloading_disabled
    assert cache.onloading_disabled is cache.__class__.onloading_disabled
    assert cache.keep_onloaded_values is cache.__class__.keep_onloaded_values

    assert not hasattr(cache.__class__, "onload_device")
    assert not hasattr(cache.__class__, "offloaded_values")
