# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from weakref import ref

import torch
from tests.test_offload.conftest import assert_device_equal, assert_tensor_equal


def _test_onloading(offload_device, onload_device, offload_cache):
    tensor = torch.ones(10)
    offload_cache["weight"] = tensor
    onloaded = offload_cache["weight"]

    assert type(onloaded) is type(tensor)
    assert_tensor_equal(onloaded, tensor, onload_device)


def _test_garbage_collect(offload_device, onload_device, offload_cache):
    offload_cache["weight"] = torch.ones(10)
    onloaded = offload_cache["weight"]

    onloaded_ref = ref(onloaded)
    del onloaded
    gc.collect()
    assert onloaded_ref() is None


def _test_offload(offload_device, onload_device, offload_cache):
    tensor = torch.ones(10, device=onload_device)
    offloaded = offload_cache.offload(tensor)
    assert_device_equal(offloaded.device, offload_device)
    assert_tensor_equal(offloaded, tensor, offload_device)


def _test_onload(offload_device, onload_device, offload_cache):
    tensor = torch.ones(10, device=onload_device)
    onloaded = offload_cache.onload(offload_cache.offload(tensor))
    assert_device_equal(onloaded.device, onload_device)
    assert_tensor_equal(onloaded, tensor, onload_device)


def _test_disable_offloading(offload_device, onload_device, offload_cache):
    offload_cache["weight"] = torch.ones(10)

    outside_onloaded = offload_cache["weight"]
    outside_onloaded_ref = ref(outside_onloaded)
    assert_device_equal(outside_onloaded.device, onload_device)

    with offload_cache.disable_offloading():
        inside_onloaded = offload_cache["weight"]
        inside_onloaded_ref = ref(inside_onloaded)
        assert_device_equal(inside_onloaded.device, onload_device)

        del outside_onloaded
        del inside_onloaded
        gc.collect()

        assert outside_onloaded_ref() is None
        assert inside_onloaded_ref() is not None

    assert outside_onloaded_ref() is None
    assert inside_onloaded_ref() is None


def _test_disable_onloading(offload_device, onload_device, offload_cache):
    tensor = torch.ones(10)
    offload_cache.offloaded_values["weight"] = tensor

    with offload_cache.disable_onloading():
        onloaded = offload_cache["weight"]
        assert onloaded is tensor

    assert onloaded is tensor


def _test_delete(offload_device, onload_device, offload_cache):
    offload_cache["weight"] = torch.ones(10)
    onloaded = offload_cache["weight"]
    onloaded_ref = ref(onloaded)

    with offload_cache.disable_offloading():
        del offload_cache["weight"]
        del onloaded
        gc.collect()

        assert onloaded_ref() is None

    assert onloaded_ref() is None


def _test_shared_attributes(offload_device, onload_device, offload_cache):
    assert (
        offload_cache.offloading_disabled is offload_cache.__class__.offloading_disabled
    )
    assert (
        offload_cache.onloading_disabled is offload_cache.__class__.onloading_disabled
    )
    assert (
        offload_cache.keep_onloaded_values
        is offload_cache.__class__.keep_onloaded_values
    )

    assert not hasattr(offload_cache.__class__, "onload_device")
    assert not hasattr(offload_cache.__class__, "offloaded_values")


def _test_tensor_subclass(offload_device, onload_device, offload_cache):
    tensor = torch.ones(10)
    param = torch.nn.Parameter(torch.ones(10), requires_grad=False)
    buffer = torch.nn.Buffer(torch.ones(10))

    offload_cache["tensor"] = tensor
    offload_cache["param"] = param
    offload_cache["buffer"] = buffer

    assert_tensor_equal(offload_cache["tensor"], tensor, onload_device)
    assert_tensor_equal(offload_cache["param"], param, onload_device)
    assert_tensor_equal(offload_cache["buffer"], buffer, onload_device)

    with offload_cache.disable_onloading():
        assert_tensor_equal(offload_cache["tensor"], tensor, offload_device)
        assert_tensor_equal(offload_cache["param"], param, offload_device)
        assert_tensor_equal(offload_cache["buffer"], buffer, offload_device)
