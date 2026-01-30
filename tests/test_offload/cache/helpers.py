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

import torch
from compressed_tensors.offload import OffloadCache


def _test_onloading(offload_device: str, onload_device: str):
    cache = OffloadCache.cls_from_device(offload_device)(onload_device)
    tensor = torch.ones(10)
    cache["weight"] = tensor
    onloaded = cache["weight"]

    assert type(onloaded) is type(tensor)
    assert torch.equal(onloaded.to(tensor.device), tensor)


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
    assert offloaded.device == offload_device
    assert torch.equal(offloaded.to(onload_device), tensor)


def _test_onload(offload_device: str, onload_device: str):
    cache = OffloadCache.cls_from_device(offload_device)(onload_device)
    tensor = torch.ones(10, device=onload_device)
    onloaded = cache.onload(cache.offload(tensor))
    assert onloaded.device == onload_device
    assert torch.equal(onloaded, onloaded)


def _test_disable_offloading(offload_device: str, onload_device: str):
    cache = OffloadCache.cls_from_device(offload_device)(onload_device)
    cache["weight"] = torch.ones(10)

    outside_onloaded = cache["weight"]
    outside_onloaded_ref = ref(outside_onloaded)
    assert outside_onloaded.device == onload_device

    with cache.disable_offloading():
        inside_onloaded = cache["weight"]
        inside_onloaded_ref = ref(inside_onloaded)
        assert inside_onloaded.device == onload_device

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
