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
from compressed_tensors.offload import (
    align_module_device,
    align_modules,
    disable_offloading,
    disable_onloading,
    get_execution_device,
    get_offloaded_device,
    update_offload_parameter,
)
from compressed_tensors.offload.cache import CPUCache
from compressed_tensors.offload.module import offload_module
from tests.testing_utils import requires_gpu


ONLOAD_DEVICE = torch.device("cuda:0")
OFFLOAD_DEVICE = torch.device("cpu")


@pytest.fixture(scope="function")
def cache():
    return CPUCache(ONLOAD_DEVICE)


@pytest.fixture(scope="function")
def linear():
    return torch.nn.Linear(5, 5, bias=True, device=OFFLOAD_DEVICE)


@pytest.fixture(scope="function")
def offloaded_linear(linear, cache):
    offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)
    return linear


@pytest.mark.unit
@requires_gpu
def test_disable_offloading():
    cache1 = CPUCache(ONLOAD_DEVICE)
    cache2 = CPUCache(ONLOAD_DEVICE)

    cache1["weight"] = torch.tensor(0, device=OFFLOAD_DEVICE)
    cache2["weight"] = torch.tensor(1, device=OFFLOAD_DEVICE)

    with disable_offloading():
        assert cache1["weight"] in cache1.keep_onloaded_values.values()
        assert cache2["weight"] in cache2.keep_onloaded_values.values()


@pytest.mark.unit
@requires_gpu
def test_disable_onloading():
    cache1 = CPUCache(ONLOAD_DEVICE)
    cache2 = CPUCache(ONLOAD_DEVICE)

    cache1["weight"] = torch.tensor(0, device=OFFLOAD_DEVICE)
    cache2["weight"] = torch.tensor(1, device=OFFLOAD_DEVICE)

    with disable_onloading():
        assert cache1["weight"].device == OFFLOAD_DEVICE
        assert cache2["weight"].device == OFFLOAD_DEVICE


@pytest.mark.unit
@requires_gpu
@pytest.mark.parametrize("offload", (True, False))
def test_update_offload_parameter(linear: torch.nn.Linear, cache, offload):
    init_data = torch.tensor(0.0, device=OFFLOAD_DEVICE)
    linear.weight = torch.nn.Parameter(init_data, requires_grad=False)
    if offload:
        offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)

    assert linear.weight == 0

    update_offload_parameter(linear, "weight", 1)
    assert linear.weight == 1

    with disable_offloading():
        update_offload_parameter(linear, "weight", 2)
        assert linear.weight == 2
    assert linear.weight == 2

    with disable_onloading():
        update_offload_parameter(linear, "weight", 3)
        assert linear.weight == 3
    assert linear.weight == 3


@pytest.mark.unit
@requires_gpu
def test_get_execution_device(linear: torch.nn.Linear, cache):
    assert get_execution_device(linear) == OFFLOAD_DEVICE
    linear.to(ONLOAD_DEVICE)
    assert get_execution_device(linear) == ONLOAD_DEVICE

    linear.to(OFFLOAD_DEVICE)
    offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)
    assert get_execution_device(linear) == ONLOAD_DEVICE

    with disable_onloading():
        assert get_execution_device(linear) == ONLOAD_DEVICE

    with disable_offloading():
        assert get_execution_device(linear) == ONLOAD_DEVICE


@pytest.mark.unit
@requires_gpu
def test_get_offloaded_device(linear: torch.nn.Linear, cache):
    assert get_offloaded_device(linear) == OFFLOAD_DEVICE
    linear.to(ONLOAD_DEVICE)
    assert get_offloaded_device(linear) == ONLOAD_DEVICE

    linear.to(OFFLOAD_DEVICE)
    offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)
    assert get_offloaded_device(linear) == OFFLOAD_DEVICE

    with disable_onloading():
        assert get_offloaded_device(linear) == OFFLOAD_DEVICE

    with disable_offloading():
        assert get_offloaded_device(linear) == OFFLOAD_DEVICE


@pytest.mark.unit
@requires_gpu
def register_offload_module(linear: torch.nn.Linear, cache):
    sub1 = torch.nn.Linear(1, 1)
    register_offload_module(linear, "sub1", sub1)
    assert linear.sub1 is sub1

    offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)
    sub2 = torch.nn.Linear(1, 1)
    register_offload_module(linear, "sub2", sub2)
    assert linear.sub2 is sub2
    assert sub2.weight.device == ONLOAD_DEVICE


@pytest.mark.unit
@requires_gpu
def test_align_modules(offloaded_linear: torch.nn.Linear):
    linear = torch.nn.Linear(1, 1, device=ONLOAD_DEVICE)

    with align_modules((linear, offloaded_linear), OFFLOAD_DEVICE):
        assert linear.weight.device == OFFLOAD_DEVICE
        assert offloaded_linear.weight.device == OFFLOAD_DEVICE


@pytest.mark.unit
@requires_gpu
@pytest.mark.parametrize("offload", (True, False))
def test_align_module_device(linear: torch.nn.Linear, cache, offload):
    if offload:
        offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)
    else:
        linear.to(ONLOAD_DEVICE)

    with align_module_device(linear, OFFLOAD_DEVICE):
        assert linear.weight.device == OFFLOAD_DEVICE
