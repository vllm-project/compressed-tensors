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
import inspect
from weakref import ref

import pytest
import torch
from compressed_tensors.offload import disable_offloading, disable_onloading
from compressed_tensors.offload.cache.cpu import CPUCache
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


@pytest.fixture(scope="function")
def input():
    return torch.zeros(6, device=OFFLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_onloading(linear: torch.nn.Linear, cache):
    weight = linear.weight
    bias = linear.bias

    offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)
    onloaded_weight = linear.weight
    onloaded_bias = linear.bias

    assert onloaded_weight.device == ONLOAD_DEVICE
    assert onloaded_bias.device == ONLOAD_DEVICE

    assert type(onloaded_weight) is type(weight)
    assert type(onloaded_bias) is type(bias)
    assert torch.equal(onloaded_weight.to(weight.device), weight)
    assert torch.equal(onloaded_bias.to(bias.device), bias)


@pytest.mark.unit
@requires_gpu
def test_garbage_collect(offloaded_linear: torch.nn.Linear):
    weight_ref = ref(offloaded_linear.weight)
    bias_ref = ref(offloaded_linear.bias)

    del offloaded_linear
    gc.collect()

    assert weight_ref() is None
    assert bias_ref() is None


@pytest.mark.unit
@requires_gpu
def test_disable_offloading(offloaded_linear: torch.nn.Linear):
    outside_onloaded = offloaded_linear.weight
    outside_onloaded_ref = ref(outside_onloaded)
    assert outside_onloaded.device == ONLOAD_DEVICE

    with disable_offloading():
        inside_onloaded = offloaded_linear.weight
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
def test_disable_onloading(linear: torch.nn.Linear, cache):
    offloaded_weight = linear.weight

    offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)

    with disable_onloading():
        weight = linear.weight
        assert weight is offloaded_weight

        # new parameter assignments are direct
        new_param = torch.nn.Parameter(torch.ones(5, device=ONLOAD_DEVICE))
        linear.new_param = new_param
        assert linear.new_param is new_param

    assert weight is offloaded_weight


@pytest.mark.unit
@requires_gpu
def test_delete(offloaded_linear: torch.nn.Linear):
    weight_ref = ref(offloaded_linear.weight)
    bias_ref = ref(offloaded_linear.bias)

    del offloaded_linear.weight
    del offloaded_linear.bias
    gc.collect()

    assert weight_ref() is None
    assert bias_ref() is None


@pytest.mark.unit
@requires_gpu
def test_forward_call(linear: torch.nn.Linear, cache):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.device == ONLOAD_DEVICE
        return torch.nn.functional.linear(input, linear.weight, linear.bias)

    linear.forward = forward.__get__(linear)

    offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)

    with torch.no_grad():
        input = torch.zeros(5, device=OFFLOAD_DEVICE)
        output = linear.forward(input)
        assert output.device == ONLOAD_DEVICE


@pytest.mark.parametrize("param_device", (ONLOAD_DEVICE, OFFLOAD_DEVICE))
@pytest.mark.parametrize("use_register_parameter", (True, False))
@pytest.mark.parametrize("requires_grad", (True, False))
def test_register_parameter(
    offloaded_linear: torch.nn.Linear,
    param_device,
    use_register_parameter,
    requires_grad,
):
    # register param
    data = torch.ones(5, device=param_device)
    param = torch.nn.Parameter(data, requires_grad=requires_grad)
    if use_register_parameter:
        offloaded_linear.register_parameter("param_name", param)
    else:
        offloaded_linear.param_name = param

    # new param is correctly onloaded
    assert offloaded_linear.param_name.device == ONLOAD_DEVICE
    assert torch.equal(offloaded_linear.param_name.to(param_device), param)


@pytest.mark.parametrize("param_device", (ONLOAD_DEVICE, OFFLOAD_DEVICE))
@pytest.mark.parametrize("use_register_parameter", (True, False))
@pytest.mark.parametrize("requires_grad", (True, False))
def test_register_parameter_invalidates(
    offloaded_linear: torch.nn.Linear,
    param_device,
    use_register_parameter,
    requires_grad,
):
    with disable_offloading():
        # original weight is kept onloaded
        onloaded_weight = offloaded_linear.weight
        assert onloaded_weight in set(CPUCache.keep_onloaded_values.values())

        # add new param
        data = torch.ones(5, device=param_device)
        param = torch.nn.Parameter(data, requires_grad=requires_grad)
        if use_register_parameter:
            offloaded_linear.register_parameter("weight", param)
        else:
            offloaded_linear.weight = param

        # new param is correct
        assert offloaded_linear.weight.device == ONLOAD_DEVICE
        assert torch.equal(offloaded_linear.weight.to(param_device), param)

        # original weight is invalidated
        assert onloaded_weight not in set(CPUCache.keep_onloaded_values.values())


def test_forward_signature(linear: torch.nn.Linear, cache):
    original_signature = inspect.signature(linear.forward)

    offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)
    assert inspect.signature(linear.forward) == original_signature
