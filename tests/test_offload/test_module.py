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
from compressed_tensors.offload.cache.cpu import CPUCache
from compressed_tensors.offload.module import offload_module
from tests.testing_utils import requires_gpu


ONLOAD_DEVICE = torch.device("cuda:0")
OFFLOAD_DEVICE = torch.device("cpu")


@pytest.fixture(scope="function")
def cache():
    return CPUCache(ONLOAD_DEVICE)


@pytest.fixture(scope="function")
def linear(cache):
    linear = torch.nn.Linear(5, 5, bias=True, device=OFFLOAD_DEVICE)
    return OffloadedModule.from_module(linear, cache)


@pytest.fixture(scope="function")
def input():
    return torch.zeros(6, device=OFFLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_onloading(linear: torch.nn.Linear | OffloadedModule):
    weight = linear._module.weight
    bias = linear._module.bias

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
def test_garbage_collect(linear: torch.nn.Linear | OffloadedModule):
    weight_ref = ref(linear.weight)
    bias_ref = ref(linear.bias)

    del linear
    gc.collect()

    assert weight_ref() is None
    assert bias_ref() is None


@pytest.mark.unit
@requires_gpu
def test_disable_offloading(linear: torch.nn.Linear | OffloadedModule):
    outside_onloaded = linear.weight
    outside_onloaded_ref = ref(outside_onloaded)
    assert outside_onloaded.device == ONLOAD_DEVICE

    with linear.disable_offloading():
        inside_onloaded = linear.weight
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
def test_disable_onloading(linear: torch.nn.Linear | OffloadedModule):
    offloaded_weight = linear._module.weight

    with linear.disable_onloading():
        weight = linear.weight
        assert weight is offloaded_weight

        # new parameter assignments are direct
        new_param = torch.nn.Parameter(torch.ones(5, device=ONLOAD_DEVICE))
        linear.new_param = new_param
        assert linear.new_param is new_param

    assert weight is offloaded_weight


@pytest.mark.unit
@requires_gpu
def test_delete(linear: torch.nn.Linear | OffloadedModule):
    weight_ref = ref(linear.weight)
    bias_ref = ref(linear.bias)

    del linear.weight
    del linear.bias
    gc.collect()

    assert weight_ref() is None
    assert bias_ref() is None


@pytest.mark.unit
@requires_gpu
@pytest.mark.parametrize("no_split", [True, False])
def test_forward_call(linear: torch.nn.Linear | OffloadedModule, no_split):
    linear._no_split = no_split

    with torch.no_grad():
        input = torch.zeros(5, device=OFFLOAD_DEVICE)
        output = linear.forward(input)
        assert output.device == ONLOAD_DEVICE

        def pre_hook(module, args, *_):
            assert args[0].device == ONLOAD_DEVICE
            assert module._cache.offloading_disabled == no_split

        def post_hook(module, args, *_):
            assert args[0].device == ONLOAD_DEVICE
            assert module._cache.offloading_disabled == no_split

        linear.register_forward_pre_hook(pre_hook)
        linear.register_forward_hook(post_hook)

        output = linear(input)
        assert output.device == ONLOAD_DEVICE


def test_modules(cache):
    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear0 = torch.nn.Linear(5, 5)
            self.linear1 = torch.nn.Linear(5, 5)

    parent = Parent()
    parent_modules = list(parent.modules())
    parent_children = list(parent.children())

    parent.linear0 = OffloadedModule.from_module(parent.linear0, cache)
    parent.linear1 = OffloadedModule.from_module(parent.linear1, cache)
    offloaded = OffloadedModule.from_module(parent, cache)

    for (_, module), orig_module in zip(offloaded.named_modules(), parent_modules):
        assert isinstance(module, OffloadedModule)
        assert module._module is orig_module

    for module, orig_module in zip(offloaded.modules(), parent_modules):
        assert isinstance(module, OffloadedModule)
        assert module._module is orig_module

    for (_, module), orig_module in zip(offloaded.named_children(), parent_children):
        assert isinstance(module, OffloadedModule)
        assert module._module is orig_module

    for module, orig_module in zip(offloaded.children(), parent_children):
        assert isinstance(module, OffloadedModule)
        assert module._module is orig_module


@pytest.mark.parametrize("param_device", (ONLOAD_DEVICE, OFFLOAD_DEVICE))
@pytest.mark.parametrize("use_register_parameter", (True, False))
@pytest.mark.parametrize("requires_grad", (True, False))
def test_register_parameter(
    linear: torch.nn.Linear | OffloadedModule,
    param_device,
    use_register_parameter,
    requires_grad,
):
    # register param
    data = torch.ones(5, device=param_device)
    param = torch.nn.Parameter(data, requires_grad=requires_grad)
    if use_register_parameter:
        linear.register_parameter("param_name", param)
    else:
        linear.param_name = param

    # new param is correctly onloaded
    assert linear.param_name.device == ONLOAD_DEVICE
    assert torch.equal(linear.param_name.to(param_device), param)


@pytest.mark.parametrize("param_device", (ONLOAD_DEVICE, OFFLOAD_DEVICE))
@pytest.mark.parametrize("use_register_parameter", (True, False))
@pytest.mark.parametrize("requires_grad", (True, False))
def test_register_parameter_invalidates(
    linear: torch.nn.Linear | OffloadedModule,
    param_device,
    use_register_parameter,
    requires_grad,
):
    with linear.disable_offloading():
        # original weight is kept onloaded
        onloaded_weight = linear.weight
        assert onloaded_weight in linear._cache.keep_onloaded_values

        # add new param
        data = torch.ones(5, device=param_device)
        param = torch.nn.Parameter(data, requires_grad=requires_grad)
        if use_register_parameter:
            linear.register_parameter("weight", param)
        else:
            linear.weight = param

        # new param is correct
        assert linear.weight.device == ONLOAD_DEVICE
        assert torch.equal(linear.weight.to(param_device), param)

        # original weight is invalidated
        assert onloaded_weight not in linear._cache.keep_onloaded_values


def test_forward_signature(linear: torch.nn.Linear | OffloadedModule):
    assert inspect.signature(linear.forward) == inspect.signature(
        linear._module.forward
    )
