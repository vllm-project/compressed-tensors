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

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from compressed_tensors.offload.cache import CPUCache, OffloadCache
from compressed_tensors.offload.dispatch import (
    dispatch_model,
    get_device_memory,
    offload_model,
)
from compressed_tensors.offload.utils import module_size
from tests.testing_utils import requires_gpu
from transformers import AutoModelForCausalLM, AutoTokenizer


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(5, 5)
        self.linear1 = torch.nn.Linear(5, 5)

    def forward(self, input):
        return self.linear1(self.linear0(input))


class Model(torch.nn.Module):
    _no_split_modules = ["Decoder"]

    def __init__(self):
        super().__init__()
        self.decoder0 = Decoder()
        self.decoder1 = Decoder()

    def forward(self, input):
        return self.decoder1(self.decoder0(input))


def assert_module_on_device(module: torch.nn.Module, device: torch.device | str):
    assert not isinstance(module._parameters, CPUCache)
    for name, param in module.named_parameters():
        assert torch.device(param.device) == torch.device(device), name


def assert_module_offloaded(
    module: torch.nn.Module,
    onload_device: torch.device | str,
    offload_device: torch.device | str,
    req_params: bool = False,
):
    for name, submodule in module.named_modules():
        if isinstance(submodule, torch.nn.ModuleList):
            continue
        if req_params and module_size(submodule)[0] <= 0:
            continue

        assert isinstance(submodule._parameters, OffloadCache), name
        assert torch.device(submodule._parameters.onload_device) == torch.device(
            onload_device
        )
        assert torch.device(submodule._parameters.offload_device) == torch.device(
            offload_device
        )


@pytest.mark.unit
@requires_gpu
def test_dispatch_one_device():
    model = Model()
    memory = get_device_memory()
    if len(memory) < 1 or memory[0].memory < module_size(model)[1]:
        pytest.skip("Cannot perform one device dispatch test, not enough device memory")

    with patch("compressed_tensors.offload.dispatch.get_device_memory") as mock_fn:
        mock_fn.return_value = deepcopy(memory[:1])

        model = dispatch_model(model)
        assert_module_on_device(model, memory[0].device)


@pytest.mark.unit
@requires_gpu
def test_dispatch_two_devices():
    model = Model()
    memory = get_device_memory()
    if (
        len(memory) < 2
        or memory[0].memory < module_size(model.decoder0)[1]
        or memory[1].memory < module_size(model)[1] - module_size(model.decoder0)[1]
    ):
        pytest.skip("Cannot perform split dispatch test: not enough devices or memory")

    memory[0].memory = module_size(model.decoder0)[1]
    memory[1].memory = module_size(model)[1] - module_size(model.decoder0)[1]
    memory = memory[:2]

    with patch("compressed_tensors.offload.dispatch.get_device_memory") as mock_fn:
        mock_fn.return_value = memory

        # first decoder on first device, rest on second device
        model = dispatch_model(model)
        assert_module_on_device(model.decoder0, memory[0].device)
        assert_module_on_device(model.decoder1, memory[1].device)


@pytest.mark.unit
@requires_gpu
def test_dispatch_no_split():
    model = Model()
    memory = get_device_memory()
    first_linear = model.decoder0.linear0
    if (
        len(memory) < 2
        or memory[0].memory < module_size(first_linear)[1]
        or memory[1].memory < module_size(model)[1]
    ):
        pytest.skip("Cannot perform split dispatch test: not enough devices or mem")

    # reduce memory of first device
    memory[0].memory = module_size(first_linear)[1]
    memory[1].memory = module_size(model)[1]
    memory = memory[:2]

    with patch("compressed_tensors.offload.dispatch.get_device_memory") as mock_fn:
        mock_fn.return_value = memory

        # first device is skipped: all ends up on second device
        model = dispatch_model(model)
        assert_module_on_device(model, memory[1].device)


@pytest.mark.unit
@requires_gpu
def test_dispatch_split():
    model = Model()
    memory = get_device_memory()
    first_linear = model.decoder0.linear0
    if (
        len(memory) < 2
        or memory[0].memory < module_size(first_linear)[1]
        or memory[1].memory < module_size(model)[1] - module_size(first_linear)[1]
    ):
        pytest.skip("Cannot perform split dispatch test: not enough devices or memory")

    memory[0].memory = module_size(first_linear)[1]
    memory[1].memory = module_size(model)[1] - module_size(first_linear)[1]
    memory = memory[:2]

    with patch("compressed_tensors.offload.dispatch.get_device_memory") as mock_fn:
        mock_fn.return_value = memory

        # first linear on first device, rest on second device
        model = dispatch_model(model, no_split_modules=tuple())
        assert_module_on_device(model.decoder0.linear0, memory[0].device)
        assert_module_on_device(model.decoder0.linear1, memory[1].device)
        assert_module_on_device(model.decoder1, memory[1].device)


@pytest.mark.unit
@requires_gpu
def test_dispatch_offloaded():
    model = Model()
    memory = get_device_memory()
    if len(memory) < 1 or memory[0].memory < 360:
        pytest.skip("Cannot perform split dispatch test: not enough devices or mem")

    memory[0].memory = 360  # fits three linear layers
    memory = memory[:1]

    with patch(
        "compressed_tensors.offload.dispatch.get_device_memory"
    ) as mock_mem, patch(
        "compressed_tensors.offload.dispatch.get_module_sizes"
    ) as mock_sizes:
        mock_mem.return_value = deepcopy(memory)
        # first two linears are disjoint, but not enough memory to fit decoder1
        mock_sizes.return_value = [
            (model.decoder0.linear0, 120),
            (model.decoder0.linear1, 120),
            (model.decoder1, 240),
        ]

        # first linear stays onloaded
        # second linear is popped off to fit offloaded decoder1
        model = dispatch_model(model, no_split_modules=tuple())
        assert_module_on_device(model.decoder0.linear0, memory[0].device)
        assert_module_offloaded(model.decoder0.linear1, memory[0].device, "cpu")
        assert_module_offloaded(model.decoder1, memory[0].device, "cpu")


@pytest.mark.integration
@requires_gpu
@pytest.mark.parametrize("model_id", ["nm-testing/tinysmokellama-3.2"])
@torch.inference_mode()
def test_offload_and_dispatch_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda").eval()

    sample = tokenizer("Hello my name is", return_tensors="pt")
    sample = {k: v.to("cuda:0") for k, v in sample.items()}
    true_logits = model(**sample).logits

    model = offload_model(model, "cuda:0", "cpu")
    offloaded_logits = model(**sample).logits
    for child in model.children():
        assert_module_offloaded(child, "cuda:0", torch.device("cpu"))
    assert torch.allclose(offloaded_logits, true_logits)

    # model = dispatch_model(model)
    # dispatched_logits = model(**sample).logits
    # TODO: make test only one gpu, i guess
    # assert_module_on_device(model, get_device_memory()[0].device)
    # assert torch.allclose(dispatched_logits.to(true_logits.device), true_logits)
