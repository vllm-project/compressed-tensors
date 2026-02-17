# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


def has_memory_requirements(device_memory: dict[torch.device, int]):
    real_device_memory = get_device_memory()
    for key, req in device_memory.items():
        if key not in real_device_memory or real_device_memory[key] < req:
            return False

    return True


@pytest.mark.unit
@requires_gpu
def test_dispatch_one_device():
    model = Model()
    device_memory = {torch.device("cuda:0"): module_size(model)}
    if not has_memory_requirements(device_memory):
        pytest.skip("Cannot perform one device dispatch test, not enough device memory")

    dispatch_model(model, device_memory=device_memory, extra_memory=0)
    assert_module_on_device(model, "cuda:0")


@pytest.mark.unit
@requires_gpu
def test_dispatch_two_devices():
    model = Model()
    device_memory = {
        torch.device("cuda:0"): module_size(model.decoder0),
        torch.device("cuda:1"): module_size(model) - module_size(model.decoder0),
    }
    if not has_memory_requirements(device_memory):
        pytest.skip("Cannot perform split dispatch test: not enough devices or memory")

    # first decoder on first device, rest on second device
    dispatch_model(model, device_memory=device_memory, extra_memory=0)
    assert_module_on_device(model.decoder0, "cuda:0")
    assert_module_on_device(model.decoder1, "cuda:1")


@pytest.mark.unit
@requires_gpu
def test_dispatch_no_split():
    model = Model()
    device_memory = {
        torch.device("cuda:0"): module_size(model.decoder0.linear0),
        torch.device("cuda:1"): module_size(model),
    }
    if not has_memory_requirements(device_memory):
        pytest.skip("Cannot perform split dispatch test: not enough devices or mem")

    # first device is skipped: all ends up on second device
    dispatch_model(model, device_memory=device_memory, extra_memory=0)
    assert_module_on_device(model, "cuda:1")


@pytest.mark.unit
@requires_gpu
def test_dispatch_split():
    model = Model()
    first_linear = model.decoder0.linear0
    device_memory = {
        torch.device("cuda:0"): module_size(first_linear),
        torch.device("cuda:1"): module_size(model) - module_size(first_linear),
    }
    if not has_memory_requirements(device_memory):
        pytest.skip("Cannot perform split dispatch test: not enough devices or memory")

    # first linear on first device, rest on second device
    dispatch_model(model, device_memory=device_memory, no_split_modules=tuple(), extra_memory=0)
    assert_module_on_device(model.decoder0.linear0, "cuda:0")
    assert_module_on_device(model.decoder0.linear1, "cuda:1")
    assert_module_on_device(model.decoder1, "cuda:1")


@pytest.mark.unit
@requires_gpu
def test_dispatch_offloaded():
    model = Model()
    device_memory = {
        torch.device("cuda:0"): (
            module_size(model.decoder0.linear0) + module_size(model.decoder1)
        ),
    }
    if not has_memory_requirements(device_memory):
        pytest.skip("Cannot perform split dispatch test: not enough devices or mem")

    with patch("compressed_tensors.offload.dispatch.get_module_sizes") as mock_sizes:
        # first two linears are disjoint, but not enough memory to fit decoder1
        mock_sizes.return_value = [
            (model.decoder0.linear0, module_size(model.decoder0.linear0)),
            (model.decoder0.linear1, module_size(model.decoder0.linear1)),
            (model.decoder1, module_size(model.decoder1)),
        ]

        # first linear stays onloaded
        # second linear is popped off to fit offloaded decoder1
        dispatch_model(model, device_memory=device_memory, no_split_modules=tuple(), extra_memory=0)
        assert_module_on_device(model.decoder0.linear0, "cuda:0")
        assert_module_offloaded(model.decoder0.linear1, "cuda:0", "cpu")
        assert_module_offloaded(model.decoder1, "cuda:0", "cpu")


@pytest.mark.integration
@requires_gpu
@pytest.mark.parametrize("model_id", ["nm-testing/tinysmokellama-3.2"])
@torch.inference_mode()
def test_offload_and_dispatch_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tied_tensors_size = model.lm_head.weight.nbytes
    device_memory = {torch.device("cuda:0"): module_size(model) + tied_tensors_size}
    if not has_memory_requirements(device_memory):
        pytest.skip("Cannot perform split dispatch test: not enough devices or mem")

    model.to("cuda:0")
    sample = tokenizer("Hello my name is", return_tensors="pt")
    sample = {k: v.to("cuda:0") for k, v in sample.items()}
    true_logits = model(**sample).logits

    # offload entire model
    model.to("cpu")
    model = offload_model(model, "cuda:0")
    offloaded_logits = model(**sample).logits
    for module in model.modules():
        assert_module_offloaded(module, "cuda:0", torch.device("cpu"))
    assert torch.allclose(offloaded_logits, true_logits)

    # dispatch model and fits
    model = dispatch_model(model, device_memory=device_memory, extra_memory=0)
    dispatched_logits = model(**sample).logits
    for module in model.modules():
        assert_module_on_device(module, "cuda:0")
    assert torch.allclose(dispatched_logits, true_logits)

    # dispatch model with offload
    device_memory[torch.device("cuda:0")] = device_memory[torch.device("cuda:0")] // 2
    model = dispatch_model(model, device_memory=device_memory, extra_memory=0)
    dispatched_logits = model(**sample).logits
    assert torch.allclose(dispatched_logits, true_logits)
