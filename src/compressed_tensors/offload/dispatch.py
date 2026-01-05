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

from collections.abc import Container
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional, TypeVar

import torch
from compressed_tensors.offload.module import offload_module, remove_module_offload
from compressed_tensors.offload.utils import get_module_sizes
from loguru import logger


__all__ = [
    "offload_model",
    "dispatch_model",
    "remove_dispatch",
    "get_device_memory",
    "DeviceMemory",
]

ModelType = TypeVar("ModelType", bound=torch.nn.Module)


def offload_model(
    model: ModelType,
    onload_device: torch.device | str,
    offload_device: Optional[torch.device | str | Literal["disk"]] = None,
    no_split_modules: Optional[Container[str]] = None,
) -> ModelType:
    """
    Offload a model to the `offload_device`. During forward passes, model weights will
    be onloaded to the `onload_device`

    :param model: model to dispatch
    :param onload_device: device to move weights to during forward pass
    :param offload_device: device to offload weights to
    :param no_split_modules: names of module classes which should not be split
        across multiple devices
    :return: dispatched model
    """
    # remove any previous dispatches
    remove_dispatch(model)

    # infer no_split_modules
    if no_split_modules is None:
        no_split_modules = getattr(model, "_no_split_modules", tuple())

    # offload modules in place
    for name, module in model.named_modules():
        no_split = module.__class__.__name__ in no_split_modules
        offload_module(module, onload_device, offload_device, no_split)

    return model


def dispatch_model(
    model: ModelType,
    no_split_modules: Optional[Container[str]] = None,
) -> ModelType:
    """
    Dispatch a model for autoregressive generation. This means that modules are
    dispatched evenly across available devices and kept onloaded if possible.

    Disclaimers:
    * Optimal runtime assumes that modules are called in order of `model.modules()`

    :param model: model to dispatch
    :param no_split_modules: names of module classes which should not be split
        across multiple devices
    :return: dispatched model
    """
    # remove previous dispatches
    remove_dispatch(model)

    # infer no_split_modules
    if no_split_modules is None:
        no_split_modules = getattr(model, "_no_split_modules", tuple())

    # collect devices
    devices: list[DeviceMemory] = get_device_memory()
    if len(devices) <= 0:
        raise MemoryError("Did not find any devices to dispatch model to")

    # collect module sizes
    sizes = get_module_sizes(model, no_split_modules)

    # linear search
    max_extra_memory = min(device.memory for device in devices)
    search_step = 100  # TODO: make configurable, and/or use binary search
    for extra_memory in reversed(range(0, max_extra_memory + search_step, search_step)):
        dispatch = get_greedy_dispatch(sizes, devices, extra_memory)

        if dispatch is not None:
            for module, device in dispatch.items():
                for submodule in module.modules():
                    offload_module(submodule, device, device, no_split=True)

            logger.debug(f"Dispatched model with {extra_memory} bytes of extra memory")
            break
    else:
        raise NotImplementedError(
            "CPU Offloading is not implemented for dispatch_model yet"
        )

    return model


def get_greedy_dispatch(
    sizes: list[tuple[torch.nn.Module, int]],
    devices: list["DeviceMemory"],
    extra_memory: int = 0,
) -> dict[torch.nn.Module, torch.device]:
    dispatch = dict()
    memory_remaining = deepcopy(devices)
    for module, size in sizes:
        while True:
            if len(memory_remaining) <= 0:
                return None

            if size > memory_remaining[0].memory - extra_memory:
                memory_remaining.pop(0)
                continue

            dispatch[module] = memory_remaining[0].device
            memory_remaining[0].memory -= size
            break

    return dispatch


@dataclass
class DeviceMemory:
    device: torch.device | str
    memory: int


def get_device_memory() -> list[DeviceMemory]:
    """
    Get the total memory of all available cuda devices

    :return: list of device memory dataclasses
    """
    if not torch.cuda.is_available():
        return []

    devices: list[DeviceMemory] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        devices.append(
            DeviceMemory(
                device=torch.device(f"cuda:{idx}"),
                memory=props.total_memory,
            )
        )
    return devices


def remove_dispatch(
    module: torch.nn.Module, onload_tensors: bool = False
) -> torch.nn.Module:
    """
    Remove any existing dispatches from module

    :param onload_tensors: Whether to move tensors to the onloaded device, or keep them
        on the offload device. Defaults to False.
    :return: module with offloading functionality removed
    """
    for submodule in module.modules():
        remove_module_offload(submodule, onload_tensors)

    return module
