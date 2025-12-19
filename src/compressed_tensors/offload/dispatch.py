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
from dataclasses import dataclass
from typing import Literal, Optional, TypeVar

import torch
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.module import OffloadedModule
from compressed_tensors.offload.utils import module_size, module_to
from compressed_tensors.utils import getattr_chain
from loguru import logger
from transformers import PreTrainedModel


__all__ = [
    "offload_model",
    "dispatch_model",
    "remove_dispatch",
    "get_device_memory",
    "DeviceMemory",
]

ModelType = TypeVar("", bound=torch.nn.Module)


def offload_model(
    model: ModelType,
    onload_device: torch.device | str,
    offload_device: Optional[torch.device | str | Literal["disk"]] = None,
    no_split_modules: Optional[Container[str]] = None,
) -> ModelType:
    """
    Offload a model to the `offload_device`. During forward passes, model weights will
    be onloaded to the `onload_device`

    Disclaimers:
    * Shared modules are not preserved

    :param model: model to dispatch
    :param onload_device: device to move weights to during forward pass
    :param offload_device: device to offload weights to
    :param no_split_modules: names of module classes which should not be split
        across multiple devices
    :return: dispatched model
    """
    if len(model._parameters) > 0:
        raise NotImplementedError(
            "Offloading is achieved by replacing modules which have direct parameters "
            "with new modules which have been wrapped. However, replacing the root "
            "can break functionality with previous implementation of `dispatch_model`. "
            "Please either remove any direct parameters to the model root, or refactor "
            "this function and its usages to use the new, wrapped root"
        )

    # ensure model starts offloaded
    model = remove_dispatch(model)
    if offload_device not in (None, "disk"):
        model = model.to(offload_device)

    # infer no_split_modules
    if no_split_modules is None:
        no_split_modules = getattr(model, "_no_split_modules", tuple())

    # each model shares a single shared cache because we have to
    # coordinate the onloading of shared tensors within the model
    cache = OffloadCache.from_devices(onload_device, offload_device)
    for name, module in model.named_modules(remove_duplicate=False):
        # exclude wrapping the root
        if name == "" or isinstance(module, torch.nn.ModuleList):
            continue

        # create offloaded version of module
        no_split = module.__class__.__name__ in no_split_modules
        offloaded = OffloadedModule.from_module(module, cache, no_split)

        # replace in model
        model.set_submodule(name, offloaded)

    return model


def dispatch_model(
    model: ModelType,
    hint_batch_size: int = 0,
    hint_batch_seq_len: int = 0,
    hint_model_dtype: Optional[torch.dtype] = None,
    hint_extra_memory: int = 0,
    no_split_modules: Optional[Container[str]] = None,
) -> ModelType:
    """
    Dispatch a model autoregressive generation. This means that modules are dispatched
    evenly across avaiable devices and kept onloaded if possible.

    Disclaimers:
    * Shared modules are not preserved
    * Optimal runtime assumes that modules are called in order of `model.modules()`

    :param model: model to dispatch
    :param hint_batch_size: reserve memory for batch size of inputs
    :param hint_batch_seq_len: reserve memory for sequence of length of inputs
    :param hint_model_dtype: reserve memory for model's dtype.
        Will be inferred from model if none is provided
    :param hint_extra_memory: extra memory reserved for model serving
    :param no_split_modules: names of module classes which should not be split
        across multiple devices
    :return: dispatched model
    """
    # remove previous dispatches
    model = remove_dispatch(model)

    # infer no_split_modules
    if no_split_modules is None:
        no_split_modules = getattr(model, "_no_split_modules", tuple())

    # infer dtype
    if hint_model_dtype is None:
        hint_model_dtype = getattr(model, "dtype", torch.bfloat16)

    # estimate memory requirement
    if isinstance(model, PreTrainedModel):
        hidden_dim: int = getattr_chain(model, "_config.hidden_dim", 0)
        hint_extra_memory += (
            hint_batch_size
            * hint_batch_seq_len
            * hidden_dim
            * hint_model_dtype.itemsize
        )

    # collect devices
    devices: list[DeviceMemory] = get_device_memory(hint_extra_memory)
    total_memory = sum((device.memory for device in devices), 0)
    if len(devices) <= 0:
        raise MemoryError("Did not find any devices to dispatch model to")

    # estimate model size
    _, model_size = module_size(model)
    if model_size > sum((device.memory for device in devices), 0):
        logger.warning(
            f"Model has size {model_size} bytes, but only {total_memory} bytes"
            "of device memory is available. Resorting to CPU offloading."
        )

    # allocate a fallback cache if we ever run out of memory
    cache = OffloadCache.from_devices(devices[0].device, torch.device("cpu"))

    # assign modules to devices
    def dfs(module: torch.nn.Module) -> torch.nn.Module:
        no_split = module.__class__.__name__ in no_split_modules
        direct_size, total_size = module_size(module)

        # no devices left
        if total_size > 0 and len(devices) <= 0:
            logger.warning(
                "Could not dispatch module of size "
                f"{total_size if no_split else direct_size} bytes. "
                "Resorting to CPU offloading."
            )
            return OffloadedModule.from_module(module, cache, no_split=no_split)

        # can fit entire module
        if total_size <= devices[0].memory:
            devices[0].memory -= total_size
            return module_to(module, devices[0].device, recurse=True)

        else:
            # cannot split, try with next device
            if no_split or direct_size > devices[0].memory:
                devices.pop(0)
                return dfs(module)

            # can split, assign and recurse on children
            else:
                assert direct_size <= devices[0].memory
                devices[0].memory -= direct_size
                module = module_to(module, devices[0].device, recurse=False)

                for name, child in module.named_children():
                    module.add_module(name, dfs(child))

                return module

    return dfs(model)


@dataclass
class DeviceMemory:
    device: torch.device | str
    memory: int


def get_device_memory(hint_extra_memory: int = 0) -> list[DeviceMemory]:
    """
    Get the total memory of all available cuda devices

    :param hint_extra_memory: amount of memory to subtract from each device's total
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
                memory=props.total_memory - hint_extra_memory,
            )
        )
    return devices


def remove_dispatch(module: torch.nn.Module) -> torch.nn.Module:
    """
    Remove any existing dispatches from module

    :param module: module which may be dispatched with hf hooks
    :return: module without dispatch
    """
    for name, submodule in module.named_modules(remove_duplicate=False):
        if isinstance(submodule, OffloadedModule):
            if name == "":
                module = submodule._module
            else:
                module.set_submodule(name, submodule._module)

    return module
