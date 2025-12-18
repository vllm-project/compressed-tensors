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
from itertools import chain
from typing import Literal, NamedTuple, Optional, TypeVar

import torch
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.module import OffloadedModule
from compressed_tensors.offload.utils import move_module_tensor
from compressed_tensors.utils import getattr_chain
from loguru import logger
from transformers import PreTrainedModel


__all__ = ["offload_model", "dispatch_model", "remove_dispatch"]

ModelType = TypeVar("", bound=torch.nn.Module)


def offload_model(
    model: ModelType,
    onload_device: torch.device | str,
    offload_device: Optional[torch.device | str | Literal["disk"]] = None,
    no_split_modules: Container[str] = tuple(),
) -> ModelType:
    """
    Dispatch a model using offloading

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
    if len(no_split_modules) <= 0 and isinstance(model, PreTrainedModel):
        no_split_modules = getattr(model, "_no_split_modules", tuple())

    # each model shares a single shared cache because we have to
    # coordinate the onloading of shared tensors within the model
    cache = OffloadCache.from_devices(onload_device, offload_device)
    memo = dict()
    for name, module in model.named_modules(remove_duplicate=False):
        # exclude wrapping the root
        if name == "" or isinstance(module, torch.nn.ModuleList):
            continue

        no_split = module.__class__.__name__ in no_split_modules
        offloaded_module = OffloadedModule.from_module(module, cache, no_split)

        model.set_submodule(name, offloaded_module)
        memo[module] = offloaded_module

    return model


def dispatch_model(
    model: ModelType,
    hint_batch_size: int = 1,
    hint_batch_seq_len: int = 1,
    hint_extra_memory: int = 0,
    hint_model_dtype: Optional[torch.dtype] = None,
    no_split_modules: Container[str] = tuple(),
) -> ModelType:
    # remove previous dispatches
    model = remove_dispatch(model)

    # infer no_split_modules
    if len(no_split_modules) <= 0 and isinstance(model, PreTrainedModel):
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
    if len(devices) <= 0:
        raise MemoryError("Did not find any devices to dispatch model to")

    # estimate model size
    _, model_size = module_nbytes(model)
    if model_size > sum((device.memory for device in devices), 0):
        raise MemoryError(
            f"Cannot dispatch model of size {model_size} "
            f"bytes to devices:\n{devices}"
        )

    # allocate a fallback cache if we ever run out of memory
    cache = OffloadCache.from_devices(devices[0].device, torch.device("cpu"))

    # assign modules to devices
    def dfs(module: torch.nn.Module) -> torch.nn.Module:
        no_split = module.__class__.__name__ in no_split_modules
        direct_size, total_size = module_nbytes(module)

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

                for name, child in module.named_children(recurse=False):
                    module.add_module(name, dfs(child))

                return module

    return dfs(model)


class DeviceMemory(NamedTuple):
    device: torch.device | str
    memory: int


def get_device_memory(hint_extra_memory: int) -> list[DeviceMemory]:
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


def module_nbytes(module: torch.nn.Module) -> tuple[int, int]:
    direct = sum(
        (
            param.nbytes
            for param in chain(
                module.parameters(recurse=False), module.buffers(recurse=False)
            )
        ),
        0,
    )
    total = sum(
        (
            param.nbytes
            for param in chain(
                module.parameters(recurse=True), module.buffers(recurse=True)
            )
        ),
        0,
    )
    return direct, total


def module_to(
    module: torch.nn.Module, device: torch.device, recurse: bool = False
) -> torch.nn.Module:
    if recurse:
        return module.to(device)
    else:
        for name in chain(module._parameters.keys(), module._buffers.keys()):
            move_module_tensor(module, name, device)
        return module


def remove_dispatch(module: torch.nn.Module) -> torch.nn.Module:
    """
    Remove any existing dispatches from module

    :param module: module which may be dispatched with hf hooks
    :return: module without dispatch
    """
    for name, submodule in module.named_modules():
        if isinstance(submodule, OffloadedModule):
            if name == "":
                module = submodule._module
            else:
                module.set_submodule(name, submodule._module)

    return module
