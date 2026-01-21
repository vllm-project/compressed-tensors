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
from dataclasses import fields, is_dataclass
from itertools import chain
from typing import Optional, TypeVar

import torch
from loguru import logger


__all__ = [
    "send_tensors",
    "get_module_device",
    "move_module_tensor",
    "module_size",
]

T = TypeVar("T")


def send_tensors(value: T, *args, **kwargs) -> T:
    """
    Recursively identify and move tensors using `torch.Tensor.to`

    :param value: value containing tensors to move
    :param args: arguments to `to`
    :param kwargs: keyword arguments to `to`
    :return: value with moved tensors
    """
    match value:
        case torch.nn.Parameter():
            data = value.to(*args, **kwargs)
            # special case: avoid changing param pointer when possible
            if data.data_ptr() == value.data_ptr():
                return value
            return value.__class__(data, requires_grad=value.requires_grad)
        case torch.Tensor():
            return value.to(*args, **kwargs)
        case list():
            return [send_tensors(v, *args, **kwargs) for v in value]
        case tuple():
            return tuple(send_tensors(v, *args, **kwargs) for v in value)
        case dict():
            return {k: send_tensors(v, *args, **kwargs) for k, v in value.items()}
        case _ if is_dataclass(value):
            return type(value)(
                **{
                    f.name: send_tensors(getattr(value, f.name), *args, **kwargs)
                    for f in fields(value)
                }
            )
        case _:
            return value


def get_module_device(
    module: torch.nn.Module, default: Optional[torch.device] = None
) -> torch.device:
    """
    Infer the device of a module using the first
    parameter or buffer registered to the module

    :param module: module to check
    :param default: default device if module does not have tensors or buffers
    :return: device of module
    """
    tensor = next(module.parameters(), next(module.buffers(), None))
    if tensor is not None:
        return tensor.device
    elif default is not None:
        return default
    else:
        logger.warning(
            f"Unable to get execution device of {module}, falling back to CPU"
        )
        return torch.device("cpu")


def move_module_tensor(
    module: torch.nn.Module,
    name: str,
    device: int | str | torch.device,
):
    """
    Move a module's tensor to a new device

    :param module: module containing tensors to move
    :param name: name of tensor to move
    :param device: new devices
    """
    if name in module._parameters:
        module._parameters[name] = send_tensors(module._parameters[name], device=device)

    elif name in module._buffers:
        module._buffers[name] = send_tensors(module._buffers[name], device=device)


def get_module_sizes(
    model: torch.nn.Module, no_split_modules: Container[str] = tuple()
) -> list[tuple[torch.nn.Module, int]]:
    """
    Returns a list of modules and their sizes. Only non-splittable modules are returned.
    Non-splittable modules are modules specified by `no_split_modules` or modules with
    direct parameters.

    :param model: model to get sizes from
    :param no_split_modules: module class names which cannot be split
    :return: list of modules and their sizes
    """
    module_sizes = []

    def dfs(module: torch.nn.Module):
        # modules with direct parameters cannot be split
        # otherwise, submodules could return a device that is different than params
        direct_size = module_size(module, recurse=False)
        no_split = module.__class__.__name__ in no_split_modules or direct_size > 0

        total_size = module_size(module, recurse=no_split)
        if total_size > 0:
            module_sizes.append((module, total_size))

        if not no_split:
            for child in module.children():
                dfs(child)

    dfs(model)

    return module_sizes


def module_size(module: torch.nn.Module, recurse: bool = True) -> int:
    """
    Get the size of the module's parameters and buffers in bytes

    :param module: module to check
    :param recurse: whether calculate recursive size, or only direct tensors
    :return: total size of module parameters and buffers
    """
    from compressed_tensors.offload import disable_onloading

    with disable_onloading():
        tensors = chain(
            module.parameters(recurse=recurse), module.buffers(recurse=recurse)
        )
        return sum((tensor.nbytes for tensor in tensors), 0)
