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

import contextlib
from typing import Iterable, Optional

import torch
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.dispatch import (  # noqa: F401
    dispatch_model,
    offload_model,
    remove_dispatch,
)
from compressed_tensors.offload.module import offload_module, unwrap_offload_forward
from compressed_tensors.offload.utils import get_module_device, move_module_tensor
from compressed_tensors.utils.helpers import patch_attr


__all__ = [
    # dispatch models
    "offload_model",
    "dispatch_model",
    "remove_dispatch",
    # control movement
    "disable_onloading",
    "disable_offloading",
    # manipulate parameters
    "update_offload_parameter",
    "get_execution_device",
    "get_offloaded_device",
    "register_offload_module",
    # manipulate forward
    "unwrap_offload_forward",
    # backwards compatibility: should be deprecated
    "align_modules",
    "align_module_device",
]


@contextlib.contextmanager
def disable_offloading():
    """
    When offloading is disabled, onloaded tensors remain onloaded in memory until exit

    ```
    with OffloadCache.disable_offloading():
        ... = cache["weight"]
        ... = cache["weight"]  # cache hit
        ... = cache["weight"]  # cache hit

    # upon exit, all onloaded weights are released
    ```
    """
    with OffloadCache.disable_offloading():
        yield


@contextlib.contextmanager
def disable_onloading():
    """
    When onloading is disabled, tensors are not offloaded on access, and assignments do
    not trigger offloading. This is mostly used to disable device movement for debugging

    ```
    with OffloadCache.disable_onloading():
        tensor = ...
        cache["weight"] = tensor   # assignments do not trigger onloading
        cache["weight"] is tensor  # tensor remains offloaded
    ```
    """
    with OffloadCache.disable_onloading():
        yield


def update_offload_parameter(module: torch.nn.Module, name: str, data: torch.Tensor):
    """
    Update the data of an existing parameter and its offload dict. Supports both
    parameters of offloaded modules and non-offloaded modules

    :param module: module containing the parameter to update
    :param name: name of module parameter to update
    :param data: tensor to update parameter with
    """
    if isinstance(module._parameters, OffloadCache):
        with module._parameters.disable_onloading():
            value = getattr(module, name)
            value.copy_(module._parameters.offload(data))
            setattr(module, name, value)

    else:
        getattr(module, name).copy_(data)


def get_execution_device(module: torch.nn.Module) -> torch.device | str:
    """
    Get the device which inputs should be moved to before module execution.

    :param module: module to check, may be offloaded
    :return: onload device of module
    """
    if isinstance(module._parameters, OffloadCache):
        return module._parameters.onload_device

    else:
        return get_module_device(module)


def get_offloaded_device(module: torch.nn.Module) -> torch.device:
    """
    :param module: module to check
    :return: device module is offloaded to onto after forward pass
    """
    with disable_onloading():
        return get_module_device(module)


def register_offload_module(base: torch.nn.Module, name: str, module: torch.nn.Module):
    """
    Register a submodule with offloading if the parent module is offloaded

    :param base: module to attach submodule to
    :param name: name of submodule
    :param module: submodule to attach
    """
    cache = base._parameters
    if isinstance(cache, OffloadCache):
        offload_module(module, cache.onload_device, cache.offload_device)

    base.register_module(name, module)


""" Implemented for backwards compatibility """


@contextlib.contextmanager
def align_modules(
    modules: torch.nn.Module | Iterable[torch.nn.Module],
    execution_device: Optional[torch.device] = None,
):
    """
    Context manager for onloading modules to a device, and disabling onload and offload
    attempts triggered by forward calls. Used for sequential onloading of layers

    :param modules: `torch.nn.Module` or iterable of `torch.nn.Module`s to onload
    :param execution_device: device to onload to
    """
    with contextlib.ExitStack() as stack:
        for module in modules:
            stack.enter_context(align_module_device(module, execution_device))
        yield


@contextlib.contextmanager
def align_module_device(
    module: torch.nn.Module, execution_device: Optional[torch.device] = None
):
    """
    Context manager that moves a module's parameters to the specified execution device.

    :param module: Module with parameters to align
    :param execution_device: If provided, overrides the module's execution device
        within the context. Otherwise, use hook execution device or pass
    """

    if isinstance(module._parameters, OffloadCache):
        assert isinstance(module._buffers, OffloadCache)
        with module._parameters.disable_offloading():
            if execution_device is not None:
                with patch_attr(
                    module._parameters, "onload_device", execution_device
                ), patch_attr(module._buffers, "onload_device", execution_device):
                    yield
            else:
                yield

    else:
        original_device = {}
        for name, param in module.named_parameters(recurse=False):
            original_device[name] = param.device
            move_module_tensor(module, name, execution_device)

        try:
            yield
        finally:
            for name, param in module.named_parameters(recurse=False):
                device = original_device[name]
                move_module_tensor(module, name, device)
