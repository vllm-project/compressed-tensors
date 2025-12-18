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
from typing import Iterable, Iterator, Optional

import torch
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.dispatch import dispatch_model
from compressed_tensors.offload.module import OffloadedModule
from compressed_tensors.offload.utils import get_module_device
from compressed_tensors.utils.helpers import patch_attr


__all__ = [
    "dispatch_model",
    "update_offload_parameter",
    "get_execution_device",
    "disable_offloading",
    "register_offload_module",
    "disable_onloading",
]


def update_offload_parameter(module: torch.nn.Module, name: str, data: torch.Tensor):
    """
    Update the data of an existing parameter and its offload dict. Supports both
    parameters of offloaded modules and non-offloaded modules

    :param module: module containing the parameter to update
    :param name: name of module parameter to update
    :param data: tensor to update parameter with
    """
    if isinstance(module, OffloadedModule):
        with module.disable_onloading():
            getattr(module, name).copy_(data)
    else:
        getattr(module, name).copy_(data)


def get_execution_device(module: torch.nn.Module) -> torch.device | str:
    """
    Get the device which inputs should be moved to before module execution.

    :param module: module to check, may be offloaded
    :return: onload device of module
    """
    if isinstance(module, OffloadedModule):
        return module._cache.onload_device

    else:
        return get_module_device(module)


def get_offloaded_device(module: torch.nn.Module) -> torch.device:
    """
    :param module: module to check
    :return: device module is offloaded to onto after forward pass
    """
    if isinstance(module, OffloadedModule):
        with module.disable_onloading():
            return get_module_device(module)
    else:
        return get_module_device(module)


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


def register_offload_module(base: torch.nn.Module, name: str, module: torch.nn.Module):
    """
    Register a submodule with offloading if the parent module is offloaded

    :param base: module to attach submodule to
    :param name: name of submodule
    :param module: submodule to attach
    """
    if isinstance(base, OffloadedModule):
        offloaded = OffloadedModule.from_module(module, base._cache, no_split=False)
        base.register_module(name, offloaded)


@contextlib.contextmanager
def disable_offloading():
    """
    Keep modules onloaded and disable offloading until this context exits.
    Affects modules which have been hooked with accelerate's `AlignDevicesHook`
    """
    with contextlib.ExitStack() as stack:
        for cache in OffloadCache.instances():
            stack.enter_context(cache.disable_offloading())
        yield


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
    with disable_offloading():
        if execution_device is None:
            yield

        elif isinstance(module, OffloadedModule):
            with patch_attr(module, "execution_device", execution_device):
                yield

        else:
            from accelerate.utils import set_module_tensor_to_device

            devices = {
                name: param.device
                for name, param in module.named_parameters(recurse=False)
            }
            try:
                for name in devices:
                    set_module_tensor_to_device(module, name, execution_device)
                yield
            finally:
                for name, device in devices.items():
                    set_module_tensor_to_device(module, name, device)
            # with contextlib.ExitStack() as stack:
            #     for name, param in module.named_parameters(recurse=False):
            #         data = param.to(execution_device)
            #         param = torch.nn.Parameter(data, requires_grad=False)
            #         stack.enter_context(patch_attr(module, name, param))
            #     yield


@contextlib.contextmanager
def unwrap_offload(module: torch.nn.Module) -> Iterator[torch.nn.Module]:
    """
    Context manager that returns the module without offload wrapping.
    This can be used to modify the original module. The module is rewrapped upon exit

    :param module: module that may be offloaded
    :returns: module with offload wrapping removed
    """
    if isinstance(module, OffloadedModule):
        yield module._module
    else:
        yield module
