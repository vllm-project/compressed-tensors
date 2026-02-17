# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from collections.abc import Iterable
from typing import Literal

import torch
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.convert import from_accelerate, to_accelerate
from compressed_tensors.offload.dispatch import (  # noqa: F401
    dispatch_model,
    dispatch_with_map,
    get_device_map,
    offload_model,
    remove_dispatch,
)
from compressed_tensors.offload.dist_utils import is_distributed, is_rank0
from compressed_tensors.offload.load import load_offloaded_model
from compressed_tensors.offload.module import offload_module, unwrap_offload_forward
from compressed_tensors.offload.utils import get_module_device, move_module_tensor
from compressed_tensors.utils.helpers import patch_attr


__all__ = [
    # dispatch models
    "offload_model",
    "dispatch_model",
    "remove_dispatch",
    "dispatch_with_map",
    "get_device_map",
    # accelerate conversion
    "load_offloaded_model",
    "from_accelerate",
    "to_accelerate",
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
    # utilities
    "is_distributed",
    "is_rank0",
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
    Update the offload and onload data of an existing parameter/buffer. Supports both
    parameters of both offloaded modules and non-offloaded modules.

    NOTE: This function does not guard against multiple processes writing to offload
    at the same time. It is the responsibility of the caller to ensure that, for any
    parameter/buffer, only one rank calls this function at a time.

    NOTE: This function does not update onloaded values across ranks. The caller is
    responsible for broadcasting any updates to other ranks, if they are onloaded.

    :param module: module containing the parameter to update
    :param name: name of module parameter to update
    :param data: tensor to update parameter with
    """
    if isinstance(module._parameters, OffloadCache):
        # | Component | Update Implementation       |
        # | --------- | --------------------------- |
        # | CPU       | Copy into shared cpu memory |
        # | Disk      | Write file to disk          |
        # | Device    | Copy into local device      |
        # | --------- | --------------------------- |
        # all implementations update onloaded data if applicable
        setattr(module, name, torch.nn.Parameter(data.data, requires_grad=False))

    else:
        with torch.no_grad():
            getattr(module, name).copy_(data)


def get_execution_device(
    module: torch.nn.Module, default: torch.device | None = None
) -> torch.device | Literal["disk"]:
    """
    Get the device which inputs should be moved to before module execution.

    :param module: module to check, may be offloaded
    :return: onload device of module
    """
    if isinstance(module._parameters, OffloadCache):
        return module._parameters.onload_device

    else:
        return get_module_device(module, default)


def get_offloaded_device(
    module: torch.nn.Module, default: torch.device | None = None
) -> torch.device | Literal["disk"]:
    """
    :param module: module to check
    :return: device module is offloaded to onto after forward pass
    """
    if isinstance(module._parameters, OffloadCache):
        return module._parameters.offload_device

    else:
        return get_module_device(module, default)


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
    execution_device: torch.device | None = None,
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
    module: torch.nn.Module, execution_device: torch.device | None = None
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
                with (
                    patch_attr(module._parameters, "onload_device", execution_device),
                    patch_attr(module._buffers, "onload_device", execution_device),
                ):
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
