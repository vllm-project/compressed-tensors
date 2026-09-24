# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import warnings
from functools import wraps

import torch
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.utils import send_tensors


def offload_module(
    module: torch.nn.Module,
    onload_device: torch.device | str,
    offload_device: torch.device | str,
    **kwargs,
):
    """
    Offload a module. Any existing parameters or buffers will be offloaded to the
    offload device specified by the `cache`. Accessing module parameters or buffers will
    cause them to be onloaded to the `onload_device`.

    Calling `forward` will result in input tensors being moved to the `onload_device`,
    and any onloaded parameters or buffers will remain onloaded for the duration of
    the forward call if `no_split` is set to `True`.

    :param module: module to offload
    :param onload_device: device used to onload parameters and buffers
    :param offload_device: device used to offload parameters and buffers
    :param \\**kwargs: keyword arguments for cache constructor
    """
    if isinstance(module._parameters, OffloadCache):
        raise ValueError(
            "Attempted to offload a module twice. "
            "Please call `remove_module_offload` first."
        )

    cache_cls = OffloadCache.cls_from_device(offload_device)
    module._parameters = cache_cls.from_mapping(
        module._parameters, onload_device, offload_device, **kwargs
    )
    module._buffers = cache_cls.from_mapping(
        module._buffers, onload_device, offload_device, **kwargs
    )

    original_forward_func = module.forward.__func__
    module._original_forward_func = original_forward_func

    @wraps(original_forward_func)
    def forward(self, *args, **kwargs):
        if not OffloadCache.onloading_disabled and isinstance(
            module._parameters, OffloadCache
        ):
            onload_device = module._parameters.onload_device
            args = send_tensors(args, device=onload_device)
            kwargs = send_tensors(kwargs, device=onload_device)

        return self._original_forward_func(self, *args, **kwargs)

    module.forward = forward.__get__(module)
    module.is_staged = False

    return module


def stage_module_offload(module: torch.nn.Module, pin_memory: bool = False):
    """
    Stage all offloaded tensors in a module for a later onload.

    :param module: module whose offloaded tensors should be staged
    :param pin_memory: whether to use page-locked CPU memory
    """
    if isinstance(module._parameters, OffloadCache):
        assert isinstance(module._buffers, OffloadCache)
        module._parameters.offloaded_values = {
            name: module._parameters.stage(tensor, pin_memory=pin_memory)
            for name, tensor in module._parameters.offloaded_values.items()
        }
        module._parameters.is_staged = True
        module._buffers.offloaded_values = {
            name: module._buffers.stage(tensor, pin_memory=pin_memory)
            for name, tensor in module._buffers.offloaded_values.items()
        }
        module._buffers.is_staged = True

        module.forward = module._original_forward_func.__get__(module)
        del module._original_forward_func


def remove_module_offload(module: torch.nn.Module, onload_tensors: bool = False):
    """
    Remove any offloading applied to the module

    :param onload_tensors: Whether to move tensors to the onloaded device, or keep them
        on the offload device. Defaults to False.
    """
    if isinstance(module._parameters, OffloadCache):
        assert isinstance(module._buffers, OffloadCache)

        if not module._parameters.is_staged:
            # for staged modules, the forward is already restored
            module.forward = module._original_forward_func.__get__(module)
            del module._original_forward_func

        if onload_tensors:
            module._parameters = {
                name: module._parameters.onload(param)
                for name, param in module._parameters.offloaded_values.items()
            }
            module._buffers = {
                name: module._buffers.onload(param)
                for name, param in module._buffers.offloaded_values.items()
            }
        else:
            module._parameters = module._parameters.offloaded_values
            module._buffers = module._buffers.offloaded_values


@contextlib.contextmanager
def unwrap_offload_forward(module: torch.nn.Module):
    """
    Upon entering, module forward function is unwrapped. Upon exiting the offloading
    wrapper is added again. Any modifications made to the forward function while within
    the context will be reflected upon exiting.
    """
    if hasattr(module, "_original_forward_func"):
        offload_forward = module.forward
        module.forward = module._original_forward_func.__get__(module)
        yield
        module._original_forward_func = module.forward.__func__
        module.forward = offload_forward

    else:
        yield


def subgraph_stage_modules(
    modules: dict[str, torch.nn.Module],
    pin_memory: bool = False,
) -> None:
    """Stage offloaded module tensors in CPU memory for a later onload."""
    for name, module in modules.items():
        if not isinstance(module._parameters, OffloadCache):
            # we should consider raising warnings, but observers will
            # clog the output with warnings, so we will skip for now
            continue

        stage_module_offload(module, pin_memory=pin_memory)


def subgraph_onload_modules(
    modules: dict[str, torch.nn.Module],
) -> dict[str, dict]:
    """Onload modules, consuming tensors staged in CPU memory."""
    from compressed_tensors.offload import get_cache_init_kwargs

    offload_kwargs = {}
    for name, module in modules.items():
        if isinstance(module._parameters, OffloadCache):
            init_kwargs = get_cache_init_kwargs(module)
            offload_kwargs[name] = init_kwargs

            remove_module_offload(module, onload_tensors=True)
        else:
            pass
            # we should consider raising warnings, but observers will
            # clog the output with warnings, so we will skip for now
    return offload_kwargs


def subgraph_offload_modules(
    modules: dict[str, torch.nn.Module],
    offload_kwargs: dict[str, dict],
):
    """
    Offload a list of modules, using the provided device map and kwargs.
    """
    from compressed_tensors.offload import get_cache_init_kwargs

    for name, module in modules.items():
        if name in offload_kwargs:
            offload_module(module, **offload_kwargs[name])
        else:
            # we should consider raising warnings, but observers will
            # clog the output with warnings, so we will skip for now
            module_offload_kwargs = get_cache_init_kwargs(module)
            offload_module(module, **module_offload_kwargs)
