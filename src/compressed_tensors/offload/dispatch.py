# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Container
from copy import deepcopy
from functools import partial
from typing import Any, Literal, Optional, TypeVar

import torch
import torch.distributed as dist
from compressed_tensors.distributed import is_distributed
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.module import offload_module, remove_module_offload
from compressed_tensors.offload.utils import (
    get_module_sizes,
    module_size,
)
from compressed_tensors.utils import getattr_chain
from compressed_tensors.utils.binary_search import SearchFailureError, max_binary_search
from compressed_tensors.utils.helpers import deprecated
from loguru import logger
from tqdm import tqdm
from transformers import PreTrainedModel


__all__ = [
    "set_onload_device",
    "offload_model",
    "dispatch_with_map",
    "get_device_map",
    "dispatch_model",
    "remove_dispatch",
    "get_device_memory",
    "DeviceMap",
]

ModelType = TypeVar("ModelType", bound=torch.nn.Module)
DeviceMap = dict[str, tuple[torch.device | None, torch.device | str | None]]


def set_onload_device(
    model: ModelType,
    onload_device: torch.device | str,
) -> ModelType:
    """
    Modify the dispatch of a model to onload to the provided `onload_device`. Existing
    offloaded tensors will not be modified. If a module is not already offloaded, it
    will be offloaded to the offload device of the first offloaded module in its
    subtree, or to the default offload device (cpu) if none of its modules are
    offloaded.

    :param model: model to dispatch
    :param onload_device: device to move weights to during forward pass
    :return: dispatched model
    """
    _set_onload_device_recursively(model, onload_device)
    return model


def _set_onload_device_recursively(
    module: torch.nn.Module,
    onload_device: torch.device | str,
) -> tuple[torch.device | Literal["disk"], torch.nn.Module, bool]:
    """
    Recursively set the onload device of `module` and all of its modules to
    `onload_device`, offloading any module which is not already offloaded.

    The offload device of a module is resolved as follows:
    1. If the module is offloaded, its onload device is set to `onload_device` and its
       existing offload device is returned together with the module and `True`.
    2. Otherwise the module is offloaded to the offload device of the first child which
       reports an offloaded module (that child's device). If none of the children are
       offloaded, the default offload device (cpu) is used and `False` is returned.

    When the resolved offload device is ``"disk"``, the offload directory is read from
    the offloaded module's cache (the same value returned by
    ``get_cache_init_kwargs``) so the module is offloaded to the same location.

    Resolving devices this way reads an offloaded child's offload device directly from
    its cache rather than through its tensors, so the child's weights are never
    onloaded. This keeps the operation free of unnecessary device movement.

    :param module: module to set onload device for
    :param onload_device: device to move weights to during forward pass
    :return: `(offload_device, offloaded_module, found_offloaded)`, where
        `offload_device` is the resolved offload device of `module`,
        `offloaded_module` is the offloaded module whose device/directory it was
        resolved from (or `module` itself if already offloaded), and
        `found_offloaded` is whether that device was resolved from an offloaded
        module in `module`'s subtree
    """
    if isinstance(module._parameters, OffloadCache):
        offload_device = module._parameters.offload_device
        module._parameters.onload_device = onload_device
        module._buffers.onload_device = onload_device
        return offload_device, module, True

    offload_device: torch.device | Literal["disk"] = torch.device("cpu")
    offloaded_module = None
    found_offloaded = False
    for child in module.children():
        child_offload_device, child_offloaded_module, child_found = (
            _set_onload_device_recursively(child, onload_device)
        )
        if not found_offloaded and child_found:
            offload_device = child_offload_device
            offloaded_module = child_offloaded_module
            found_offloaded = True

    offload_kwargs: dict[str, object] = {
        "onload_device": onload_device,
        "offload_device": offload_device,
    }
    if offload_device == "disk":
        # disk offloading requires an offload directory, which is read from the
        # offloaded module's cache via `get_cache_init_kwargs`
        from compressed_tensors.offload import get_cache_init_kwargs

        cache_kwargs = get_cache_init_kwargs(offloaded_module)
        offload_kwargs["offload_device"] = cache_kwargs["offload_device"]
        offload_kwargs["offload_dir"] = cache_kwargs["offload_dir"]

    offload_module(module, **offload_kwargs)
    return offload_device, offloaded_module, found_offloaded


@deprecated("set_onload_device")
def offload_model(
    model: ModelType,
    onload_device: torch.device | str,
    offload_device: Any = None,
) -> ModelType:
    """
    .. deprecated::
        Use :func:`set_onload_device` instead.
    """
    return set_onload_device(model, onload_device)


def dispatch_with_map(
    model: torch.nn.Module,
    device_map: DeviceMap,
    offload_dir: Optional[str] = None,
    show_progress: bool = True,
):
    """
    Dispatch a model according to the provided device map

    :param model: model to dispatch
    :param device_map: device map specifying the onload and offload of each module
    :param offload_dir: optional directory for disk offloading
    :param show_progress: show tqdm progress
    """
    for name, (onload_device, offload_device) in tqdm(
        list(device_map.items()),
        desc="Dispatching model",
        disable=(not show_progress),
        position=(dist.get_rank() if is_distributed() else 0),
    ):
        try:
            module = model.get_submodule(name)
        except AttributeError:
            # The device map is authored from the source rank's view of the
            # module tree. On other ranks, sharded structures -- e.g. an
            # nn.ModuleList of routed MoE experts where slots the rank does
            # not own are None placeholders -- legitimately lack some of
            # these submodules. Skip map entries with no local module rather
            # than crashing the dispatch.
            logger.debug(f"Skipping '{name}' from device map: not present locally")
            continue

        if offload_device == "disk":
            offload_module(
                module, onload_device, offload_device, offload_dir=offload_dir
            )

        elif offload_device is not None:
            offload_module(module, onload_device, offload_device)


def get_device_map(
    model: torch.nn.Module, default_device: torch.device = torch.device("cpu")
) -> DeviceMap:
    """
    Get the device map of a CT-offloaded model

    :param: model: model to get device map of
    :param default_device: the default onload/offload device
        when module has no parameters
    :return: device map specifying the onload and offload device of all modules
    """
    from compressed_tensors.offload import get_execution_device, get_offloaded_device

    return {
        name: (
            get_execution_device(module, default_device),
            get_offloaded_device(module, default_device),
        )
        for name, module in model.named_modules(remove_duplicate=False)
    }


def dispatch_model(
    model: ModelType,
    device_memory: dict[torch.device, int] | None = None,
    extra_memory: int | None = None,
    no_split_modules: Container[str] | None = None,
) -> ModelType:
    """
    Dispatch a model for autoregressive generation. This means that modules are
    dispatched evenly across available devices and kept onloaded if possible. If
    onloading the entire model is not possible, some modules may be offloaded. Any
    existing offloads will be removed.

    Disclaimers:
    * Optimal runtime assumes that modules are called in order of `model.modules()`

    :param model: model to dispatch
    :param device_memory: optional dictionary mapping torch device to available memory.
        If none is provided, all available devices will be used
    :param extra_memory: the amount of memory to be reserved for activations
    :param no_split_modules: names of module classes which should not be split
        across multiple devices
    :return: dispatched model
    """
    # infer no_split_modules
    if no_split_modules is None:
        no_split_modules = getattr(model, "_no_split_modules", tuple())

    # collect devices
    if device_memory is None:
        device_memory: dict[torch.device, int] = get_device_memory()
    if len(device_memory) <= 0:
        raise MemoryError("Did not find any devices to dispatch model to")

    # collect module sizes
    sizes = get_module_sizes(model, no_split_modules)
    if len(sizes) <= 0:
        raise ValueError("Model does not have any modules")

    # estimate memory requirement
    if extra_memory is None:
        # fragmentation, kv cache, embeddings, ect.
        extra_memory = max(module_size(model) * 0.05, 1e9)

        # activations
        if isinstance(model, PreTrainedModel):
            extra_memory += (
                1  # batch_size
                * 2048  # seq_len
                * getattr_chain(model, "config.intermediate_size", 256)
                * getattr(model, "dtype", torch.bfloat16).itemsize
            )

    # search for the best dispatch which maximizes extra memory across devices
    try:
        max_extra_memory = min(device_memory.values())
        extra_memory, (dispatch, _) = max_binary_search(
            fn=partial(_get_greedy_dispatch, sizes, device_memory),
            cond=(lambda result: len(result[0]) == len(sizes)),
            start=extra_memory,
            end=max_extra_memory,
        )

    # fallback: create a cpu dispatch
    except SearchFailureError:
        dispatch, device_memory = _get_greedy_dispatch(
            sizes, device_memory, extra_memory
        )
        assert len(dispatch) < len(sizes)

        last_device = dispatch[-1][1] if len(dispatch) else list(device_memory)[0]
        sizes_dict = {module: size for module, size in sizes}
        largest_offloaded_module = max(size for _, size in sizes[len(dispatch) :])

        # pop off modules until all offloaded modules can fit in last device
        while largest_offloaded_module + extra_memory > device_memory[last_device]:
            if len(dispatch) <= 0:
                raise ValueError(
                    f"Cannot fit no_split module of size {largest_offloaded_module} "
                    f"bytes into any device: {device_memory}"
                )

            module, last_device, _ = dispatch.pop(-1)
            device_memory[last_device] += sizes_dict[module]
            largest_offloaded_module = max(largest_offloaded_module, sizes_dict[module])

        # fill dispatch back with cpu offloading
        for module, _ in list(sizes[len(dispatch) :]):
            dispatch.append((module, last_device, "cpu"))

        logger.warning("Forced to offload modules due to insufficient gpu resources")

    # dispatch
    assert len(dispatch) == len(sizes)

    dispatch_dict = {
        submodule: (onload, offload)
        for module, onload, offload in dispatch
        for submodule in module.modules()
    }

    for module in model.modules():
        remove_module_offload(module, onload_tensors=True)
        if module in dispatch_dict:
            onload, offload = dispatch_dict[module]
            offload_module(module, onload, offload)

    logger.debug(f"Dispatched model with {extra_memory} bytes of extra memory")
    return model


def get_device_memory() -> dict[torch.device, int]:
    """
    Get the total memory of all available accelerator devices. Returns accelerator
    device memory when available, otherwise falls back to CPU with system RAM.

    :return: mapping from torch device to total memory
    """
    if not torch.accelerator.is_available():
        import os

        total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return {torch.device("cpu"): total_ram}

    accel_type = torch.accelerator.current_accelerator().type

    if dist.is_available() and dist.is_initialized():
        logger.info("Detected distributed context. Dispatching to local rank gpu")
        device_memory = torch.accelerator.get_memory_info(
            torch.accelerator.current_device_index()
        )[1]
        return {torch.device(accel_type): device_memory}

    return {
        torch.device(accel_type, idx): torch.accelerator.get_memory_info(idx)[1]
        for idx in range(torch.accelerator.device_count())
    }


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


def _get_greedy_dispatch(
    sizes: list[tuple[torch.nn.Module, int]],
    device_memory: dict[torch.device, int],
    extra_memory: int = 0,
) -> tuple[
    list[tuple[torch.nn.Module, torch.device, torch.device]], dict[torch.device, int]
]:
    dispatch = list()
    memory_remaining = deepcopy(device_memory)

    device_index = 0
    devices = list(memory_remaining.keys())

    if len(devices) <= 0:
        raise ValueError()

    for module, size in sizes:
        while True:
            if device_index >= len(devices):
                return dispatch, memory_remaining

            device = devices[device_index]
            if size > memory_remaining[device] - extra_memory:
                device_index += 1
                continue

            dispatch.append((module, device, device))
            memory_remaining[device] -= size
            break

    return dispatch, memory_remaining
