# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Container
from copy import deepcopy
from functools import partial
from typing import Any, Optional, TypeVar

import torch
import torch.distributed as dist
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.module import offload_module, remove_module_offload
from compressed_tensors.offload.utils import get_module_device, get_module_sizes
from compressed_tensors.utils import getattr_chain
from compressed_tensors.utils.binary_search import SearchFailureError, max_binary_search
from loguru import logger
from transformers import PreTrainedModel


__all__ = [
    "offload_model",
    "dispatch_with_map",
    "dispatch_model",
    "remove_dispatch",
    "get_device_memory",
    "DeviceMap",
]

ModelType = TypeVar("ModelType", bound=torch.nn.Module)
DeviceMap = dict[str, tuple[torch.device | None, torch.device | str | None]]


def offload_model(
    model: ModelType,
    onload_device: torch.device | str,
    offload_device: Any = None,
) -> ModelType:
    """
    Modify the dispatch of a model to onload to the provided `onload_device`. Existing
    offloaded tensors will not be modified. If a module is not offloaded, it will be
    offloaded to the provided `offload_device`.

    :param model: model to dispatch
    :param onload_device: device to move weights to during forward pass
    :param offload_device: device to offload weights to, if not already offloaded
    :return: dispatched model
    """
    if offload_device is not None:
        logger.warning(
            "`offload_model` now keeps the same offload device that model was loaded "
            "on. Please specify offload by loading the model on its offload device(s)"
        )

    # offload modules in place
    for module in model.modules():
        if isinstance(module._parameters, OffloadCache):
            module._parameters.onload_device = onload_device
            module._buffers.onload_device = onload_device
        else:
            offload_device = get_module_device(module, torch.device("cpu"))
            offload_module(module, onload_device, offload_device)

    return model


def dispatch_with_map(
    model: torch.nn.Module,
    device_map: DeviceMap,
    offload_dir: Optional[str] = None,
):
    """
    Dispatch a model according to the provided device map

    :param model: model to dispatch
    :param device_map: device map specifying the onload and offload of each module
    :param offload_dir: optional directory for disk offloading
    """
    for name, (onload_device, offload_device) in device_map.items():
        module = model.get_submodule(name)

        if offload_device == "disk":
            offload_module(
                module, onload_device, offload_device, offload_dir=offload_dir
            )

        elif offload_device is not None:
            offload_module(module, onload_device, offload_device)


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

    # estimate activations memory requirement
    if extra_memory is None:
        if isinstance(model, PreTrainedModel):
            extra_memory = (
                1  # batch_size
                * 2048  # seq_len
                * getattr_chain(model, "config.intermediate_size", 256)
                * getattr(model, "dtype", torch.bfloat16).itemsize
                + 1e9  # extra memory for fragmentation, kv cache, ect.
            )
        else:
            extra_memory = 0

    # collect devices
    if device_memory is None:
        device_memory: dict[torch.device, int] = get_device_memory()
    if len(device_memory) <= 0:
        raise MemoryError("Did not find any devices to dispatch model to")

    # collect module sizes
    sizes = get_module_sizes(model, no_split_modules)
    if len(sizes) <= 0:
        raise ValueError("Model does not have any modules")

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
    finally:
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
    Get the total memory of all available cuda devices

    :return: mapping from torch device to total memory
    """
    if not torch.cuda.is_available():
        return dict()

    if dist.is_available() and dist.is_initialized():
        logger.info("Detected distributed context. Dispatching to local rank gpu")
        device_memory = torch.cuda.get_device_properties(dist.get_rank()).total_memory
        return {torch.device("cuda"): device_memory}

    return {
        # TODO: extend to xpu, ect.
        torch.device(f"cuda:{idx}"): torch.cuda.get_device_properties(idx).total_memory
        for idx in range(torch.cuda.device_count())
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
