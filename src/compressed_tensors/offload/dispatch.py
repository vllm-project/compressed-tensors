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
from functools import partial
from typing import Literal, Optional, TypeVar

import torch
from compressed_tensors.offload.module import offload_module, remove_module_offload
from compressed_tensors.offload.utils import get_module_sizes
from compressed_tensors.utils import getattr_chain
from compressed_tensors.utils.binary_search import SearchFailureError, max_binary_search
from loguru import logger
from transformers import PreTrainedModel


__all__ = [
    "offload_model",
    "dispatch_model",
    "remove_dispatch",
    "get_device_memory",
]

ModelType = TypeVar("ModelType", bound=torch.nn.Module)


def offload_model(
    model: ModelType,
    onload_device: torch.device | str,
    offload_device: torch.device | str | Literal["disk"] = torch.device("cpu"),
) -> ModelType:
    """
    Offload a model to the `offload_device`. During forward passes, model weights will
    be onloaded to the `onload_device`

    :param model: model to dispatch
    :param onload_device: device to move weights to during forward pass
    :param offload_device: device to offload weights to
    :return: dispatched model
    """
    # remove any previous dispatches
    remove_dispatch(model)

    # offload modules in place
    for module in model.modules():
        offload_module(module, onload_device, offload_device)

    return model


def dispatch_model(
    model: ModelType,
    device_memory: Optional[dict[torch.device, int]] = None,
    extra_memory: Optional[int] = None,
    no_split_modules: Optional[Container[str]] = None,
) -> ModelType:
    """
    Dispatch a model for autoregressive generation. This means that modules are
    dispatched evenly across available devices and kept onloaded if possible. If
    onloading the entire model is not possible, some modules may be offloaded.

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
    # remove previous dispatches
    remove_dispatch(model)

    # infer no_split_modules
    if no_split_modules is None:
        no_split_modules = getattr(model, "_no_split_modules", tuple())

    # estimate activations memory requirement
    if extra_memory is None:
        if isinstance(model, PreTrainedModel):
            extra_memory = (
                1  # batch_size
                * 2048  # seq_len
                * getattr_chain(model, "_config.hidden_dim", 256)
                * getattr(model, "dtype", torch.bfloat16).itemsize
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
        while largest_offloaded_module > device_memory[last_device] - extra_memory:
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

        extra_memory = 0
        logger.warning("Forced to offload modules due to insufficient gpu resources")

    # dispatch
    finally:
        assert len(dispatch) == len(sizes)
        for module, onload, offload in dispatch:
            for submodule in module.modules():
                offload_module(submodule, onload, offload)

        logger.debug(f"Dispatched model with {extra_memory} bytes of extra memory")
        return model


def get_device_memory() -> dict[torch.device, int]:
    """
    Get the total memory of all available cuda devices

    :return: list of device memory dataclasses
    """
    if not torch.cuda.is_available():
        return dict()

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
