# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Container
from copy import deepcopy
from functools import partial
from typing import Any, Optional, TypeVar

import torch
import torch.distributed as dist
from compressed_tensors.distributed import (
    get_source_rank,
    is_distributed,
    is_source_process,
)
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.module import (
    install_offload_forward,
    offload_module,
    remove_module_offload,
)
from compressed_tensors.offload.utils import (
    get_module_device,
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
    will be offloaded to its current device.

    :param model: model to dispatch
    :param onload_device: device to move weights to during forward pass
    :return: dispatched model
    """
    from compressed_tensors.utils.module import get_direct_state_dict

    for name, module in model.named_modules():
        if isinstance(module._parameters, OffloadCache):
            module._parameters.onload_device = onload_device
            module._buffers.onload_device = onload_device
        else:
            #tensor = next(get_direct_state_dict(module).values(), None)
            offload_device = "disk"# tensor.device if tensor is not None else torch.device("cpu")
            offload_module(module, onload_device, offload_device, offload_dir="/data/kylesayrs/hub/offload_folder")

    return model


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

    When running distributed with cpu and/or disk offloading, the ranks must
    exchange offload metadata (shared memory file handles for cpu, disk file
    handles for disk) so that every rank points at the same offloaded data.
    Rather than synchronizing per-tensor -- which incurs tens of thousands of
    fixed-cost rank synchronizations when loading a large model -- the metadata
    for the entire model is exchanged in a single broadcast. See
    `_dispatch_with_map_batched`.

    :param model: model to dispatch
    :param device_map: device map specifying the onload and offload of each module
    :param offload_dir: optional directory for disk offloading
    :param show_progress: show tqdm progress
    """
    # Batching the metadata exchange only helps in the distributed case. When
    # distributed, cpu/disk offloads (whose metadata is broadcast as python
    # objects) are batched into a single exchange, while accelerator offloads
    # (which broadcast raw tensor data) keep their per-tensor path.
    if is_distributed():
        return _dispatch_with_map_batched(
            model, device_map, offload_dir, show_progress
        )

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


def _is_batchable_offload(offload_device: torch.device | str) -> bool:
    """
    Whether an offload device's metadata can be batched across ranks. True for
    disk and cpu offloads; False for accelerator offloads, which broadcast raw
    tensor data rather than python-object metadata.
    """
    if offload_device == "disk":
        return True
    return torch.device(offload_device).type == "cpu"


def _iter_local_dispatch(model: torch.nn.Module, device_map: DeviceMap):
    """
    Yield `(name, module, onload_device, offload_device)` for each map entry that
    should be offloaded on this rank. Entries with no offload, no local module
    (see the sharded-MoE note in `dispatch_with_map`), or a module that is already
    offloaded (e.g. a tied module reached under a second name) are skipped.

    Both the source and non-source ranks iterate this identically, so the single
    batched broadcast stays consistent even when ranks own different submodules.
    """
    for name, (onload_device, offload_device) in device_map.items():
        if offload_device is None:
            continue

        try:
            module = model.get_submodule(name)
        except AttributeError:
            logger.debug(f"Skipping '{name}' from device map: not present locally")
            continue

        if isinstance(module._parameters, OffloadCache):
            continue

        yield name, module, onload_device, offload_device


def _dispatch_with_map_batched(
    model: torch.nn.Module,
    device_map: DeviceMap,
    offload_dir: Optional[str],
    show_progress: bool,
):
    """
    Distributed dispatch which exchanges all offload metadata in a single
    broadcast rather than synchronizing per-tensor.

    The source rank offloads every module locally (writing shared memory / disk
    files without any rank synchronization) while collecting the metadata needed
    to reconstruct each tensor on other ranks. All of that metadata is broadcast
    at once. Non-source ranks then reconstruct their offloaded tensors from the
    received metadata. This collapses the tens of thousands of per-tensor
    `broadcast_object_list` + `barrier` pairs into a single exchange.

    Accelerator offloads are not batched: they broadcast raw tensor data (to
    replicate weights across devices) rather than python-object metadata, so
    they keep their existing per-tensor `offload_module` path. Every rank walks
    the device map in the same order, so the interleaved accelerator broadcasts
    stay collective-consistent; the batched cpu/disk metadata is then exchanged
    in a single broadcast afterwards.

    Metadata is keyed by `(module_name, attr, tensor_name)` rather than by
    position so that ranks which own different submodules (e.g. sharded MoE
    experts) can each look up only the entries relevant to them.
    """
    position = dist.get_rank()
    source = is_source_process()

    # metadata collected on the source; cpu/disk modules deferred on other ranks
    metadata: dict = {}
    deferred: list[tuple[str, torch.nn.Module, Any, Any]] = []

    # tied weights (e.g. embed_tokens/lm_head) resolve to the same storage. Detect
    # ties up front -- while the model is still intact and every original tensor is
    # alive -- so that a shared tensor is offloaded once and its aliases reuse it.
    # This preserves ties across ranks (avoiding a duplicate copy of the large
    # embedding) and, critically, avoids re-sharing an already shared cpu storage:
    # `_share_filename_cpu_` rotates (and unlinks) the previously captured
    # shared-memory file each time it is called on the same storage. Ties MUST be
    # keyed on the original storage: the offloaded storage's `data_ptr` is recycled
    # across distinct tensors (freed pageable copies), so keying on it produces
    # false aliases. Only computed on the source, which holds the real weights.
    aliases = _build_tie_groups(model, device_map) if source else {}
    canonical_offloaded: dict[tuple, torch.Tensor | None] = {}

    for name, module, onload, offload in tqdm(
        list(_iter_local_dispatch(model, device_map)),
        desc="Dispatching model (batched)",
        disable=(not show_progress),
        position=position,
    ):
        # accelerator offload: replicate weights via the per-tensor data broadcast
        if not _is_batchable_offload(offload):
            offload_module(module, onload, offload)
            continue

        # cpu / disk offload: batch the metadata exchange
        if source:
            _offload_module_source(
                module, onload, offload, offload_dir, name, metadata,
                aliases, canonical_offloaded,
            )
        else:
            deferred.append((name, module, onload, offload))

    # single metadata exchange for all cpu/disk offloads in the whole model
    broadcast_obj = [metadata] if source else [None]
    dist.broadcast_object_list(broadcast_obj, src=get_source_rank())

    if not source:
        metadata = broadcast_obj[0]
        reconstructed: dict[tuple, torch.Tensor | None] = {}
        for name, module, onload, offload in deferred:
            _offload_module_replica(
                module, onload, offload, offload_dir, name, metadata, reconstructed
            )

    # ensure the source rank keeps shared storages / files alive until every
    # replica has mapped them (mirrors the per-tensor `offload` barrier)
    dist.barrier()


# sentinel marking a metadata entry as an alias of an already-offloaded tensor
_OFFLOAD_ALIAS = "__ct_offload_alias__"


def _cache_kwargs(offload_device, offload_dir: Optional[str]) -> dict:
    return {"offload_dir": offload_dir} if offload_device == "disk" else {}


def _offload_tag(offload_device) -> str:
    """Normalize a batchable offload device to a tie-grouping tag."""
    return "disk" if offload_device == "disk" else "cpu"


def _original_storage_id(tensor: torch.Tensor | None) -> Optional[int]:
    """
    Storage identity of an *original* (not yet offloaded) tensor, used to detect
    tied/shared weights. Returns None for tensors without real backing storage
    (None, meta, or empty tensors), which must not be deduplicated.
    """
    if tensor is None or tensor.is_meta:
        return None
    storage_id = tensor.untyped_storage().data_ptr()
    return storage_id if storage_id != 0 else None


def _build_tie_groups(model: torch.nn.Module, device_map: DeviceMap) -> dict:
    """
    Identify tied tensors (those sharing storage) among the batchable offloads,
    keyed by the original tensor storage while the model is fully materialized.

    :return: mapping from an alias key `(module_name, attr, tensor_name)` to the
        canonical key it should reuse. Only alias (non-first) keys are present.
        Ties are only grouped within the same offload type (cpu vs disk), since
        aliases reuse the canonical rank's offloaded tensor, which must belong to
        the same cache kind.
    """
    canonical: dict[tuple, tuple] = {}
    aliases: dict[tuple, tuple] = {}

    for name, module, _onload, offload in _iter_local_dispatch(model, device_map):
        if not _is_batchable_offload(offload):
            continue
        tag = _offload_tag(offload)
        for attr in ("_parameters", "_buffers"):
            for tensor_name, tensor in getattr(module, attr).items():
                storage_id = _original_storage_id(tensor)
                if storage_id is None:
                    continue
                group = (tag, storage_id)
                key = (name, attr, tensor_name)
                if group in canonical:
                    aliases[key] = canonical[group]
                else:
                    canonical[group] = key

    return aliases


def _offload_module_source(
    module: torch.nn.Module,
    onload_device,
    offload_device,
    offload_dir: Optional[str],
    name: str,
    metadata: dict,
    aliases: dict,
    canonical_offloaded: dict,
):
    """
    Offload a module on the source rank without synchronizing, recording the
    metadata needed for other ranks to reconstruct each offloaded tensor.

    Tensors whose storage is tied to an already-offloaded tensor (per
    `aliases`) reuse the first offload and record an alias rather than being
    re-offloaded. The canonical tensor is always offloaded before its aliases,
    since ties are keyed on first occurrence in device-map order.
    """
    cache_cls = OffloadCache.cls_from_device(offload_device)
    kwargs = _cache_kwargs(offload_device, offload_dir)

    for attr in ("_parameters", "_buffers"):
        mapping = getattr(module, attr)
        cache = cache_cls(
            onload_device=onload_device, offload_device=offload_device, **kwargs
        )
        offloaded_values = {}
        for tensor_name, tensor in mapping.items():
            key = (name, attr, tensor_name)
            canonical_key = aliases.get(key)

            if canonical_key is not None:
                # tied weight: reuse the canonical offload and record an alias
                offloaded_values[tensor_name] = canonical_offloaded[canonical_key]
                metadata[key] = (_OFFLOAD_ALIAS, canonical_key)
            else:
                offloaded = cache.offload_local(tensor)
                offloaded_values[tensor_name] = offloaded
                metadata[key] = cache.get_offload_meta(offloaded)
                canonical_offloaded[key] = offloaded

        cache.offloaded_values = offloaded_values
        setattr(module, attr, cache)

    install_offload_forward(module)


def _offload_module_replica(
    module: torch.nn.Module,
    onload_device,
    offload_device,
    offload_dir: Optional[str],
    name: str,
    metadata: dict,
    reconstructed: dict,
):
    """
    Reconstruct a module's offloaded tensors on a non-source rank from the
    metadata broadcast by the source rank. Alias entries reuse the tensor already
    reconstructed for their canonical key, preserving tied weights across ranks.
    """
    cache_cls = OffloadCache.cls_from_device(offload_device)
    kwargs = _cache_kwargs(offload_device, offload_dir)

    for attr in ("_parameters", "_buffers"):
        mapping = getattr(module, attr)
        cache = cache_cls(
            onload_device=onload_device, offload_device=offload_device, **kwargs
        )
        offloaded_values = {}
        for tensor_name, tensor in mapping.items():
            key = (name, attr, tensor_name)
            meta = metadata.get(key)

            if _is_alias(meta):
                # tied weight: reuse the canonical tensor reconstructed earlier.
                # The source assigns the canonical to the first occurrence, and
                # both ranks walk the device map in the same order, so it is
                # always reconstructed before its aliases.
                offloaded_values[tensor_name] = reconstructed[meta[1]]
            else:
                offloaded = cache.recv_offload(tensor, meta)
                offloaded_values[tensor_name] = offloaded
                reconstructed[key] = offloaded

        cache.offloaded_values = offloaded_values
        setattr(module, attr, cache)

    install_offload_forward(module)


def _is_alias(meta) -> bool:
    return (
        isinstance(meta, tuple) and len(meta) == 2 and meta[0] == _OFFLOAD_ALIAS
    )


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
