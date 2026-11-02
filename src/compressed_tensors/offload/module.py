# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import socket
from dataclasses import dataclass
from functools import wraps

import torch
import torch.distributed as dist
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.cache.dist_disk import DistributedDiskCache
from compressed_tensors.offload.cache.disk_utils import disk_load_context
from compressed_tensors.offload.shared_stage import (
    attach_staged_tensor,
    register_shared_storage,
    share_staged_tensor,
    unregister_shared_storage,
)
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
    """
    Stage offloaded module tensors in CPU memory for a later onload.

    Reads are grouped under `disk_load_context`, so tensors from the same
    safetensors shard share one open instead of re-parsing its header per read.
    """
    with disk_load_context():
        for name, module in modules.items():
            if not isinstance(module._parameters, OffloadCache):
                # we should consider raising warnings, but observers will
                # clog the output with warnings, so we will skip for now
                continue

            stage_module_offload(module, pin_memory=pin_memory)


@dataclass
class SharedDiskStageResult:
    descriptors: dict[tuple[str, str, str], tuple]
    error: Exception | None = None


def prepare_shared_disk_stage(
    modules: dict[str, torch.nn.Module],
    pin_memory: bool = True,
    source_rank: int = 0,
) -> SharedDiskStageResult:
    """Prefetch disk weights on the source rank without running collectives.

    Other cache types keep their existing rank-local staging behavior. Errors are
    returned so all ranks can receive them at the main-thread synchronization point.
    """
    descriptors = {}
    try:
        with disk_load_context():
            for module_name, module in modules.items():
                cache = module._parameters
                if not isinstance(cache, OffloadCache):
                    continue
                if not isinstance(cache, DistributedDiskCache):
                    stage_module_offload(module, pin_memory=pin_memory)
                    continue
                if dist.get_rank() != source_rank:
                    continue

                # The source makes one shared copy. The CUDA registrations happen
                # after all ranks attach, on the main thread before onloading.
                stage_module_offload(module, pin_memory=False)
                for attr in ("_parameters", "_buffers"):
                    staged_cache = getattr(module, attr)
                    for tensor_name, tensor in staged_cache.offloaded_values.items():
                        if tensor is not None:
                            descriptors[(module_name, attr, tensor_name)] = (
                                share_staged_tensor(tensor)
                            )
    except Exception as error:
        return SharedDiskStageResult({}, error)
    return SharedDiskStageResult(descriptors)


def _release_shared_stage(cache: OffloadCache) -> None:
    pointers = getattr(cache, "_shared_stage_pointers", None)
    if pointers:
        for pointer in tuple(pointers):
            unregister_shared_storage(pointer)
            pointers.remove(pointer)


def finish_shared_disk_stage(
    modules: dict[str, torch.nn.Module],
    prepared: SharedDiskStageResult,
    pin_memory: bool = True,
    source_rank: int = 0,
) -> None:
    """Attach every rank to the staged storage and pin its local CUDA mapping.

    This function must run on the main thread and in the same order on every rank;
    staging itself may run in a background thread while modifiers use collectives.
    """
    if not dist.is_initialized():
        raise RuntimeError("Shared disk staging requires an initialized process group")

    states = [None] * dist.get_world_size()
    dist.all_gather_object(
        states, (socket.gethostname(), repr(prepared.error) if prepared.error else None)
    )
    if len({hostname for hostname, _ in states}) != 1:
        raise RuntimeError("Shared disk staging requires all ranks on one host")
    errors = [error for _, error in states]
    if any(errors):
        if prepared.error is not None:
            raise prepared.error
        raise RuntimeError(f"Shared disk staging failed on another rank: {errors}")

    is_source = dist.get_rank() == source_rank
    payload = [prepared.descriptors if is_source else None]
    dist.broadcast_object_list(payload, src=source_rank)
    descriptors = payload[0]
    memo = {}
    local_error = None
    try:
        for module_name, module in modules.items():
            if not isinstance(module._parameters, DistributedDiskCache):
                continue
            for attr in ("_parameters", "_buffers"):
                cache = getattr(module, attr)
                pointers = cache._shared_stage_pointers = set()
                staged_values = {}
                for tensor_name, tensor in cache.offloaded_values.items():
                    if tensor is None:
                        staged_values[tensor_name] = None
                        continue
                    descriptor = descriptors.get((module_name, attr, tensor_name))
                    if descriptor is None:
                        # A rank may own a module absent on the source rank.
                        staged = (
                            tensor
                            if is_source
                            else cache.stage(tensor, pin_memory=pin_memory)
                        )
                    else:
                        staged = (
                            tensor
                            if is_source
                            else attach_staged_tensor(tensor, descriptor, memo)
                        )
                        if (
                            pin_memory
                            and staged is not None
                            and torch.device(cache.onload_device).type == "cuda"
                        ):
                            pointer = staged.untyped_storage().data_ptr()
                            if pointer not in pointers:
                                registered = register_shared_storage(staged)
                                if registered is not None:
                                    pointers.add(registered)
                    staged_values[tensor_name] = staged
                cache.offloaded_values = staged_values
                cache.is_staged = True

            if hasattr(module, "_original_forward_func"):
                module.forward = module._original_forward_func.__get__(module)
                del module._original_forward_func
    except Exception as error:
        local_error = error

    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, repr(local_error) if local_error else None)
    if any(errors):
        for module in modules.values():
            for attr in ("_parameters", "_buffers"):
                cache = getattr(module, attr)
                if isinstance(cache, DistributedDiskCache):
                    _release_shared_stage(cache)
        if local_error is not None:
            raise local_error
        raise RuntimeError(f"Shared disk staging failed on another rank: {errors}")


def subgraph_onload_modules(
    modules: dict[str, torch.nn.Module],
) -> dict[str, dict]:
    """Onload modules, consuming tensors staged in CPU memory."""
    from compressed_tensors.offload import get_cache_init_kwargs

    offload_kwargs = {}
    # tensors that were not staged are read from disk here, grouped like staging
    with disk_load_context():
        for name, module in modules.items():
            if isinstance(module._parameters, OffloadCache):
                init_kwargs = get_cache_init_kwargs(module)
                offload_kwargs[name] = init_kwargs

                parameter_cache, buffer_cache = module._parameters, module._buffers
                try:
                    remove_module_offload(module, onload_tensors=True)
                finally:
                    _release_shared_stage(parameter_cache)
                    _release_shared_stage(buffer_cache)
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
