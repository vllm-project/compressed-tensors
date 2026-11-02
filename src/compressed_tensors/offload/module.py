# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import errno
import os
import socket
import tempfile
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
from compressed_tensors.offload.utils import _pin_memory, send_tensors


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
    if pin_memory and dist.is_available() and dist.is_initialized() and modules:
        # The sequential pipeline prefetches here on a worker thread. Prepare
        # shared disk storage without collectives, then attach/register on the
        # main thread in subgraph_onload_modules. Rank 0 is stable even when a
        # modifier temporarily changes the distributed offload source rank.
        first_module = next(iter(modules.values()))
        first_module._shared_disk_stage = prepare_shared_disk_stage(modules)
        return

    with disk_load_context():
        for name, module in modules.items():
            if not isinstance(module._parameters, OffloadCache):
                # we should consider raising warnings, but observers will
                # clog the output with warnings, so we will skip for now
                continue

            stage_module_offload(module, pin_memory=pin_memory)


def prepare_shared_disk_stage(
    modules: dict[str, torch.nn.Module],
) -> dict:
    """Prefetch disk weights on the source rank without running collectives.

    Other cache types keep their existing rank-local staging behavior. Errors are
    returned so all ranks can receive them at the main-thread synchronization point.
    """
    descriptors = {}
    shared_memo = {}
    temporary = None
    is_source = dist.get_rank() == 0
    try:
        with disk_load_context():
            for module_name, module in modules.items():
                cache = module._parameters
                if not isinstance(cache, OffloadCache):
                    continue
                if not isinstance(cache, DistributedDiskCache):
                    stage_module_offload(module, pin_memory=True)
                    continue
                if not is_source:
                    continue

                if not os.path.isdir("/dev/shm") or not os.access("/dev/shm", os.W_OK):
                    stage_module_offload(module, pin_memory=True)
                    continue

                # The source makes one shared copy. The CUDA registrations happen
                # after all ranks attach, on the main thread before onloading.
                stage_module_offload(module, pin_memory=False)
                for attr in ("_parameters", "_buffers"):
                    staged_cache = getattr(module, attr)
                    for tensor_name, tensor in staged_cache.offloaded_values.items():
                        if tensor is not None:
                            try:
                                if temporary is None:
                                    temporary = tempfile.TemporaryDirectory(
                                        prefix="ct-stage-", dir="/dev/shm"
                                    )
                                descriptor = share_staged_tensor(
                                    tensor, shared_memo, temporary.name
                                )
                            except OSError as error:
                                if error.errno not in (
                                    errno.ENOSPC,
                                    errno.EOPNOTSUPP,
                                    errno.ENOSYS,
                                ):
                                    raise
                                staged_cache.offloaded_values[tensor_name] = (
                                    _pin_memory(tensor)
                                    if torch.device(staged_cache.onload_device).type
                                    != "cpu"
                                    else tensor
                                )
                                continue
                            descriptors[(module_name, attr, tensor_name)] = descriptor
    except Exception as error:
        if temporary is not None:
            temporary.cleanup()
        return {"descriptors": {}, "temporary": None, "error": error}
    return {"descriptors": descriptors, "temporary": temporary, "error": None}


def finish_shared_disk_stage(
    modules: dict[str, torch.nn.Module],
    prepared: dict,
) -> dict[int, torch.UntypedStorage]:
    """Attach every rank to the staged storage and pin its local CUDA mapping.

    This function must run on the main thread and in the same order on every rank;
    staging itself may run in a background thread while modifiers use collectives.
    Returns the CUDA registrations to release after the blocking onload.
    """
    if not dist.is_initialized():
        raise RuntimeError("Shared disk staging requires an initialized process group")

    is_source = dist.get_rank() == 0
    # Holding each storage here keeps its mapping alive through unregister.
    registered = {}
    try:
        # This broadcast also waits for rank 0's background staging to finish.
        payload = [
            (
                socket.gethostname(),
                prepared["descriptors"],
                repr(prepared["error"]) if prepared["error"] else None,
            )
            if is_source
            else None
        ]
        dist.broadcast_object_list(payload, src=0)
        source_host, descriptors, source_error = payload[0]
        on_source_host = socket.gethostname() == source_host
        memo = {}
        local_error = prepared["error"]
        if source_error and local_error is None:
            local_error = RuntimeError(f"Source rank staging failed: {source_error}")
        if local_error is None:
            try:
                for module_name, module in modules.items():
                    if not isinstance(module._parameters, DistributedDiskCache):
                        continue
                    for attr in ("_parameters", "_buffers"):
                        cache = getattr(module, attr)
                        staged_values = {}
                        for tensor_name, tensor in cache.offloaded_values.items():
                            if tensor is None:
                                staged_values[tensor_name] = None
                                continue
                            descriptor = descriptors.get(
                                (module_name, attr, tensor_name)
                            )
                            can_attach = (
                                descriptor is not None
                                and on_source_host
                                and (
                                    is_source
                                    or descriptor[0] is None
                                    or os.path.exists(descriptor[0])
                                )
                            )
                            if can_attach:
                                staged = (
                                    tensor
                                    if is_source
                                    else attach_staged_tensor(tensor, descriptor, memo)
                                )
                                if torch.device(cache.onload_device).type == "cuda":
                                    pointer = staged.untyped_storage().data_ptr()
                                    if pointer not in registered:
                                        pointer = register_shared_storage(staged)
                                        if pointer is not None:
                                            registered[pointer] = (
                                                staged.untyped_storage()
                                            )
                            else:
                                # Other hosts and separate /dev/shm mounts stage
                                # their own weights from disk.
                                staged = (
                                    tensor
                                    if is_source
                                    else cache.stage(tensor, pin_memory=True)
                                )
                            staged_values[tensor_name] = staged
                        cache.offloaded_values = staged_values
                        cache.is_staged = True

                    if hasattr(module, "_original_forward_func"):
                        module.forward = module._original_forward_func.__get__(module)
                        del module._original_forward_func
            except Exception as error:
                local_error = error

        # Wait until every rank has attached before the source unlinks /dev/shm.
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, repr(local_error) if local_error else None)
        if any(errors):
            if local_error is not None:
                raise local_error
            raise RuntimeError(f"Shared disk staging failed on another rank: {errors}")
        return registered
    except BaseException:
        for pointer in registered:
            unregister_shared_storage(pointer)
        raise
    finally:
        if is_source and prepared["temporary"] is not None:
            prepared["temporary"].cleanup()


def subgraph_onload_modules(
    modules: dict[str, torch.nn.Module],
) -> dict[str, dict]:
    """Onload modules, consuming tensors staged in CPU memory."""
    from compressed_tensors.offload import get_cache_init_kwargs

    first_module = next(iter(modules.values()), None)
    prepared = getattr(first_module, "_shared_disk_stage", None)
    registered = {}
    try:
        if prepared is not None:
            try:
                registered = finish_shared_disk_stage(modules, prepared)
            finally:
                del first_module._shared_disk_stage

        offload_kwargs = {}
        # tensors that were not staged are read from disk here, grouped like staging
        with disk_load_context():
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
    finally:
        for pointer in registered:
            unregister_shared_storage(pointer)


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
