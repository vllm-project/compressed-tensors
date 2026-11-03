# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from functools import wraps

import torch
import torch.distributed as dist
from compressed_tensors.distributed import as_broadcastable
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.cache.dist_disk import DistributedDiskCache
from compressed_tensors.offload.cache.disk_utils import disk_load_context
from compressed_tensors.offload.utils import send_tensors, to_empty


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
    if (
        pin_memory
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
        and modules
    ):
        # Prefetch on rank 0 without collectives in the worker thread. The
        # main thread broadcasts GPU weights during subgraph onload.
        first_module = next(iter(modules.values()))
        first_module._distributed_disk_stage_error = prepare_distributed_disk_stage(
            modules
        )
        return

    with disk_load_context():
        for name, module in modules.items():
            if not isinstance(module._parameters, OffloadCache):
                # we should consider raising warnings, but observers will
                # clog the output with warnings, so we will skip for now
                continue

            stage_module_offload(module, pin_memory=pin_memory)


def prepare_distributed_disk_stage(
    modules: dict[str, torch.nn.Module],
) -> Exception | None:
    """Prefetch pinned disk weights on rank 0 without worker-thread collectives."""
    is_source = dist.get_rank() == 0
    use_gpu_broadcast = dist.get_backend() in ("nccl", "gloo")
    try:
        with disk_load_context():
            for module in modules.values():
                cache = module._parameters
                if not isinstance(cache, OffloadCache):
                    continue
                if (
                    use_gpu_broadcast
                    and isinstance(cache, DistributedDiskCache)
                    and torch.device(cache.onload_device).type == "cuda"
                    and not is_source
                ):
                    continue
                stage_module_offload(module, pin_memory=True)
    except Exception as error:
        # The main thread relays this error to every rank before GPU broadcasts.
        return error
    return None


def finish_distributed_disk_stage(
    modules: dict[str, torch.nn.Module], stage_error: Exception | None
) -> None:
    """Onload matching disk weights through rank 0, then broadcast GPU tensors."""
    use_gpu_broadcast = dist.get_backend() in ("nccl", "gloo")
    is_source = dist.get_rank() == 0
    metadata = {}
    local_error = stage_error
    try:
        if local_error is None and use_gpu_broadcast:
            for module_name, module in modules.items():
                if not isinstance(module._parameters, DistributedDiskCache):
                    continue
                for attr in ("_parameters", "_buffers"):
                    cache = getattr(module, attr)
                    if torch.device(cache.onload_device).type != "cuda":
                        continue
                    for tensor_name, tensor in cache.offloaded_values.items():
                        if tensor is not None and (
                            not is_source or tensor.is_contiguous()
                        ):
                            metadata[(module_name, attr, tensor_name)] = (
                                tensor.dtype,
                                tuple(tensor.shape),
                            )
    except Exception as error:
        local_error = error

    states = [None] * dist.get_world_size()
    dist.all_gather_object(
        states, (metadata, repr(local_error) if local_error else None)
    )
    errors = [error for _, error in states]
    if any(errors):
        if local_error is not None:
            raise local_error
        raise RuntimeError(f"Distributed disk staging failed on another rank: {errors}")
    if not use_gpu_broadcast:
        return

    # Only broadcast weights that every rank owns with matching metadata.
    common = set(states[0][0])
    for rank_metadata, _ in states[1:]:
        common.intersection_update(rank_metadata)
    common = sorted(
        key
        for key in common
        if all(rank_metadata[key] == states[0][0][key] for rank_metadata, _ in states)
    )
    common_set = set(common)

    ready = []
    gpu_values = {}
    local_error = None
    try:
        for module_name, module in modules.items():
            if not isinstance(module._parameters, DistributedDiskCache):
                continue
            for attr in ("_parameters", "_buffers"):
                cache = getattr(module, attr)
                if torch.device(cache.onload_device).type != "cuda":
                    continue
                values = {}
                for tensor_name, tensor in cache.offloaded_values.items():
                    if tensor is None:
                        values[tensor_name] = None
                        continue
                    key = (module_name, attr, tensor_name)
                    if key in common_set:
                        gpu_tensor = (
                            cache.onload(tensor)
                            if is_source
                            else to_empty(
                                tensor,
                                device=cache.onload_device,
                                memory_format=torch.contiguous_format,
                            )
                        )
                        gpu_values[key] = gpu_tensor
                        values[tensor_name] = gpu_tensor
                    else:
                        # Ranks with different local modules keep the disk path.
                        values[tensor_name] = (
                            tensor
                            if is_source
                            else cache.stage(tensor, pin_memory=True)
                        )
                ready.append((cache, values))
    except Exception as error:
        local_error = error

    # Every rank must finish allocation and fallback staging before broadcasts.
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, repr(local_error) if local_error else None)
    if any(errors):
        if local_error is not None:
            raise local_error
        raise RuntimeError(f"Distributed disk onload failed on another rank: {errors}")

    for key in common:
        tensor = gpu_values[key]
        if tensor.numel():
            dist.broadcast(as_broadcastable(tensor), src=0)

    for cache, values in ready:
        cache.offloaded_values = values
        cache.is_staged = True
    for module in modules.values():
        if isinstance(module._parameters, DistributedDiskCache) and hasattr(
            module, "_original_forward_func"
        ):
            module.forward = module._original_forward_func.__get__(module)
            del module._original_forward_func


def subgraph_onload_modules(
    modules: dict[str, torch.nn.Module],
) -> dict[str, dict]:
    """Onload modules, consuming tensors staged in CPU memory."""
    from compressed_tensors.offload import get_cache_init_kwargs

    first_module = next(iter(modules.values()), None)
    if first_module is not None and hasattr(
        first_module, "_distributed_disk_stage_error"
    ):
        try:
            finish_distributed_disk_stage(
                modules, first_module._distributed_disk_stage_error
            )
        finally:
            del first_module._distributed_disk_stage_error

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
