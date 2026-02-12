# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from itertools import chain
from typing import TYPE_CHECKING, Iterable

import torch
import torch.distributed as dist
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.dispatch import dispatch_with_map
from compressed_tensors.offload.dist_utils import is_distributed, is_rank0
from loguru import logger


if TYPE_CHECKING:
    from accelerate.utils import OffloadedWeightsLoader
    from compressed_tensors.offload.dispatch import DeviceMap


__all__ = ["from_accelerate", "remove_accelerate", "remove_accelerate_from_module"]


def from_accelerate(model: torch.nn.Module) -> tuple["DeviceMap", str | None]:
    """
    Convert a model from accelerate offloading to compressed-tensors offloading. Often
    called by `load_offloaded_model` to load offloaded models across ranks.

    If in a distributed setting, rank0 is expected to provide an accelerate-offloaded
    model, and other ranks are expected to provide a meta model with no offloading

    :param model: accelerate-offloaded model if rank0, meta model otherwise
    """
    if is_rank0():
        device_map, offload_dir = remove_accelerate(model)
    else:
        device_map, offload_dir = None, None

    print(device_map)
    broadcast_obj = [device_map, offload_dir]
    if is_distributed():
        dist.broadcast_object_list(broadcast_obj, src=0)

    dispatch_with_map(model, *broadcast_obj)
    return tuple(broadcast_obj)


def remove_accelerate(model: torch.nn.Module) -> tuple["DeviceMap", str | None]:
    """
    Remove accelerate offloading from a model, if applicable

    :param model: model containing accelerate offloaded modules
    :returns: `(device_map, offload_dir)`
    """
    offload_dir = None
    device_map = {}

    for name, module in model.named_modules(remove_duplicate=False):
        onload, offload, module_dir = remove_accelerate_from_module(module)

        if module_dir is not None and offload_dir not in (None, module_dir):
            raise ValueError(
                "Expected model to only have one `offload_dir`, "
                f"instead got {offload_dir} and {module_dir}"
            )

        if module_dir is not None:
            offload_dir = module_dir

        device_map[name] = (onload, offload)

    if hasattr(model, "hf_device_map"):
        delattr(model, "hf_device_map")

    return device_map, offload_dir


def remove_accelerate_from_module(
    module: torch.nn.Module,
) -> tuple[torch.device | None, torch.device | str | None, str | None]:
    """
    Remove accelerate offloading from a module, if present

    :param module: module to remove offloading from
    :returns: `(onload_device, offload_device, disk_offload_dir)`
    """
    try:
        from accelerate.hooks import AlignDevicesHook, remove_hook_from_module
        from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset
    except ImportError:
        device = _infer_module_device(module)
        return device, device, None

    hook = getattr(module, "_hf_hook", None)
    direct_tensors = _direct_tensors(module)

    # No AlignDevicesHook: treat as "not offloaded"
    if not isinstance(hook, AlignDevicesHook):
        device = _infer_device_from_tensors(direct_tensors)
        return device, device, None

    # Hook exists but no active offload (or nothing to consider)
    if not hook.offload or not direct_tensors:
        remove_hook_from_module(module, recurse=False)
        device = _infer_device_from_tensors(direct_tensors)
        return device, device, None

    # Unwrap PrefixedDataset chain so we can look up real tensor keys
    prefix, dataset = _unwrap_prefixed_dataset(hook.weights_map, PrefixedDataset)
    assert isinstance(dataset, (OffloadedWeightsLoader, dict))

    offload_device: str | None = None

    for local_name, tensor in direct_tensors.items():
        full_name = prefix + local_name

        # CPU offload: present in state_dict (OffloadedWeightsLoader) or dict itself
        if isinstance(dataset, dict) or full_name in dataset.state_dict:
            offload_device = _set_or_validate_offload(offload_device, "cpu")

        # Disk offload: present in dataset.index
        elif full_name in dataset.index:
            offload_device = _set_or_validate_offload(offload_device, "disk")

            # Copy accelerate's disk index into DiskCache for our later use
            assert tensor.device.type == "meta"
            _save_ct_index_entry(dataset, full_name, tensor)

            # Prevent onloading disk tensors after removing hook
            hook.offload = False

    remove_hook_from_module(module, recurse=False)
    return (
        _norm_device(hook.execution_device),
        _norm_device(offload_device),
        dataset.save_folder,
    )


def _save_ct_index_entry(
    dataset: "OffloadedWeightsLoader", name: str, offloaded: torch.Tensor
):
    entry: dict = dataset.index[name]

    if "safetensors_file" in entry:
        # typical case: model is loaded from safetensors file
        DiskCache.index[offloaded] = entry

    else:
        # unfortunately, ct's implementation does not support loading non-safetensors
        # we must onload and save as safetensors. This should only occur while testing
        onloaded = dataset[name]
        DiskCache("cpu", dataset.save_folder).offload(onloaded, offloaded=offloaded)
        logger.warning(
            "Attempting to disk offload a model which was not saved with safetensors. "
            "compressed-tensors only supports disk onload from safetensors files, so "
            "weights must be onloaded and re-saved as safetensors files.",
            log_once=True,
        )

        # remove original weight_file
        original_weight_file = os.path.join(dataset.save_folder, f"{name}.dat")
        if os.path.exists(original_weight_file):
            os.remove(original_weight_file)


def _norm_device(device: str | torch.device | None) -> str | torch.device | None:
    if device not in ("disk", None):
        device = torch.device(device)

    if (
        is_distributed()
        and isinstance(device, torch.device)
        and device.index == dist.get_rank()
    ):
        device = torch.device(type=device.type, index=None)
    
    return device


def _get_tensors(
    module: torch.nn.Module, recurse: bool = False
) -> Iterable[tuple[str, torch.Tensor | None]]:
    return chain(
        module.named_parameters(recurse=recurse),
        module.named_buffers(recurse=recurse),
    )


def _direct_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: t for name, t in _get_tensors(module) if t is not None}


def _infer_device_from_tensors(tensors: dict[str, torch.Tensor]) -> torch.device | None:
    t = next(iter(tensors.values()), None)
    return _norm_device(t.device if t is not None else None)


def _infer_module_device(module: torch.nn.Module) -> torch.device | None:
    return _infer_device_from_tensors(_direct_tensors(module))


def _unwrap_prefixed_dataset(weights_map, PrefixedDatasetType):
    prefix = ""
    dataset = weights_map
    while isinstance(dataset, PrefixedDatasetType):
        prefix += dataset.prefix
        dataset = dataset.dataset
    return prefix, dataset


def _set_or_validate_offload(current: str | None, new: str) -> str:
    if current not in (None, new):
        raise ValueError("Expected all accelerate tensors to share offload")
    return new
