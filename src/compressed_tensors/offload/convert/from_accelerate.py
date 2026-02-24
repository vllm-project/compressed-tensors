# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING, Literal

import torch
import torch.distributed as dist
from compressed_tensors.offload.cache import DiskCache
from compressed_tensors.offload.convert.helpers import (
    DEFAULT_OFFLOAD_DEVICE,
    get_tensors,
    norm_device,
)
from compressed_tensors.offload.dispatch import dispatch_with_map
from compressed_tensors.offload.dist_utils import is_distributed, is_rank0
from compressed_tensors.offload.utils import to_tensor
from loguru import logger


if TYPE_CHECKING:
    from accelerate.utils import OffloadedWeightsLoader
    from compressed_tensors.offload.dispatch import DeviceMap


__all__ = ["from_accelerate", "remove_accelerate", "remove_accelerate_from_module"]


def from_accelerate(model: torch.nn.Module) -> tuple["DeviceMap", str | None]:
    """
    Convert a model from accelerate offloading to compressed-tensors offloading. Often
    called by `load_offloaded_model` to load offloaded models across ranks.

    Rank 0 is always expected to provide an accelerate-offloaded model.
    Other ranks (if they exist) may provide a model on any device, with or
    without accelerate offloading.

    - If called after `load_offloaded_model()`, other ranks will provide a meta model
        with no accelerate offloading
    - If called after `to_accelerate`, other ranks will provide an accelerate-offloaded
        model shared cpu tensors/file paths.

    :param model: accelerate-offloaded model if rank0, meta model otherwise
    """
    device_map, offload_dir = remove_accelerate(model)

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
        onload_dev, offload_dev, _offload_dir = remove_accelerate_from_module(module)

        if _offload_dir is not None:
            if offload_dir is not None and _offload_dir != offload_dir:
                raise ValueError(
                    "Expected model to only have one `offload_dir`, "
                    f"instead got {offload_dir} and {_offload_dir}"
                )

            offload_dir = _offload_dir

        device_map[name] = (onload_dev, offload_dev)

    if hasattr(model, "hf_device_map"):
        delattr(model, "hf_device_map")

    return device_map, offload_dir


def remove_accelerate_from_module(
    module: torch.nn.Module,
) -> tuple[torch.device | None, torch.device | Literal["disk"] | None, str | None]:
    """
    Remove accelerate offloading from a module, if present.
    Absolutely no device movement occurs, and parameters/buffers pointers from state
    dicts coming from `to_accelerate` remain unchanged so as to avoid memory duplication

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
        hook.offload = False
        remove_hook_from_module(module, recurse=False)
        device = _infer_device_from_tensors(direct_tensors)
        return device, device, None

    # Unwrap PrefixedDataset chain so we can look up real tensor keys
    prefix, dataset = _unwrap_prefixed_dataset(hook.weights_map, PrefixedDataset)
    assert isinstance(dataset, (OffloadedWeightsLoader, dict))

    offload_dev: str | None = None

    for local_name, tensor in direct_tensors.items():
        full_name = prefix + local_name

        # Device/CPU offload
        if isinstance(dataset, dict) and local_name in dataset:
            offload = dataset[local_name]
            offload_dev = _set_or_validate_offload(offload_dev, offload.device.type)

        # Device/CPU offload
        elif (
            isinstance(dataset, OffloadedWeightsLoader)
            and full_name in dataset.state_dict
        ):
            offload = dataset.state_dict[full_name]
            offload_dev = _set_or_validate_offload(offload_dev, offload.device.type)

        # Disk offload
        elif isinstance(dataset, OffloadedWeightsLoader) and full_name in dataset.index:
            offload = tensor
            offload_dev = _set_or_validate_offload(offload_dev, "disk")
            assert offload.device.type == "meta"
            assert isinstance(offload, (torch.nn.Parameter, torch.nn.Buffer))

            # Copy accelerate's disk index into DiskCache for our later use
            if is_rank0():
                _save_ct_index_entry(dataset, full_name, tensor)

        # Not offloaded, likely a buffer
        else:
            offload = tensor

        # Replace meta tensor with offloaded value (no ptr rematerialization occurs)
        # In the disk case, the tensor remains as the meta tensor
        if not isinstance(offload, (torch.nn.Parameter, torch.nn.Buffer)):
            to_tensor(offload, tensor)
        setattr(module, local_name, offload)

    # Prevent onloading disk tensors while removing hook
    hook.offload = False
    remove_hook_from_module(module, recurse=False)

    return (
        norm_device(hook.execution_device),
        norm_device(offload_dev if offload_dev is not None else DEFAULT_OFFLOAD_DEVICE),
        dataset.save_folder if isinstance(dataset, OffloadedWeightsLoader) else None,
    )


def _save_ct_index_entry(
    dataset: "OffloadedWeightsLoader",
    name: str,
    offloaded: torch.Tensor,
):
    entry: dict = dataset.index[name]

    if "safetensors_file" in entry and offloaded not in DiskCache.index:
        # typical case: model is loaded from safetensors file
        # create a symlink that points to the model safetensor file
        # if the value is ever updated, the symlink is broken and a real file
        # is written to that location
        DiskCache.create_checkpoint_symlink(offloaded, entry, dataset.save_folder)

    else:
        # unfortunately, ct's implementation does not support loading non-safetensors
        # we must onload and save as safetensors. This typically only occurs in testing
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


def _direct_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: t for name, t in get_tensors(module) if t is not None}


def _infer_device_from_tensors(tensors: dict[str, torch.Tensor]) -> torch.device | None:
    t = next(iter(tensors.values()), None)
    return norm_device(t.device if t is not None else None)


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
