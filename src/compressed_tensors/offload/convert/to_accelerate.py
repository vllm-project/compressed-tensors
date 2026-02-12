# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from typing import Optional

import torch
from compressed_tensors.offload.cache import DiskCache, OffloadCache
from compressed_tensors.offload.convert.helpers import get_tensors
from compressed_tensors.offload.module import remove_module_offload
from compressed_tensors.utils import patch_attr
from loguru import logger


__all__ = ["to_accelerate", "to_accelerate_module"]


def to_accelerate(model: torch.nn.Module) -> dict[str, str]:
    """
    Convert a model from `compressed_tensors` offloading to `accelerate` offloading.
    This is is often called before `PreTrainedModel.save_pretrained`, as without this
    conversion, `save_pretrained` will use excessive memory and device movement.

    :param model: model dispatched with `compressed_tensors` offloading
    :return: accelerate-style device map
    """
    hf_device_map = {}
    hf_disk_index = _to_accelerate_disk_index(model, DiskCache.index)

    for name, module in model.named_modules():
        offload = to_accelerate_module(module, name, hf_disk_index)
        hf_device_map[name] = offload

    setattr(model, "hf_device_map", hf_device_map)
    return hf_device_map


def to_accelerate_module(
    module: torch.nn.Module,
    name: Optional[str] = None,
    hf_disk_index: Optional[dict[str, dict[str, str]]] = None,
) -> str:
    """
    Convert a module from `compressed_tensors` offloading to `accelerate` offloading

    :param module: module to convert to accelerate offloading
    :param name: name of module in model
    :param hf_disk_index: accelerate-style disk index to attach to weight loaders
    :return: str of offloaded device. Defaults to cpu if module does not have parameters
    """
    has_accelerate = True
    try:
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module
        from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset
    except ImportError:
        logger.warning(
            "Cannot convert module without `accelerate` installed. This may result "
            "in high memory usage during saving and tied tensors being saved twice",
            log_once=True,
        )
        has_accelerate = False

    offload_device = torch.device("cpu")
    if has_accelerate and isinstance(module._parameters, OffloadCache):
        cache = module._parameters
        offload_device = cache.offload_device
        remove_module_offload(module, onload_tensors=False)

        # create weights map
        if isinstance(cache, DiskCache):
            if name is None or hf_disk_index is None:
                raise ValueError(
                    "Must provide `name` and `hf_disk_index` "
                    "to convert disk offloaded module"
                )

            weights_map = PrefixedDataset(
                prefix=f"{name}.",
                dataset=OffloadedWeightsLoader(
                    index=hf_disk_index,
                    save_folder=cache.offload_dir,
                ),
            )
        else:
            weights_map = dict(get_tensors(module, recurse=False))

        # create hook
        hook = AlignDevicesHook(
            execution_device=cache.onload_device,
            offload=True,
            io_same_device=True,
            weights_map=weights_map,
            offload_buffers=True,
            place_submodules=False,
        )

        # add hook (skip onloading)
        with patch_attr(AlignDevicesHook, "init_hook", lambda self, module: module):
            add_hook_to_module(module, hook)

    return str(offload_device)


def _to_accelerate_disk_index(
    model: torch.nn.Module, index: dict[torch.Tensor, dict[str, str]]
) -> dict[str, dict[str, str]]:
    from compressed_tensors.offload import disable_onloading  # circular dependency

    with disable_onloading():
        offloaded_to_key = _invert_dict(model.state_dict(keep_vars=True))

    return {
        key: weight_info
        for offloaded, weight_info in index.items()
        for key in offloaded_to_key[offloaded]
    }


def _invert_dict(d: dict) -> dict:
    inverted = defaultdict(list)
    for key, value in d.items():
        inverted[value].append(key)
    return inverted
