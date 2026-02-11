# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections import defaultdict
from itertools import chain
from typing import Iterable

import torch
from compressed_tensors.offload.cache import DiskCache, OffloadCache
from compressed_tensors.offload.module import remove_module_offload
from compressed_tensors.utils import patch_attr
from loguru import logger


__all__ = ["to_accelerate"]


def to_accelerate(model: torch.nn.Module):
    """
    Convert a model from `compressed_tensors` offloading to `accelerate` offloading.
    This is is often called before `PreTrainedModel.save_pretrained`, as without this
    conversion, `save_pretrained` will use excessive memory and device movement.

    :param model: model dispatched with `compressed_tensors` offloading
    """
    from compressed_tensors.offload import get_offloaded_device  # avoid circular import

    try:
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module
        from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset
    except ImportError:
        if any(isinstance(m._parameters, OffloadCache) for m in model.modules()):
            logger.warning(
                "Cannot convert model without `accelerate` installed. This may result "
                "in high memory usage during saving and tied tensors being saved twice"
            )
        return

    hf_device_map = {}
    hf_disk_index = _to_accelerate_disk_index(model, DiskCache.index)

    for name, module in model.named_modules():
        cache = module._parameters
        if isinstance(cache, OffloadCache):
            remove_module_offload(module, onload_tensors=False)

            # create weights map
            if isinstance(cache, DiskCache):
                weights_map = PrefixedDataset(
                    prefix=f"{name}.",
                    dataset=OffloadedWeightsLoader(
                        index=hf_disk_index,
                        save_folder=cache.offload_dir,
                    ),
                )
            else:
                weights_map = dict(_get_tensors(module, recurse=False))

            # create hook
            hook = AlignDevicesHook(
                execution_device=cache.onload_device,
                offload=True,
                io_same_device=True,
                weights_map=weights_map,
                offload_buffers=True,
                place_submodules=False,
            )

            # add hook
            with patch_attr(AlignDevicesHook, "init_hook", lambda self, module: module):
                add_hook_to_module(module, hook)

        hf_device_map[name] = get_offloaded_device(module, torch.device("cpu")).type

    # for some reason, in transformers<5, we need at least 2 device types to save
    # this is essentially always going to be the case
    # this is pretty much always the case, but let's catch it here anyways
    if len(set(hf_device_map.values())) <= 1:
        raise NotImplementedError("Accelerate requires hybrid offloading for saving")

    setattr(model, "hf_device_map", hf_device_map)


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


def _get_tensors(
    module: torch.nn.Module, recurse: bool = False
) -> Iterable[tuple[str, torch.Tensor | None]]:
    return chain(
        module.named_parameters(recurse=recurse), module.named_buffers(recurse=recurse)
    )


def _invert_dict(d: dict) -> dict:
    inverted = defaultdict(list)
    for key, value in d.items():
        inverted[value].append(key)
    return inverted
