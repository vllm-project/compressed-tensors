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


from itertools import chain
from typing import Iterable, Literal

import torch
from compressed_tensors.offload.cache import DiskCache, OffloadCache
from compressed_tensors.offload.module import offload_module, remove_module_offload
from loguru import logger


__all__ = ["from_accelerate", "to_accelerate"]


def from_accelerate(model: torch.nn.Module):
    """
    Convert a model from `accelerate` offloading to `compressed_tensors` offloading.
    This is often called after loading a model with `from_pretrained(device_map=...)`.

    :param model: model dispatched with `accelerate` offloading
    """
    try:
        from accelerate.hooks import remove_hook_from_module
    except ImportError:
        return

    for module in model.modules():
        onload_device, offload_device = _get_accelerate_devices(module)
        match (onload_device, offload_device):
            case torch.device(), torch.device():
                remove_hook_from_module(module, recurse=False)
                offload_module(module, onload_device, offload_device)

            case torch.device(), "disk":
                _convert_accelerate_disk(module, onload_device)

            case None, None:
                remove_hook_from_module(module, recurse=False)

            case _:
                raise ValueError()

    if hasattr(model, "hf_device_map"):
        delattr(model, "hf_device_map")

    print("done convert")


def to_accelerate(model: torch.nn.Module):
    """
    Convert a model from `compressed_tensors` offloading to `accelerate` offloading.
    This is is often called before `PreTrainedModel.save_pretrained`, as without this
    conversion, `save_pretrained` will use excessive memory and device movement.

    :param model: model dispatched with `compressed_tensors` offloading
    """
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

    hf_disk_index = {
        weight_info["weight_name"]: weight_info
        for weight_info in DiskCache.index.values()
    }
    hf_device_map = {}

    for name, module in model.named_modules():
        cache = module._parameters
        if isinstance(cache, OffloadCache):
            remove_module_offload(module, onload_tensors=False)

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

            hook = AlignDevicesHook(
                execution_device=cache.onload_device,
                offload=True,
                io_same_device=True,
                weights_map=weights_map,
                offload_buffers=True,
                place_submodules=False,
            )
            add_hook_to_module(module, hook)
            hf_device_map[name] = str(cache.offload_device)

    setattr(model, "hf_device_map", hf_device_map)


def _get_accelerate_devices(
    module: torch.nn.Module,
) -> tuple[torch.device | None, torch.device | Literal["disk"] | None]:
    try:
        from accelerate.hooks import AlignDevicesHook
        from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset
    except ImportError:
        return None, None

    hook = getattr(module, "_hf_hook", None)
    if not isinstance(hook, AlignDevicesHook) or not hook.offload:
        return None, None
    if hook.place_submodules:
        raise ValueError("Cannot convert dispatches with `place_submodules`")
    onload_device = torch.device(hook.execution_device)

    name, _ = next(_get_tensors(module))
    dataset = hook.weights_map
    while isinstance(dataset, PrefixedDataset):
        name = dataset.prefix + name
        dataset = dataset.dataset

    if isinstance(dataset, dict):
        return onload_device, dataset[name].device

    elif isinstance(dataset, OffloadedWeightsLoader):
        if name in dataset.state_dict:
            return onload_device, torch.device("cpu")
        elif name in dataset.index:
            return onload_device, "disk"
        else:
            return None, None

    else:
        raise ValueError()


def _convert_accelerate_disk(module: torch.nn.Module, onload_device: torch.device):
    try:
        from accelerate.hooks import AlignDevicesHook, remove_hook_from_module
        from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset
    except ImportError:
        return

    from compressed_tensors.offload.cache.disk import DiskCache

    hook: AlignDevicesHook = getattr(module, "_hf_hook")
    prefix = ""
    dataset = hook.weights_map
    while isinstance(dataset, PrefixedDataset):
        prefix += dataset.prefix
        dataset = dataset.dataset

    assert isinstance(dataset, OffloadedWeightsLoader)

    for name, tensor in _get_tensors(module):
        if tensor is not None and tensor.device.type == "meta":
            DiskCache.index[tensor] = dataset.index[prefix + name]

    # changing this flag means that tensors are not moved again
    # remove wrapped forward, ect.
    hook.offload = False
    remove_hook_from_module(module, recurse=False)

    # meta tensors are no-ops
    # non-meta tensors are offloaded onto new files
    print("pre")
    offload_module(module, onload_device, "disk", offload_dir=dataset.save_folder)
    print("post")


def _get_tensors(
    module: torch.nn.Module, recurse: bool = False
) -> Iterable[tuple[str, torch.Tensor | None]]:
    return chain(
        module.named_parameters(recurse=recurse), module.named_buffers(recurse=recurse)
    )
