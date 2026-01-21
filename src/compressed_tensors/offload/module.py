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

import contextlib
from functools import wraps

import torch
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.utils import send_tensors


def offload_module(
    module: torch.nn.Module,
    onload_device: torch.device | str,
    offload_device: torch.device | str,
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
    """
    cache_cls = OffloadCache.cls_from_device(offload_device)
    module._parameters = cache_cls.from_mapping(module._parameters, onload_device)
    module._buffers = cache_cls.from_mapping(module._buffers, onload_device)

    original_forward_func = module.forward.__func__
    module._original_forward_func = original_forward_func

    @wraps(original_forward_func)
    def forward(self, *args, **kwargs):
        if not OffloadCache.onloading_disabled:
            args = send_tensors(args, device=onload_device)
            kwargs = send_tensors(kwargs, device=onload_device)

        return self._original_forward_func(self, *args, **kwargs)

    module.forward = forward.__get__(module)

    return module


def remove_module_offload(module: torch.nn.Module, onload_tensors: bool = False):
    """
    Remove any offloading applied to the module

    :param onload_tensors: Whether to move tensors to the onloaded device, or keep them
        on the offload device. Defaults to False.
    """
    if isinstance(module._parameters, OffloadCache):
        assert isinstance(module._buffers, OffloadCache)

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

        module.forward = module._original_forward_func.__get__(module)
        del module._original_forward_func


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
