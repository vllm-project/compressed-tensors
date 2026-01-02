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
from typing import Iterator

import torch
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.utils import send_tensors


def offload_module(
    module: torch.nn.Module,
    cache_cls: type[OffloadCache],
    onload_device: torch.device | str,
    no_split: bool = False,
):
    module.__dict__["_parameters"] = cache_cls.from_mapping(
        module._parameters, onload_device
    )
    module.__dict__["_buffers"] = cache_cls.from_mapping(module._buffers, onload_device)

    original_forward_func = module.forward.__func__
    module._original_forward_func = original_forward_func

    @wraps(original_forward_func)
    def forward(self, *args, **kwargs):
        if not cache_cls.onloading_disabled[0]:
            args = send_tensors(args, device=onload_device)
            kwargs = send_tensors(kwargs, device=onload_device)

        if no_split:
            with cache_cls.disable_offloading():
                return module._original_forward_func(self, *args, **kwargs)
        else:
            return module._original_forward_func(self, *args, **kwargs)

    module.forward = forward.__get__(module)

    return module


def remove_module_offload(module: torch.nn.Module):
    if isinstance(module._parameters, OffloadCache):
        assert isinstance(module._buffers, OffloadCache)

        module._parameters = {
            name: module._parameters.onload(param)
            for name, param in module._parameters.offloaded_values.items()
        }
        module._buffers = {
            name: module._buffers.onload(param)
            for name, param in module._buffers.offloaded_values.items()
        }

        original_forward_func = module.forward.__func__.__wrapped__
        module.forward = original_forward_func.__get__(module)


@contextlib.contextmanager
def unwrap_offload_forward(module: torch.nn.Module) -> Iterator[torch.nn.Module]:
    """
    Context manager that returns the module without offload wrapping.
    This can be used to modify the original module. The module is rewrapped upon exit

    :param module: module that may be offloaded
    :returns: module with offload wrapping removed
    """
    if hasattr(module, "_original_forward_func"):
        offload_forward = module.forward
        module.forward = module._original_forward_func.__get__(module)
        yield
        module._original_forward_func = module.forward.__func__
        module.forward = offload_forward

    else:
        yield
