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
import inspect
from functools import wraps
from types import FunctionType
from typing import Any, TypeVar

import torch
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.utils import send_tensors


ModuleType = TypeVar("ModuleType", bound=torch.nn.Module)


class OffloadedModule(torch.nn.Module):
    @classmethod
    def from_module(
        cls,
        module: ModuleType,
        cache: OffloadCache,
        no_split=False
    ) -> ModuleType:
        module.__dict__["_parameters"] = cache.from_mapping(module._parameters)
        module.__dict__["_buffers"] = cache.from_mapping(module._buffers)

        original_forward = module.forward

        def forward(self, *args, **kwargs):
            args, kwargs = (
                send_tensors(args, device=self._parameters.onload_device),
                send_tensors(kwargs, device=self._parameters.onload_device),
            )

            if no_split:
                #with self.disable_offloading():
                return original_forward.__func__(self, *args, **kwargs)
            else:
                return original_forward.__func__(self, *args, **kwargs)

        module.forward = wraps(original_forward.__func__)(forward).__get__(module)
        assert inspect.signature(module.forward) == inspect.signature(original_forward)
        
        return module
