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
from types import FunctionType
from typing import Any, TypeVar

import torch
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.utils import send_tensors


_offloaded_module_subclasses: dict[str, type] = dict()

ModuleType = TypeVar("ModuleType", bound=torch.nn.Module)


class OffloadedModule(torch.nn.Module):
    _direct_attributes = {
        # core attributes
        "__class__",
        "__dict__",
        "__weakref__",
        # instance attributes
        "_module",
        "_cache",
        "_no_split",
        "register_parameter",
        "disable_offloading",
        "disable_onloading",
        # these functions should return wrapped modules :. are called with wrapped self
        "named_modules",
        "modules",
        # call path
        "__call__",
        "_compiled_call_impl",
        "_call_impl",
        "forward",
    }

    def __init__(self, module: torch.nn.Module, cache: OffloadCache, no_split: bool):
        self._module = module
        self._cache = cache
        self._no_split = no_split

    def __getattribute__(self, name: str) -> object:
        if name in OffloadedModule._direct_attributes:
            return object.__getattribute__(self, name)

        if (value := self._module._parameters.get(name, None)) is not None:
            return self._cache[value]

        if (value := self._module._buffers.get(name, None)) is not None:
            return self._cache[value]

        else:
            return getattr(self._module, name)

    def __setattr__(self, name: str, value: Any):
        if name in OffloadedModule._direct_attributes:
            return object.__setattr__(self, name, value)

        elif isinstance(value, torch.nn.Parameter):
            self.register_parameter(name, value)

        elif isinstance(value, torch.nn.Buffer):
            self.register_buffer(name, value)

        else:
            setattr(self._module, name, value)

    def __delattr__(self, name: str):
        if name in OffloadedModule._direct_attributes:
            return object.__delattr__(self, name)

        if (old_value := self._module._parameters.get(name, None)) is not None:
            del self._cache[old_value]

        if (old_value := self._module._buffers.get(name, None)) is not None:
            del self._cache[old_value]

        delattr(self._module, name)

    def register_parameter(self, name: str, param: torch.nn.Parameter | None):
        if isinstance(param, torch.nn.Parameter):
            param = self._cache.offload(param)

        if (old_value := self._module._parameters.get(name, None)) is not None:
            del self._cache[old_value]

        if (old_value := self._module._buffers.get(name, None)) is not None:
            del self._cache[old_value]

        self._module.register_parameter(name, param)

    def __call__(self, *args, **kwargs):
        args, kwargs = (
            send_tensors(args, device=self._cache.onload_device),
            send_tensors(kwargs, device=self._cache.onload_device),
        )

        if self._no_split:
            with self.disable_offloading():
                return self._module.__call__.__func__(self, *args, **kwargs)
        else:
            return self._module.__call__.__func__(self, *args, **kwargs)

    def forward(self, *args, **kwargs):
        args, kwargs = (
            send_tensors(args, device=self._cache.onload_device),
            send_tensors(kwargs, device=self._cache.onload_device),
        )

        if self._no_split:
            with self.disable_offloading():
                return self._module.forward.__func__(self, *args, **kwargs)
        else:
            return self._module.forward.__func__(self, *args, **kwargs)

    @contextlib.contextmanager
    def disable_offloading(self):
        with self._cache.disable_offloading():
            yield

    @contextlib.contextmanager
    def disable_onloading(self):
        with self._cache.disable_onloading():
            yield

    @classmethod
    def from_module(
        cls,
        module: ModuleType,
        cache: OffloadCache,
        no_split: bool = False,
    ) -> ModuleType:
        class_name = module.__class__.__name__
        if class_name not in _offloaded_module_subclasses:
            _offloaded_module_subclasses[class_name] = make_offload_module_subclass(
                module.__class__
            )

        return _offloaded_module_subclasses[class_name](module, cache, no_split)


def make_offload_module_subclass(parent_cls: type) -> type:
    subclass = type(
        f"Offloaded{parent_cls.__name__}", (OffloadedModule, parent_cls), {}
    )
    subclass.__name__ = parent_cls.__name__

    subclass.forward = copy_function(subclass.forward)
    subclass.forward = wraps(parent_cls.forward)(subclass.forward)

    assert issubclass(subclass, parent_cls)
    return subclass


def copy_function(func):
    return FunctionType(
        func.__code__,
        func.__globals__,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=func.__closure__,
    )
