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
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import ClassVar, Literal, Optional
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist
from compressed_tensors.utils.global_access import GlobalAccess


class OffloadCache(GlobalAccess, MutableMapping, ABC):
    onload_device: torch.device | str
    offload_device: ClassVar[Optional[torch.device | str]]

    # mutable flags for disabling
    offloading_disabled: ClassVar[list[bool]]
    onloading_disabled: ClassVar[list[bool]]

    # offloaded tensors -> onloaded tensors
    onload_values: ClassVar[WeakValueDictionary[torch.Tensor, torch.Tensor]]

    # while offloading is disabled, keep a strong reference
    keep_onloaded_values: ClassVar[set[torch.Tensor]]

    # populated by _parameters or _buffers
    # names -> offloaded tensors
    offloaded_values: dict[str, torch.Tensor]

    @classmethod
    def from_device(
        cls,
        offload_device: Optional[torch.device | str | Literal["disk"]] = None,
        distributed: Optional[bool] = None,
    ) -> type["OffloadCache"]:
        from compressed_tensors.offload.cache.cpu import CPUCache

        if distributed is None:
            distributed = dist.is_available() and dist.is_initialized()

        if offload_device == torch.device("cpu") and not distributed:
            return CPUCache
        else:
            raise NotImplementedError(
                f"Offload of type {offload_device} and "
                f"distributed={distributed} has not been implemented"
            )

    @classmethod
    def from_mapping(
        cls,
        mapping: MutableMapping[str, torch.Tensor | None],
        onload_device: torch.device | str,
    ):
        instance = cls(onload_device=onload_device)
        instance.offloaded_values = {
            name: instance.offload(tensor) for name, tensor in mapping.items()
        }

        return instance

    def __init__(self, onload_device: torch.device | str):
        self.onload_device = onload_device
        self.offloaded_values = dict()

    @abstractmethod
    def onload(self, offloaded: torch.Tensor) -> torch.Tensor:
        """
        Given an offloaded value, returns, onloaded version of that tensor

        :param offloaded: offloaded tensor
        :return: onloaded tensor
        """
        raise NotImplementedError()

    @abstractmethod
    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor:
        """
        Given an onloaded value, returns the offloaded version of that tensor

        :param tensor: tensor to offload
        :return: offloaded tensor
        """
        raise NotImplementedError()

    def __getitem__(self, key: str) -> torch.Tensor:
        """
        :param key: offloaded tensor to be onloaded
        :return: onloaded tensor
        """
        offloaded = self.offloaded_values[key]
        if offloaded is None or self.onloading_disabled[0]:
            return offloaded

        # onload value, potentially from cache
        if offloaded not in self.onload_values:

            # onload value from (cpu)
            onloaded_value = self.onload(offloaded)
            self.onload_values[offloaded] = onloaded_value

        else:
            onloaded_value = self.onload_values[offloaded]

        # keep a strong reference to keep in weakref dict
        if self.offloading_disabled[0]:
            self.keep_onloaded_values.add(onloaded_value)

        return onloaded_value

    def __setitem__(self, key: str, value: torch.Tensor):
        """ """
        if key in self:
            del self[key]

        # when onloading is disabled, parameters can be access and assigned directly
        if self.onloading_disabled:
            self.offloaded_values[key] = value
            return

        self.offloaded_values[key] = self.offload(value)

    def __delitem__(self, key: str):
        """ """
        if key not in self.offloaded_values:
            raise KeyError(key)

        offloaded = self.offloaded_values[key]
        del self.offloaded_values[key]

        # remove weakref
        if offloaded in self.onload_values:
            onloaded = self.onload_values[offloaded]
            del self.onload_values[offloaded]

            # remove strong ref
            if onloaded in self.keep_onloaded_values:
                self.keep_onloaded_values.remove(onloaded)

    def __contains__(self, key) -> bool:
        return key in self.offloaded_values

    def __iter__(self):
        return iter(self.offloaded_values)

    def __len__(self):
        return len(self.offloaded_values)

    @classmethod
    @contextlib.contextmanager
    def disable_offloading(cls):
        """
        Context to disable all offloading for offloaded modules which share this cache.
        After a weight has been fetched once, that onloaded value is cached and
        subsequent fetches will leverage the cache, reducing device movement
        """
        if not cls.offloading_disabled[0]:
            cls.offloading_disabled[0] = True
            cls.keep_onloaded_values.update(cls.onload_values.values())
            yield
            cls.offloading_disabled[0] = False
            cls.keep_onloaded_values.clear()
        else:
            yield

    @classmethod
    @contextlib.contextmanager
    def disable_onloading(cls):
        """
        Context to disable all onloading for offloaded modules which share this cache.
        This is mostly used for debugging purposes, and allows the caller to directly
        inspect offloaded tensors and directly assign offloaded tensors without copying
        """
        if not cls.onloading_disabled[0]:
            cls.onloading_disabled[0] = True
            yield
            cls.onloading_disabled[0] = False
        else:
            yield
