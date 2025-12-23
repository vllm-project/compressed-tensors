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
from typing import Literal, Optional
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist
from compressed_tensors.utils.global_access import GlobalAccess


class OffloadCache(GlobalAccess, ABC):
    """
    Abstract base class for offload cache. Tensors are made ready for caching via
    `offload`, updated via `__setitem__`, and retrieved via `__getitem__`.

    Subclasses must implement `onload` and `offload`.

    Note: This cache does not currently handle propagation of in-place
    operations on the onloaded tensors. Future work could support this by
    returning a tensor subclass which references on offloaded tensor. To update
    parameters, use `compressed_tensors.offload::update_offload_parameter`
    """

    onload_device: torch.device | str
    offload_device: Optional[torch.device | str]

    @classmethod
    def from_devices(
        cls,
        onload_device: torch.device | str,
        offload_device: Optional[torch.device | str | Literal["disk"]] = None,
        distributed: Optional[bool] = None,
    ):
        from compressed_tensors.offload.cache.cpu import CPUCache

        if distributed is None:
            distributed = dist.is_available() and dist.is_initialized()

        if offload_device == torch.device("cpu") and not distributed:
            return CPUCache(onload_device)
        else:
            raise NotImplementedError(
                f"Offload of type {offload_device} and "
                f"distributed={distributed} has not been implemented"
            )

    def __init__(self, onload_device: torch.device | str):
        self.onload_device = onload_device
        self.offload_device = torch.device("cpu")

        # flags for disabling
        self.onloading_disabled: bool = False
        self.offloading_disabled: bool = False

        # offloaded tensors -> onloaded tensors
        self.onload_values: WeakValueDictionary[
            torch.Tensor, torch.Tensor
        ] = WeakValueDictionary()

        # strong ref to values to disable offloading
        self.keep_onloaded_values: set[torch.Tensor] = set()

    @abstractmethod
    def onload(self, key: torch.Tensor) -> torch.Tensor:
        """
        Given an offloaded value, returns, onloaded version of that tensor

        :param key: offloaded tensor
        :return: onloaded tensor
        """
        raise NotImplementedError()

    @abstractmethod
    def offload(self, value: torch.Tensor) -> torch.Tensor:
        """
        Given an onloaded value, returns the offloaded version of that tensor

        :param key: tensor to offload
        :return: offloaded tensor
        """
        raise NotImplementedError()

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        """
        :param key: offloaded tensor to be onloaded
        :return: onloaded tensor
        """
        # return original tensor if onloading is disabled
        if self.onloading_disabled:
            return key

        # onload value, potentially from cache
        if key not in self.onload_values:

            # onload value from (cpu)
            onloaded_value = self.onload(key)
            self.onload_values[key] = onloaded_value

        else:
            onloaded_value = self.onload_values[key]

        # if offloading is disabled, keep a strong reference (to keep the value alive)
        if self.offloading_disabled:
            self.keep_onloaded_values.add(onloaded_value)

        return onloaded_value

    def __setitem__(self, key: torch.Tensor, value: torch.Tensor):
        """
        :param key: offloaded tensor whose value will be updated
        :param value: value used to update
        """
        # invalidate onloaded values
        del self[key]

        # update data
        key.copy_(value)

    def __delitem__(self, key: torch.Tensor):
        """
        :param key: offloaded tensor to be removed from the cache
        """
        # remove any strong references to onloaded values
        if (
            self.offloading_disabled
            and key in self.onload_values
            and self.onload_values[key] in self.keep_onloaded_values
        ):
            self.keep_onloaded_values.remove(self.onload_values[key])

    @contextlib.contextmanager
    def disable_offloading(self):
        """
        Context to disable all offloading for offloaded modules which share this cache.
        After a weight has been fetched once, that onloaded value is cached and
        subsequent fetches will leverage the cache, reducing device movement
        """
        if not self.offloading_disabled:
            self.offloading_disabled = True
            self.keep_onloaded_values.update(self.onload_values.values())
            yield
            self.offloading_disabled = False
            self.keep_onloaded_values.clear()
        else:
            yield

    @contextlib.contextmanager
    def disable_onloading(self):
        """
        Context to disable all onloading for offloaded modules which share this cache.
        This is mostly used for debugging purposes, and allows the caller to directly
        inspect offloaded tensors and directly assign offloaded tensors without copying
        """
        if not self.onloading_disabled:
            self.onloading_disabled = True
            yield
            self.onloading_disabled = False
        else:
            yield
