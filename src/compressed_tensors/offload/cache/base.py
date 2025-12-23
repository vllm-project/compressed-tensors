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

import torch
import torch.distributed as dist


class OffloadCache(MutableMapping, ABC):
    """
    Base class for offload caches. Subclasses must implement `offload` and `onload`.
    Instances have similar behavior to dicts, except that tensors are offloaded when
    assigned and onloaded when accessed.

    Typical usage:
    ```
    module._parameters = cache_cls.from_mapping(module._parameters, onload_device)
    tensor = ...
    module._parameters["name"] = tensor           # tensor is offloaded
    onloaded_tensor = module._parameters["name"]  # tensor is onloaded
    ```

    This class implements two contexts for more fine-grained control of device movement:
    `OffloadCache.disable_offloading` and `OffloadCache.disable_onloading`. For more
    info, see `compressed_tensors.offload::(disable_offloading|disable_onloading)`
    """

    onload_device: torch.device | str
    offload_device: Optional[torch.device | str]

    # global flags for disabling
    offloading_disabled: ClassVar[bool] = False
    onloading_disabled: ClassVar[bool] = False

    # names -> offloaded tensors (populated from _parameters or _buffers)
    offloaded_values: dict[str, torch.Tensor]

    # offloaded tensors -> onloaded tensors (only when offloading is disabled)
    keep_onloaded_values: ClassVar[dict[torch.Tensor, torch.Tensor]] = dict()

    @classmethod
    def cls_from_device(
        cls,
        device: Optional[torch.device | str | Literal["disk"]] = None,
    ) -> type["OffloadCache"]:
        """
        Get the subclass which implements offloading for the given `offload_device`.
        Use `torch.distributed` to detect if the environment is distributed

        :param device: offload device used to find subclass
        :return: subclass of `OffloadCache`
        """
        from compressed_tensors.offload.cache.cpu import CPUCache
        from compressed_tensors.offload.cache.device import DeviceCache
        from compressed_tensors.offload.cache.dist_cpu import DistributedCPUCache

        device_type = torch.device(device).type if device != "disk" else "disk"
        distributed = dist.is_available() and dist.is_initialized()

        match (device_type, distributed):
            case ("cpu", False):
                return CPUCache
            case ("cpu", True):
                return DistributedCPUCache
            case ("cuda", False):
                return DeviceCache
            case _:
                raise NotImplementedError(
                    f"Offload of type {device} and "
                    f"distributed={distributed} has not been implemented"
                )

    @classmethod
    def from_mapping(
        cls,
        mapping: MutableMapping[str, torch.Tensor | None],
        onload_device: torch.device | str,
    ):
        """
        Initialize an instance from a given mapping, typically `Module._parameters` or
        `Module._buffers`. Mapping values will be offloaded

        :param mapping: mapping used to populate cache
        :param onload_device: device which tensors will be onloaded to
        """
        instance = cls(onload_device=onload_device)
        instance.offloaded_values = {
            name: instance.offload(tensor) for name, tensor in mapping.items()
        }

        return instance

    def __init__(self, onload_device: torch.device | str):
        super().__init__()
        self.onload_device = onload_device
        self.offloaded_values = dict()

    @abstractmethod
    def onload(self, offloaded: torch.Tensor | None) -> torch.Tensor:
        """
        Given an offloaded tensor, returns that tensor after onloading

        :param offloaded: offloaded tensor
        :return: onloaded tensor
        """
        raise NotImplementedError()

    @abstractmethod
    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor:
        """
        Given a tensor, returns that tensor after offloading

        :param tensor: tensor to offload
        :return: offloaded tensor
        """
        raise NotImplementedError()

    def __getitem__(self, key: str) -> torch.Tensor:
        """
        Onload a tensor

        If called within the `disable_offloading` context, a strong reference of the
        onloaded tensor is kept so that future accesses will not require device movement

        :param key: name of tensor to access
        :return: onloaded tensor
        """
        offloaded = self.offloaded_values[key]

        # when onloading is disabled, offloaded tensors can be accessed directly
        if offloaded is None or self.onloading_disabled:
            return offloaded

        # check for cache hit
        if offloaded in self.keep_onloaded_values:
            return self.keep_onloaded_values[offloaded]

        # onload value
        onloaded = self.onload(offloaded)

        # when offloading is disabled, populate cache
        if self.offloading_disabled:
            self.keep_onloaded_values[offloaded] = onloaded

        return onloaded

    def __setitem__(self, key: str, value: torch.Tensor | None):
        """
        Offload a tensor and add it to the cache.

        If called within the `disable_onloading` context, the tensor is not offloaded
        and is instead assigned directly

        :param key: name of tensor
        :param value: tensor value to offload
        """
        if key in self:
            del self[key]

        # when onloading is disabled, parameters can be access and assigned directly
        if self.onloading_disabled:
            self.offloaded_values[key] = value
            return

        self.offloaded_values[key] = self.offload(value)

    def __delitem__(self, key: str):
        """
        Remove the offloaded tensor associated with `key`. Any references to its
        onloaded tensors held by this class are invalidated.

        :param key: name of tensor to invalidate
        """
        offloaded = self.offloaded_values[key]
        del self.offloaded_values[key]

        # remove strong ref
        if offloaded in self.keep_onloaded_values:
            del self.keep_onloaded_values[offloaded]

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
        if not OffloadCache.offloading_disabled:
            OffloadCache.offloading_disabled = True
            yield
            OffloadCache.offloading_disabled = False
            OffloadCache.keep_onloaded_values.clear()
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
        if not OffloadCache.onloading_disabled:
            OffloadCache.onloading_disabled = True
            yield
            OffloadCache.onloading_disabled = False
        else:
            yield
