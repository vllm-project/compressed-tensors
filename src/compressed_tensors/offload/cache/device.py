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
from typing import Optional
from weakref import WeakValueDictionary

import torch
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.utils import send_tensors


class DeviceCache(OffloadCache):
    """
    The device cache handles the onloading of tensors from an offload device to
    an onload device.

    When used with module offloading,
    assumes that the model starts on the offload device

    Note: This cache does not currently handle propagation of in-place
    operations on the onloaded tensors. Future work could support this by
    returning a tensor subclass which references on offloaded tensor.
    """

    def __init__(
        self,
        onload_device: torch.device | str,
        offload_device: Optional[torch.device | str] = None,
    ):
        self.onload_device = onload_device
        self.offload_device = offload_device

        # flags for disabling
        self.onloading_disabled: bool = False
        self.offloading_disabled: bool = False

        # onloaded values cache
        self.onload_values: WeakValueDictionary[
            torch.Tensor, torch.Tensor
        ] = WeakValueDictionary()  # offloaded tensors -> onloaded tensors

        # strong ref to values to disable offloading
        self.keep_onloaded_values: set[torch.Tensor] = set()

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        # return original tensor if onloading is disabled
        if self.onloading_disabled:
            return key

        # onload value, potentially from cache
        if key not in self.onload_values:

            # onload value from (cpu)
            onloaded_value = send_tensors(key, device=self.onload_device, copy=True)
            self.onload_values[key] = onloaded_value

        else:
            onloaded_value = self.onload_values[key]

        # if offloading is disabled, keep a strong reference (to keep the value alive)
        if self.offloading_disabled:
            self.keep_onloaded_values.add(onloaded_value)

        return onloaded_value

    def __setitem__(self, key: torch.Tensor, value: torch.Tensor):
        # invalidate onloaded values
        del self[key]

        # update data
        key.copy_(value)

    def __delitem__(self, key: torch.Tensor):
        # remove any strong references to onloaded values
        if (
            self.offloading_disabled
            and key in self.onload_values
            and self.onload_values[key] in self.keep_onloaded_values
        ):
            self.keep_onloaded_values.remove(self.onload_values[key])

    def offload(self, value: torch.Tensor) -> torch.Tensor:
        # return original tensor if onloading is disabled
        # to allow for direct parameter/buffer assignment
        if self.onloading_disabled:
            return value

        # allow for offload device override
        if self.offload_device is not None:
            offload_device = self.offload_device
        else:
            offload_device = value.device

        return send_tensors(value, device=offload_device, copy=True)

    @contextlib.contextmanager
    def disable_offloading(self):
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
        if not self.onloading_disabled:
            self.onloading_disabled = True
            yield
            self.onloading_disabled = False
        else:
            yield
