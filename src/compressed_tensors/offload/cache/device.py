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

import torch
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.utils import send_tensors


class DeviceCache(OffloadCache):
    """
    Handles offloading and onloading tensors from/to device memory. Onloading
    tensors is a no-op.
    """

    def __init__(self, onload_device: torch.device | str):
        self.onload_device = onload_device
        self.offload_device = onload_device
        self.offloaded_values = dict()

    def onload(self, offloaded: torch.Tensor | None) -> torch.Tensor:
        """
        No op, offloaded tensors are already on device

        :param key: cpu tensor to onload
        :return: device tensor
        """
        # move because onload_device might be modified after init
        return send_tensors(offloaded, device=self.onload_device, copy=False)

    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor:
        """
        Offload a tensor from any device to a device

        :param value: tensor on any device
        :return: tensor on cpu
        """
        return send_tensors(tensor, device=self.offload_device, copy=False)
