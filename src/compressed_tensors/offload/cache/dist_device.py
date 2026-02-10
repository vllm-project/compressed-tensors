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
import torch.distributed as dist
from compressed_tensors.offload.cache.device import DeviceCache
from compressed_tensors.offload.utils import to_empty


class DistributedDeviceCache(DeviceCache):
    """
    Handles offloading and onloading tensors from/to device memory. Onloading
    tensors is typically a no-op (except when onload device has been modified).

    The device offload is not shared between ranks. When dispatching with this cache,
    the model is replicated across devices.
    """

    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor:
        """
        Move a tensor to device, then broadcast data to all other ranks

        :param value: tensor on any device
        :return: tensor on device
        """
        if tensor is None:
            return None

        if dist.get_rank() == 0:
            tensor = super().offload(tensor)

        else:
            tensor = to_empty(tensor, device=self.offload_device)

        dist.broadcast(tensor, src=0)
        return tensor
