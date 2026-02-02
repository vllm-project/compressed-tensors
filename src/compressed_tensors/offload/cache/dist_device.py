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
from torch._prims_common import DeviceLikeType


class DistributedDeviceCache(DeviceCache):
    """
    Handles offloading and onloading tensors from/to device memory. Onloading
    tensors is a no-op (except when in a align_module_device context).
    """

    def __init__(self, onload_device: DeviceLikeType):
        super().__init__(onload_device)
        self.offload_device = self.onload_device

    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor:
        """
        TODO

        :param value: tensor on any device
        :return: tensor on cpu
        """
        if tensor is None:
            return None

        if dist.get_rank() == 0:
            tensor = super().offload(tensor)

        else:
            tensor = torch.empty_like(tensor, device=self.offload_device)

        dist.broadcast(tensor, src=0)
        return tensor
