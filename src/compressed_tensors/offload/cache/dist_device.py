# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
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
