# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from compressed_tensors.distributed import (
    as_broadcastable,
    get_source_rank,
    is_source_process,
)
from compressed_tensors.offload.cache.device import DeviceCache
from compressed_tensors.offload.utils import send_tensors, to_empty


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

        if is_source_process():
            tensor = super().offload(tensor)

        # materialize meta tensor only if necessary
        elif tensor.device.type == "meta":
            tensor = to_empty(tensor, device=self.offload_device)
        else:
            tensor = send_tensors(tensor, device=self.offload_device)

        dist.broadcast(as_broadcastable(tensor), src=get_source_rank())
        return tensor

    def update_offload(self, offloaded: torch.Tensor, data: torch.Tensor | None):
        """
        Update the offloaded device value with new data, broadcasting from source rank
        to all other ranks.

        :param offloaded: device tensor to update
        :param data: new data to copy from
        """
        if is_source_process():
            super().update_offload(offloaded, data)

        # broadcast the updated data to all ranks
        dist.broadcast(as_broadcastable(offloaded), src=get_source_rank())
