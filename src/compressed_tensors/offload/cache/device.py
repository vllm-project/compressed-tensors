# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
