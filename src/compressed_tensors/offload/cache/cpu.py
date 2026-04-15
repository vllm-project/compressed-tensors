# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.cache.utils import catch_cpu_mem_error
from compressed_tensors.offload.utils import send_tensors


class CPUCache(OffloadCache):
    """
    Handles offloading and onloading tensors from/to cpu memory
    """

    offload_device = torch.device("cpu")

    def __init__(self, onload_device: torch.device | str, offload_device=None):
        super().__init__(onload_device, offload_device=offload_device)

    def onload(self, offloaded: torch.Tensor | None) -> torch.Tensor | None:
        """
        Onload a tensor from cpu to device

        :param key: cpu tensor to onload
        :return: device tensor
        """
        return send_tensors(offloaded, device=self.onload_device, copy=False)

    @catch_cpu_mem_error
    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """
        Offload a tensor from any device to cpu

        :param value: tensor on any device
        :return: tensor on cpu
        """
        return send_tensors(tensor, device=self.offload_device, copy=False)

    @torch.no_grad()
    def update_offload(self, offloaded: torch.Tensor, data: torch.Tensor | None):
        """
        Update the offloaded cpu value with new data

        :param offloaded: cpu tensor to update
        :param data: new data to copy from
        """
        if data is not None:
            offloaded.copy_(data)
