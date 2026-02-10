# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.utils import send_tensors


if TYPE_CHECKING:
    from torch._prims_common import DeviceLikeType


class DeviceCache(OffloadCache):
    """
    Handles offloading and onloading tensors from/to device memory. Onloading
    tensors is typically a no-op (except onload device has been modified).
    """

    def __init__(self, onload_device: "DeviceLikeType"):
        super().__init__(onload_device)
        self.offload_device = self.onload_device

    def onload(self, offloaded: torch.Tensor | None) -> torch.Tensor:
        """
        Typically a no op, except when onload device has been modified

        :param key: device tensor to onload
        :return: device tensor
        """
        # move because onload_device might be modified after init
        return send_tensors(offloaded, device=self.onload_device, copy=False)

    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor:
        """
        Offload a tensor to the device

        :param value: tensor on any device
        :return: tensor on device
        """
        return send_tensors(tensor, device=self.offload_device, copy=False)
