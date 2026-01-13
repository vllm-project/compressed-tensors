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

from typing import Optional

import torch
import torch.distributed as dist
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.utils import send_tensors


class DistributedDiskCache(DiskCache):
    """
    TODO

    Note: This cache does not currently handle propagation of in-place
    operations on the onloaded tensors. Future work could support this by
    returning a tensor subclass which references on offloaded tensor. To update
    parameters, use `compressed_tensors.offload::update_offload_parameter`
    """

    offload_device: Optional[torch.device | str] = torch.device("cpu")

    def onload(self, key: torch.Tensor) -> torch.Tensor:
        if dist.get_rank() == 0:
            # read from disk
            value = super().onload(key)
        else:
            value = None

        # I assume this moves the device?
        dist.broadcast(value, src=0)

        return value

    def offload(self, value: torch.Tensor) -> torch.Tensor:
        # return original tensor if onloading is disabled
        # to allow for direct parameter/buffer assignment
        if self.onloading_disabled:
            return value

        if dist.get_rank() == 0:
            # write to disk
            super().offload(value)

        # wait for write to finish
        dist.barrier()

        return send_tensors(value, device="meta")

    def __delitem__(self, key: torch.Tensor):
        raise NotImplementedError()
