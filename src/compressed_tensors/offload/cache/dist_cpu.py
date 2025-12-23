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
from compressed_tensors.offload.cache.cpu import CPUCache


class DistributedCPUCache(CPUCache):
    """
    Handles offloading and onloading tensors from/to cpu memory shared across processes

    Note: This cache does not currently handle propagation of in-place
    operations on the onloaded tensors. Future work could support this by
    returning a tensor subclass which references on offloaded tensor. To update
    parameters, use `compressed_tensors.offload::update_offload_parameter`
    """

    offload_device: Optional[torch.device | str] = torch.device("cpu")

    def offload(self, value: torch.Tensor) -> torch.Tensor:
        # return original tensor if onloading is disabled
        # to allow for direct parameter/buffer assignment
        if self.onloading_disabled:
            return value

        # slight runtime cost for views
        value = value.contiguous()

        if dist.get_rank() == 0:
            # create shared memory cpu tensor
            key = super().offload(value).share_memory_()
            (handle, filename, nbytes) = key.untyped_storage()._share_filename_cpu_()
            broadcast_object = [handle, filename, nbytes]
        else:
            broadcast_object = [None, None, None]

        # receive shared memory file handle
        dist.broadcast_object_list(broadcast_object, src=0)

        # reconstruct tensor from shared memory file handle
        key = torch.empty_like(value, device=self.offload_device)
        key.set_(torch.UntypedStorage._new_shared_filename_cpu(*broadcast_object))
        return key
