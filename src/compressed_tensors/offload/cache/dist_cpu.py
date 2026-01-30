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
from compressed_tensors.offload.cache.cpu import CPUCache


class DistributedCPUCache(CPUCache):
    """
    Handles offloading and onloading tensors from/to cpu memory shared across processes
    """

    offload_device = torch.device("cpu")

    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor:
        if tensor is None:
            return None

        # slight runtime cost for views
        tensor = tensor.contiguous()

        if dist.get_rank() == 0:
            # create shared memory cpu tensor
            tensor = super().offload(tensor).share_memory_()
            (handle, filename, nbytes) = tensor.untyped_storage()._share_filename_cpu_()
            broadcast_obj = [handle, filename, nbytes]
        else:
            broadcast_obj = [None, None, None]

        # receive shared memory file handle
        dist.broadcast_object_list(broadcast_obj, src=0)

        if dist.get_rank() != 0:
            # reconstruct tensor from shared memory file handle
            tensor = torch.empty_like(tensor, device=self.offload_device)
            tensor.set_(torch.UntypedStorage._new_shared_filename_cpu(*broadcast_obj))

        # ensure that rank 0 does not garbage collect before other ranks reconstruct
        dist.barrier()

        return tensor
