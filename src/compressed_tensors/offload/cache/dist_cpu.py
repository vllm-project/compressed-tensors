# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from compressed_tensors.offload.cache.cpu import CPUCache
from compressed_tensors.offload.utils import to_empty


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
            tensor = to_empty(tensor, device=self.offload_device)
            tensor.set_(
                torch.UntypedStorage._new_shared_filename_cpu(*broadcast_obj),
                storage_offset=tensor.storage_offset(),
                size=tensor.size(),
                stride=tensor.stride(),
            )

        # ensure that rank 0 does not garbage collect before other ranks reconstruct
        dist.barrier()

        return tensor
