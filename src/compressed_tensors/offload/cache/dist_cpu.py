# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from compressed_tensors.distributed import get_source_rank, is_source_process
from compressed_tensors.offload.cache.cpu import CPUCache
from compressed_tensors.offload.cache.utils import catch_cpu_mem_error
from compressed_tensors.offload.utils import send_tensors, to_empty


class DistributedCPUCache(CPUCache):
    """
    Handles offloading and onloading tensors from/to cpu memory shared across processes
    """

    @catch_cpu_mem_error
    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """
        Synchronously create shared cpu memory for offload.

        The dtype of ``tensor`` on non-source ranks cannot be trusted because
        transformers may initialize buffers (e.g. ``inv_freq``) with a
        different dtype than the checkpoint value on the source rank. See
        https://github.com/huggingface/transformers/pull/47486

        :param tensor: tensor on any device
        :return: cpu tensor whose data is located in shared memory
        """
        if tensor is None:
            return None

        # slight runtime cost for views
        tensor = tensor.contiguous()

        if is_source_process():
            # create shared memory cpu tensor
            tensor = super().offload(tensor).share_memory_()
            handle, filename, nbytes = tensor.untyped_storage()._share_filename_cpu_()
            broadcast_obj = [handle, filename, nbytes, tensor.dtype]
        else:
            broadcast_obj = [None, None, None, None]

        # receive shared memory file handle
        dist.broadcast_object_list(broadcast_obj, src=get_source_rank())

        if not is_source_process():
            src_dtype = broadcast_obj.pop(3)

            if tensor.device.type == "meta" or tensor.dtype != src_dtype:
                tensor = to_empty(tensor, device=self.offload_device, dtype=src_dtype)
            else:
                tensor = send_tensors(tensor, device=self.offload_device)

            # reconstruct tensor from shared memory file handle
            with torch.no_grad():
                tensor.set_(
                    torch.UntypedStorage._new_shared_filename_cpu(*broadcast_obj),
                    storage_offset=0,
                    size=tensor.size(),
                    stride=tensor.stride(),
                )

        # ensure that rank 0 does not garbage collect before other ranks reconstruct
        dist.barrier()

        return tensor
