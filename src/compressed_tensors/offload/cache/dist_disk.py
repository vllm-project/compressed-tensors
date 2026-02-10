# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from compressed_tensors.offload.cache.disk import DiskCache


class DistributedDiskCache(DiskCache):
    """
    Handles offloading and onloading tensors from/to disk. For more information, see
    `compressed_tensors.offload.cache.disk_cache::DiskCache`.
    """

    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor:
        if dist.get_rank() == 0:
            # write to disk
            offloaded = super().offload(tensor)
            broadcast_obj = [
                self.index[offloaded]["safetensors_file"],
                self.index[offloaded]["weight_name"],
                self.index[offloaded]["dtype"],
            ]
        else:
            offloaded = tensor.to(device="meta")
            broadcast_obj = [None, None, None]

        dist.broadcast_object_list(broadcast_obj, src=0)

        if dist.get_rank() != 0:
            self.index[offloaded] = {
                "safetensors_file": broadcast_obj[0],
                "weight_name": broadcast_obj[1],
                "dtype": broadcast_obj[2],
            }

        # wait for write to finish
        dist.barrier()
        return offloaded

    def __delitem__(self, key: str):
        """
        Remove the offloaded tensor associated with `key`. Any references to its
        onloaded tensors held by this class are invalidated.

        :param key: name of tensor to invalidate
        """
        if dist.get_rank() == 0:
            super().__delitem__(key)
        else:
            offloaded = self.offloaded_values[key]
            del self.index[offloaded]
            super(DiskCache, self).__delitem__(key)
