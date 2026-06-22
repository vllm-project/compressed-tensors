# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from compressed_tensors.distributed import get_source_rank, is_source_process
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.utils import send_tensors


class DistributedDiskCache(DiskCache):
    """
    Handles offloading and onloading tensors from/to disk. For more information, see
    `compressed_tensors.offload.cache.disk_cache::DiskCache`.
    """

    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """
        Synchronously write tensor data to disk

        :param tensor: tensor on any device
        :return: meta tensor representing disk offloaded parameter
        """
        if tensor is None:
            return None

        if is_source_process():
            # write to disk
            offloaded = super().offload(tensor)
            offloaded_id = id(offloaded)
            broadcast_obj = [
                self.index[offloaded_id]["safetensors_file"],
                self.index[offloaded_id]["weight_name"],
                self.index[offloaded_id]["dtype"],
            ]
        else:
            offloaded = send_tensors(tensor, device="meta")
            broadcast_obj = [None, None, None]

        dist.broadcast_object_list(broadcast_obj, src=get_source_rank())

        if not is_source_process():
            self.index[id(offloaded)] = {
                "safetensors_file": broadcast_obj[0],
                "weight_name": broadcast_obj[1],
                "dtype": broadcast_obj[2],
            }

        # wait for write to finish
        dist.barrier()
        return offloaded

    @classmethod
    def _disk_finalizer(cls, tensor_id: int):
        """
        Finalizer attached to tensors when they are assigned in `DiskCache.index`.
        Deletes tensor from `DiskCache.index` and deletes associated safetensors file.
        Only rank 0 deletes files.

        :param tensor_id: id of offloaded meta tensor
        """
        if is_source_process():
            super()._disk_finalizer(tensor_id)
        else:
            if tensor_id in cls.index:  # multiple finalizers may be active
                file_path = cls.index[tensor_id]["safetensors_file"]
                assert cls._is_ct_file_path(file_path)
                del cls.index[tensor_id]
