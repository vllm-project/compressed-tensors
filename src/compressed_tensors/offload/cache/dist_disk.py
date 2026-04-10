# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from compressed_tensors.distributed import is_source_process
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
            broadcast_obj = [
                self.index[offloaded]["safetensors_file"],
                self.index[offloaded]["weight_name"],
                self.index[offloaded]["dtype"],
            ]
        else:
            offloaded = send_tensors(tensor, device="meta")
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
        Remove the offload associated with `key`. If a new file was created to store
        updated tensor data, that new tensor data file is deleted.

        Any references to onloaded tensors held by this class are invalidated.

        :param key: name of tensor to invalidate
        """
        if dist.get_rank() == 0:
            super().__delitem__(key)
        else:
            offloaded = self.offloaded_values[key]
            del self.index[offloaded]
            super(DiskCache, self).__delitem__(key)

    @classmethod
    def clean_offload_dir(cls, offload_dir: str | None = None) -> int:
        """
        Clean up all intermediate safetensors files created by DiskCache.
        In distributed settings, only rank 0 performs the actual file deletion.

        :param offload_dir: If provided, only clean files in this directory.
                           If None, clean all files in the shared index.
        :return: Number of files cleaned up (only on rank 0, 0 on other ranks)
        """
        if dist.get_rank() == 0:
            files_cleaned = super().clean_offload_dir(offload_dir)
            dist.barrier()
            return files_cleaned
        else:
            # Non-source processes still clear their index
            if offload_dir is not None:
                from pathlib import Path

                offload_dir = str(Path(offload_dir).resolve())

            for offloaded in list(cls.index.keys()):
                file_path = cls.index[offloaded]["safetensors_file"]
                if offload_dir is not None and not file_path.startswith(offload_dir):
                    continue
                del cls.index[offloaded]

            dist.barrier()
            return 0
