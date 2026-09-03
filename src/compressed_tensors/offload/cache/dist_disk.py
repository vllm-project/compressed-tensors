# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from compressed_tensors.distributed import get_source_rank, is_source_process
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.utils import send_tensors, to_empty


class DistributedDiskCache(DiskCache):
    """
    Handles offloading and onloading tensors from/to disk. For more information, see
    `compressed_tensors.offload.cache.disk_cache::DiskCache`.
    """

    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """
        Synchronously write tensor data to disk.

        The dtype of ``tensor`` on non-source ranks cannot be trusted because
        transformers may initialize buffers (e.g. ``inv_freq``) with a
        different dtype than the checkpoint value on the source rank. See
        https://github.com/huggingface/transformers/pull/47486

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

        dist.broadcast_object_list(broadcast_obj, src=get_source_rank())

        if not is_source_process():
            src_dtype = getattr(torch, broadcast_obj[2])
            if offloaded.dtype != src_dtype:
                offloaded = to_empty(offloaded, device="meta", dtype=src_dtype)
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
        if is_source_process():
            super().__delitem__(key)
        else:
            if not self.onloading_disabled:
                offloaded = self.offloaded_values[key]
                del self.index[offloaded]
            super(DiskCache, self).__delitem__(key)
