# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
from compressed_tensors.distributed import is_source_process
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.cache.dist_batch import BatchedOffloadMixin
from compressed_tensors.offload.utils import send_tensors, to_tensor


class DistributedDiskCache(BatchedOffloadMixin, DiskCache):
    """
    Handles offloading and onloading tensors from/to disk. For more information, see
    `compressed_tensors.offload.cache.disk_cache::DiskCache`.
    """

    def offload_local(self, tensor: torch.Tensor) -> tuple[torch.Tensor, list]:
        """
        Write tensor data to disk on the source rank.

        :param tensor: tensor on any device
        :return: meta tensor representing the disk offloaded tensor, and the file
            location, dtype and shape used by other ranks to rebuild it
        """
        offloaded = DiskCache.offload(self, tensor)
        weight_info = self.index[offloaded]
        return offloaded, [
            weight_info["safetensors_file"],
            weight_info["weight_name"],
            weight_info["dtype"],
            offloaded.shape,
        ]

    def recv_offload(
        self, tensor: torch.Tensor, metadata: list, memo: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Point a meta tensor at the file written by the source rank.

        :param tensor: this rank's local tensor (often a meta tensor)
        :param metadata: file location, dtype and shape from the source rank
        :param memo: unused, accepted for interface compatibility
        :return: meta tensor representing the disk offloaded tensor
        """
        safetensors_file, weight_name, dtype, src_shape = metadata
        offloaded = send_tensors(tensor, device="meta")
        src_dtype = getattr(torch, dtype)

        # transformers may init params/buffers on non-source (meta) ranks with a
        # different dtype or shape than the checkpoint (e.g. `inv_freq`, or
        # tied/multimodal weights), so rebuild the meta tensor to match the
        # source. See https://github.com/huggingface/transformers/pull/47486
        if offloaded.dtype != src_dtype or offloaded.shape != src_shape:
            empty = torch.empty(src_shape, dtype=src_dtype, device="meta")
            offloaded = to_tensor(empty, offloaded)

        self.index[offloaded] = {
            "safetensors_file": safetensors_file,
            "weight_name": weight_name,
            "dtype": dtype,
        }
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
