# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed as dist
from compressed_tensors.distributed import get_source_rank, is_source_process
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.utils import send_tensors, to_tensor


class DistributedDiskCache(DiskCache):
    """
    Handles offloading and onloading tensors from/to disk. For more information, see
    `compressed_tensors.offload.cache.disk_cache::DiskCache`.
    """

    def offload_local(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """
        Write tensor data to disk *without* synchronizing across ranks.

        This is only meaningful on the source rank. The written file is described
        by the metadata returned from `get_offload_meta`, which other ranks use to
        index the same file via `recv_offload`. Separating this step from rank
        synchronization allows many tensors to be offloaded before performing a
        single, batched metadata exchange (see
        `compressed_tensors.offload.dispatch::dispatch_with_map`).

        :param tensor: tensor on any device
        :return: meta tensor representing the disk offloaded parameter
        """
        return super().offload(tensor)

    def get_offload_meta(self, offloaded: torch.Tensor | None) -> list[Any] | None:
        """
        Extract the disk location of an offloaded tensor so that it can be
        indexed on another rank via `recv_offload`.

        :param offloaded: meta tensor produced by `offload_local`
        :return: picklable metadata describing the disk file, or None
        """
        if offloaded is None:
            return None

        entry = self.index[offloaded]
        return [
            entry["safetensors_file"],
            entry["weight_name"],
            entry["dtype"],
            offloaded.shape,
        ]

    def recv_offload(
        self, tensor: torch.Tensor | None, meta: list[Any] | None
    ) -> torch.Tensor | None:
        """
        Index a disk-offloaded tensor from metadata produced by the source rank.
        Only meaningful on non-source ranks.

        :param tensor: the receiving rank's local param/buffer, used to preserve
            tensor subclass and attributes. May be None
        :param meta: metadata produced by `get_offload_meta` on the source rank
        :return: meta tensor whose disk location is registered in `index`
        """
        if meta is None:
            return None

        safetensors_file, weight_name, dtype_str, src_shape = meta
        src_dtype = getattr(torch, dtype_str)

        offloaded = send_tensors(tensor, device="meta")

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
            "dtype": dtype_str,
        }
        return offloaded

    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """
        Synchronously write tensor data to disk.

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
                offloaded.shape,
            ]
        else:
            offloaded = send_tensors(tensor, device="meta")
            broadcast_obj = [None, None, None, None]

        dist.broadcast_object_list(broadcast_obj, src=get_source_rank())

        if not is_source_process():
            src_dtype = getattr(torch, broadcast_obj[2])
            src_shape = broadcast_obj[3]

            # transformers may init params/buffers on non-source (meta) ranks with a
            # different dtype or shape than the checkpoint (e.g. `inv_freq`, or
            # tied/multimodal weights), so rebuild the meta tensor to match the
            # source. See https://github.com/huggingface/transformers/pull/47486
            if offloaded.dtype != src_dtype or offloaded.shape != src_shape:
                empty = torch.empty(src_shape, dtype=src_dtype, device="meta")
                offloaded = to_tensor(empty, offloaded)

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
