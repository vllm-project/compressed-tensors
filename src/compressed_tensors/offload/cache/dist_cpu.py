# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed as dist
from compressed_tensors.distributed import get_source_rank, is_source_process
from compressed_tensors.offload.cache.cpu import CPUCache
from compressed_tensors.offload.cache.utils import catch_cpu_mem_error
from compressed_tensors.offload.utils import send_tensors, to_tensor


class DistributedCPUCache(CPUCache):
    """
    Handles offloading and onloading tensors from/to cpu memory shared across processes
    """

    @catch_cpu_mem_error
    def offload_local(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """
        Create a shared-memory cpu tensor *without* synchronizing across ranks.

        This is only meaningful on the source rank. The resulting storage can be
        mapped by other ranks using the metadata returned by `get_offload_meta`.
        Separating this step from the rank synchronization allows many tensors to
        be offloaded before performing a single, batched metadata exchange (see
        `compressed_tensors.offload.dispatch::dispatch_with_map`).

        :param tensor: tensor on any device
        :return: cpu tensor whose data is located in shared memory
        """
        if tensor is None:
            return None

        # slight runtime cost for views
        tensor = tensor.contiguous()
        return super().offload(tensor).share_memory_()

    @staticmethod
    def get_offload_meta(offloaded: torch.Tensor | None) -> list[Any] | None:
        """
        Extract the shared-memory file handle of an offloaded tensor so that it
        can be reconstructed on another rank via `recv_offload`.

        :param offloaded: shared-memory cpu tensor produced by `offload_local`
        :return: picklable metadata describing the shared storage, or None
        """
        if offloaded is None:
            return None

        handle, filename, nbytes = offloaded.untyped_storage()._share_filename_cpu_()
        return [handle, filename, nbytes, offloaded.dtype, offloaded.shape]

    def recv_offload(
        self, tensor: torch.Tensor | None, meta: list[Any] | None
    ) -> torch.Tensor | None:
        """
        Reconstruct an offloaded tensor from shared-memory metadata produced by
        the source rank. Only meaningful on non-source ranks.

        :param tensor: the receiving rank's local param/buffer, used to preserve
            tensor subclass and attributes. May be None
        :param meta: metadata produced by `get_offload_meta` on the source rank
        :return: cpu tensor pointing at the source rank's shared storage
        """
        if meta is None:
            return None

        handle, filename, nbytes, src_dtype, src_shape = meta

        # transformers may init params/buffers on non-source (meta) ranks with a
        # different dtype or shape than the checkpoint (e.g. `inv_freq`, or
        # tied/multimodal weights), so rebuild from the source's dtype and shape
        # before pointing at the shared storage. See
        # https://github.com/huggingface/transformers/pull/47486
        if (
            tensor is None
            or tensor.is_meta
            or tensor.dtype != src_dtype
            or tensor.shape != src_shape
        ):
            empty = torch.empty(src_shape, dtype=src_dtype, device=self.offload_device)
            tensor = empty if tensor is None else to_tensor(empty, tensor)
        else:
            tensor = send_tensors(tensor, device=self.offload_device)

        # reconstruct tensor from shared memory file handle
        with torch.no_grad():
            tensor.set_(
                torch.UntypedStorage._new_shared_filename_cpu(handle, filename, nbytes),
                storage_offset=tensor.storage_offset(),
                size=tensor.size(),
                stride=tensor.stride(),
            )

        return tensor

    @catch_cpu_mem_error
    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """
        Synchronously create shared cpu memory for offload.

        Prefer `dispatch_with_map` for offloading a whole model, which batches
        the metadata exchange below across all tensors. This per-tensor path is
        composed from `offload_local`, `get_offload_meta`, and `recv_offload`.

        :param tensor: tensor on any device
        :return: cpu tensor whose data is located in shared memory
        """
        if tensor is None:
            return None

        if is_source_process():
            # create shared memory cpu tensor
            offloaded = self.offload_local(tensor)
            broadcast_obj = [self.get_offload_meta(offloaded)]
        else:
            broadcast_obj = [None]

        # receive shared memory file handle
        dist.broadcast_object_list(broadcast_obj, src=get_source_rank())

        if not is_source_process():
            # reconstruct tensor from shared memory file handle
            offloaded = self.recv_offload(tensor, broadcast_obj[0])

        # ensure that rank 0 does not garbage collect before other ranks reconstruct
        dist.barrier()

        return offloaded
