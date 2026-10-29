# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
from compressed_tensors.offload.cache.cpu import CPUCache
from compressed_tensors.offload.cache.dist_batch import BatchedOffloadMixin
from compressed_tensors.offload.cache.utils import catch_cpu_mem_error
from compressed_tensors.offload.utils import send_tensors, to_tensor


class DistributedCPUCache(BatchedOffloadMixin, CPUCache):
    """
    Handles offloading and onloading tensors from/to cpu memory shared across processes
    """

    @catch_cpu_mem_error
    def offload_local(self, tensor: torch.Tensor) -> tuple[torch.Tensor, list]:
        """
        Create shared cpu memory for offload on the source rank.

        :param tensor: tensor on any device
        :return: cpu tensor whose data is located in shared memory, and the shared
            memory handle, dtype and shape used by other ranks to rebuild it
        """
        # slight runtime cost for views
        tensor = tensor.contiguous()
        tensor = CPUCache.offload(self, tensor)

        # share by filename directly. Storage already shared by filename (e.g. a tied
        # weight offloaded twice) keeps its handle, whereas `share_memory_` would move
        # it into new shared memory under the default `file_descriptor` strategy,
        # releasing a file that other ranks may not have opened yet
        handle = tensor.untyped_storage()._share_filename_cpu_()
        return tensor, [*handle, tensor.dtype, tensor.shape]

    @catch_cpu_mem_error
    def recv_offload(
        self, tensor: torch.Tensor, metadata: list, memo: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Point a tensor at the shared cpu memory created by the source rank.

        :param tensor: this rank's local tensor (often a meta tensor)
        :param metadata: shared memory handle, dtype and shape from the source rank
        :param memo: optional per-batch memo. Tensors which shared a storage on the
            source rank share the reconstructed storage on this rank as well
        :return: cpu tensor whose data is located in shared memory
        """
        *handle, src_dtype, src_shape = metadata
        tensor = tensor.contiguous()

        # transformers may init params/buffers on non-source (meta) ranks with a
        # different dtype or shape than the checkpoint (e.g. `inv_freq`, or
        # tied/multimodal weights), so rebuild from the source's dtype and shape
        # before pointing at the shared storage. See
        # https://github.com/huggingface/transformers/pull/47486
        if tensor.is_meta or tensor.dtype != src_dtype or tensor.shape != src_shape:
            empty = torch.empty(src_shape, dtype=src_dtype, device=self.offload_device)
            tensor = to_tensor(empty, tensor)
        else:
            tensor = send_tensors(tensor, device=self.offload_device)

        # handle[1] is the shared memory filename
        storage = memo.get(handle[1]) if memo is not None else None
        if storage is None:
            storage = torch.UntypedStorage._new_shared_filename_cpu(*handle)
            if memo is not None:
                memo[handle[1]] = storage

        # reconstruct tensor from shared memory file handle
        with torch.no_grad():
            tensor.set_(
                storage,
                storage_offset=tensor.storage_offset(),
                size=tensor.size(),
                stride=tensor.stride(),
            )

        return tensor
