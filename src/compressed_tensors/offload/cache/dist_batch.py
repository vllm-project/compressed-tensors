# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from abc import abstractmethod
from collections.abc import Hashable, Iterator, MutableMapping
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from compressed_tensors.distributed import (
    get_source_rank,
    is_distributed,
    is_source_process,
)
from compressed_tensors.offload.utils import send_tensors


if TYPE_CHECKING:
    from compressed_tensors.offload.cache.base import OffloadCache


__all__ = ["OffloadBatch", "batch_offload_sync", "BatchedOffloadMixin"]


class OffloadBatch:
    """
    Collects the offload metadata of distributed cpu/disk caches so that it can be
    exchanged between ranks once, instead of with a broadcast and barrier per tensor.

    On the source rank, tensors are offloaded immediately and their metadata is
    recorded. On other ranks, tensors are recorded and rebuilt from the source's
    metadata when the batch completes. Tensors on a device other than the offload
    device (e.g. an accelerator) are recorded as meta tensors, so that the batch does
    not keep their data alive. Entries are keyed by module and tensor name, so ranks
    which lack some modules (e.g. sharded experts) only rebuild their own.

    The source rank is fixed when the batch is created.
    """

    def __init__(self):
        self.source_rank = get_source_rank()
        self.is_source = dist.get_rank() == self.source_rank
        # name of the module currently being offloaded, set by the caller
        self.module_name: str = ""
        # keys recorded so far, to reject ambiguous (duplicate) keys
        self.keys: set[tuple[str, Hashable]] = set()
        # source rank: (module name, tensor name) -> metadata to rebuild the offload
        self.metadata: dict[tuple[str, Hashable], Any] = {}
        # other ranks: offloads waiting for the source's metadata
        self.pending: list[
            tuple["BatchedOffloadMixin", tuple[str, Hashable], Hashable, torch.Tensor]
        ] = []
        # other ranks: per-batch memo used by caches to share rebuilt storages
        self.memo: dict = {}

    def offload(
        self, cache: "BatchedOffloadMixin", name: Hashable, tensor: torch.Tensor | None
    ) -> torch.Tensor | None:
        """
        Offload a tensor as part of this batch

        :param cache: cache which the tensor belongs to
        :param name: name of the tensor within its cache
        :param tensor: tensor to offload
        :return: offloaded tensor on the source rank. On other ranks, a placeholder
            which is replaced when the batch completes
        """
        if tensor is None:
            return None

        key = (self.module_name, name)
        if key in self.keys:
            raise ValueError(
                f"Offload of `{name}` in module `{self.module_name}` was already "
                "recorded in this batch. Set `module_name` before offloading each "
                "module."
            )
        self.keys.add(key)

        if self.is_source:
            offloaded, self.metadata[key] = cache.offload_local(tensor)
            return offloaded

        # a tensor on a device other than the offload device (e.g. an accelerator) is
        # rebuilt as a new tensor either way, so keep only a meta copy of it (made the
        # way `recv_offload` copies tensors) and let its data be freed during dispatch.
        # Like other moved tensors, its offload doesn't require grad, also when its
        # dtype or shape differs from the source's. Tensors already on the offload
        # device are kept and rebuilt in place, as without batching
        if not (tensor.is_meta or str(tensor.device) == str(cache.offload_device)):
            tensor = send_tensors(tensor, device="meta")

        self.pending.append((cache, key, name, tensor))
        return tensor

    def complete(self):
        """
        Exchange all recorded metadata with a single broadcast, rebuild the pending
        offloads on non-source ranks, then synchronize once so that the source keeps
        its offloads alive until every rank has rebuilt them
        """
        payload = [self.metadata if self.is_source else None]
        dist.broadcast_object_list(payload, src=self.source_rank)

        if not self.is_source:
            metadata = payload[0]
            for cache, key, name, tensor in self.pending:
                if key not in metadata:
                    raise ValueError(
                        f"Rank {dist.get_rank()} offloaded `{key[1]}` of module "
                        f"`{key[0]}`, but the source rank did not"
                    )
                cache.offloaded_values[name] = cache.recv_offload(
                    tensor, metadata[key], memo=self.memo
                )

        dist.barrier()


_active_batch: Optional[OffloadBatch] = None


@contextlib.contextmanager
def batch_offload_sync() -> Iterator[Optional[OffloadBatch]]:
    """
    Context in which distributed cpu/disk caches created through `from_mapping` defer
    their metadata exchange until the context exits, so that offloading a whole model
    costs one broadcast and one barrier rather than one of each per tensor.

    Yields `None` (and changes nothing) when not distributed. Set `module_name` on
    the yielded batch before offloading each module. Every rank must enter and exit
    the context together. The context is meant for serialized use by a single
    dispatch, so it cannot be nested and is not thread-safe.

    Accelerator offloads are unaffected: they broadcast tensor data, not metadata.
    """
    global _active_batch

    if not is_distributed():
        yield None
        return

    if _active_batch is not None:
        raise RuntimeError(
            "`batch_offload_sync` cannot be nested: offloads in the inner context "
            "would join the outer batch under ambiguous names"
        )

    batch = _active_batch = OffloadBatch()
    try:
        yield batch
    finally:
        _active_batch = None

    batch.complete()


class BatchedOffloadMixin:
    """
    Shared offload logic for distributed caches whose offloads are rebuilt on other
    ranks from metadata produced by the source rank (cpu shared memory, disk files).

    Subclasses implement `offload_local` (source rank) and `recv_offload` (other ranks).
    Outside of `batch_offload_sync`, each offload is exchanged immediately.
    """

    offloaded_values: dict[Hashable, torch.Tensor]

    @abstractmethod
    def offload_local(self, tensor: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """
        Offload a tensor on the source rank without synchronizing

        :param tensor: tensor to offload
        :return: offloaded tensor and the metadata other ranks need to rebuild it
        """
        raise NotImplementedError()

    @abstractmethod
    def recv_offload(
        self, tensor: torch.Tensor, metadata: Any, memo: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Rebuild the source rank's offload on a non-source rank

        :param tensor: this rank's local tensor (often a meta tensor)
        :param metadata: metadata returned by `offload_local` on the source rank
        :param memo: optional per-batch memo for sharing rebuilt storages
        :return: offloaded tensor referring to the source rank's offload
        """
        raise NotImplementedError()

    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """
        Offload a tensor and immediately exchange its metadata with the other ranks

        :param tensor: tensor on any device
        :return: offloaded tensor
        """
        if tensor is None:
            return None

        payload = [None]
        if is_source_process():
            tensor, payload[0] = self.offload_local(tensor)

        dist.broadcast_object_list(payload, src=get_source_rank())

        if not is_source_process():
            tensor = self.recv_offload(tensor, payload[0])

        # ensure that the source does not free its offload before others rebuild it
        dist.barrier()
        return tensor

    @classmethod
    def from_mapping(
        cls,
        mapping: MutableMapping[Hashable, torch.Tensor | None],
        onload_device: torch.device | str,
        offload_device: Optional[torch.device | str] = None,
        **kwargs,
    ) -> "OffloadCache":
        """
        Same as `OffloadCache.from_mapping`, except that offloads join the active
        `batch_offload_sync` context, if any, rather than synchronizing per tensor
        """
        batch = _active_batch
        if batch is None:
            return super().from_mapping(  # type: ignore[misc]
                mapping, onload_device, offload_device, **kwargs
            )

        instance = cls(
            onload_device=onload_device, offload_device=offload_device, **kwargs
        )
        instance.offloaded_values = {
            name: batch.offload(instance, name, tensor)
            for name, tensor in mapping.items()
        }
        return instance
