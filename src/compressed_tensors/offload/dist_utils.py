# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os

import torch
import torch.distributed as dist
from compressed_tensors.utils.helpers import deprecated, patch_attr


__all__ = [
    "is_distributed",
    "is_rank0",
    "init_dist",
    "as_single_threaded",
    "set_main_process",
    "is_main_process",
    "as_broadcastable",
]


SRC_RANK = 0


@deprecated("is_main_process")
def is_rank0() -> bool:
    """
    Check if the current process is rank 0 in distributed training.

    :return: True if not distributed or if current rank is 0, False otherwise
    """
    return not is_distributed() or dist.get_rank() == 0


def is_distributed() -> bool:
    """
    Check if PyTorch distributed training is available and initialized.

    :return: True if distributed backend is available and initialized, False otherwise
    """
    return dist.is_available() and dist.is_initialized()


def init_dist():
    """
    Initialize PyTorch distributed training using torchrun environment variables.

    This function sets up the NCCL backend for distributed training using
    environment variables set by torchrun. It configures the current process
    with its rank, local rank, and world size, and synchronizes all processes
    with a barrier.

    :raises ValueError: if TORCHELASTIC_RUN_ID is not found in environment,
        indicating the script was not launched with torchrun
    """
    if "TORCHELASTIC_RUN_ID" not in os.environ:
        raise ValueError(
            "Cannot find distributed environment. "
            "Please make sure you are using `torchrun --nproc-per-node ...`."
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    dist.barrier()


@contextlib.contextmanager
def as_single_threaded():
    """
    Context manager to temporarily use single-threaded offload methods.

    This context manager patches distributed cache classes to use their
    non-distributed counterparts' offload methods. This is useful when
    operations need to be performed without distributed coordination.

    Example:
        >>> with as_single_threaded():
        ...     # Operations here use single-threaded offload
        ...     cache.offload(data)
    """
    from compressed_tensors.offload.cache import (
        CPUCache,
        DeviceCache,
        DiskCache,
        DistributedCPUCache,
        DistributedDeviceCache,
        DistributedDiskCache,
    )

    with (
        patch_attr(DistributedDeviceCache, "offload", DeviceCache.offload),
        patch_attr(DistributedCPUCache, "offload", CPUCache.offload),
        patch_attr(DistributedDiskCache, "offload", DiskCache.offload),
    ):
        yield


@contextlib.contextmanager
def set_main_process(src_rank: int):
    """
    Context manager to temporarily designate a different rank as the main process.

    This allows temporarily changing which rank is considered the "main" process
    for operations that should only be performed by one process. The original
    main process rank is restored when exiting the context.

    :param src_rank: the rank to designate as the main process within the context

    Example:
        >>> with set_main_process(2):
        ...     if is_main_process():
        ...         # Only rank 2 executes this
        ...         print("I'm the temporary main process")
    """
    global SRC_RANK

    restore_rank, SRC_RANK = SRC_RANK, src_rank
    yield
    SRC_RANK = restore_rank


def is_main_process() -> bool:
    """
    Check if the current process is the designated main process.

    The main process is determined by SRC_RANK (default 0) and can be
    temporarily changed using the set_main_process context manager.

    :return: True if not distributed or if current rank equals SRC_RANK, False otherwise
    """
    return not is_distributed() or dist.get_rank() == SRC_RANK


_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)


def as_broadcastable(tensor: torch.Tensor) -> torch.Tensor:
    """
    Return a view of the tensor that is compatible with ``dist.broadcast``.

    NCCL does not support broadcasting FP8 dtypes on hardware without sm_90
    (Hopper or later). This function works around the limitation by viewing FP8
    tensors as ``uint8``, which NCCL can broadcast on any hardware. Non-FP8
    tensors are returned unchanged.

    :param tensor: the tensor to prepare for broadcasting
    :return: the original tensor, or a ``uint8`` view if the dtype is FP8
    """
    if tensor.dtype in _FP8_DTYPES:
        return tensor.data.view(torch.uint8)
    else:
        return tensor
