# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os

import torch
import torch.distributed as dist


__all__ = ["is_distributed", "is_rank0"]


def is_rank0() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_dist():
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


_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)


@contextlib.contextmanager
def as_broadcastable(tensor: torch.Tensor):
    """
    Context manager that yields a NCCL-safe view of a tensor for broadcast.
    NCCL does not support FP8 dtypes on pre-sm90 GPUs, so FP8 tensors are
    viewed as uint8 (same byte width) for the broadcast. Since the view shares
    storage, the broadcast writes directly into the original tensor's memory.

    Usage::

        with as_broadcastable(tensor) as t:
            dist.broadcast(t, src=0)
        # tensor is still FP8 with the broadcasted data

    :param tensor: tensor to broadcast
    """
    if tensor.dtype in _FP8_DTYPES:
        yield tensor.data.view(torch.uint8)
    else:
        yield tensor
