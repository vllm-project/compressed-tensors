# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
import torch.distributed as dist


__all__ = ["is_distributed", "is_rank0"]


def is_rank0() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_dist():
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


_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)


def as_broadcastable(tensor: torch.Tensor) -> torch.Tensor:
    """Return a view of the tensor that is compatible with ``dist.broadcast``.

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
