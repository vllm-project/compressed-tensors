# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import chain
from typing import Iterable, Literal

import torch
import torch.distributed as dist
from compressed_tensors.distributed import is_distributed
from compressed_tensors.utils import is_accelerator_type


__all__ = [
    "get_tensors",
    "norm_device",
    "DEFAULT_OFFLOAD_DEVICE",
]


DEFAULT_OFFLOAD_DEVICE = torch.device("cpu")


def norm_device(
    device: str | torch.device | None,
) -> torch.device | Literal["disk"] | None:
    """
    Standardize the representation of devices for the purposes of consistency

    - when running with distributed, represent rank device as the accelerator type
    - when not running with distributed, bare accelerator type (e.g. "cuda") is
      resolved to index 0
    """
    if device in ("disk", None):
        return device

    device = torch.device(device)

    # (dist) "cuda:R" / "xpu:R" -> bare type
    if is_distributed() and device.index == dist.get_rank():
        device = torch.device(type=device.type, index=None)

    # (non-dist) bare accelerator type -> index 0
    if (
        not is_distributed()
        and is_accelerator_type(device.type)
        and device.index is None
    ):
        device = torch.device(type=device.type, index=0)

    return device


def get_tensors(
    module: torch.nn.Module, recurse: bool = False
) -> Iterable[tuple[str, torch.Tensor | None]]:
    return chain(
        module.named_parameters(recurse=recurse), module.named_buffers(recurse=recurse)
    )
