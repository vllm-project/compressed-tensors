# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import chain
from typing import Iterable

import torch
import torch.distributed as dist
from compressed_tensors.offload.dist_utils import is_distributed


__all__ = ["get_tensors", "norm_device"]


def norm_device(device: str | torch.device | None) -> str | torch.device | None:
    if device not in ("disk", None):
        device = torch.device(device)

    if (
        is_distributed()
        and isinstance(device, torch.device)
        and device.index == dist.get_rank()
    ):
        device = torch.device(type=device.type, index=None)

    return device


def get_tensors(
    module: torch.nn.Module, recurse: bool = False
) -> Iterable[tuple[str, torch.Tensor | None]]:
    return chain(
        module.named_parameters(recurse=recurse), module.named_buffers(recurse=recurse)
    )
