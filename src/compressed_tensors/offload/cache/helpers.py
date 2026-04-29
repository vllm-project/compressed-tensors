# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from compressed_tensors.distributed import is_distributed


if TYPE_CHECKING:
    from compressed_tensors.offload.dispatch import DeviceMap


__all__ = ["validate_shm_segment_limit"]


def _get_shm_limit() -> int | None:
    paths = ("/proc/sys/kernel/shmmni", "/proc/sys/kernel/file-max")
    limits = []
    for path in paths:
        try:
            with open(path) as f:
                limits.append(int(f.read().strip()))
        except (OSError, ValueError):
            pass

    return min(limits) if limits else None


def _count_required_shm_segments(
    model: torch.nn.Module,
    device_map: "DeviceMap",
) -> int:
    from compressed_tensors.offload.convert.helpers import get_tensors

    count = 0
    for name, (_onload_device, offload_device) in device_map.items():
        if offload_device is None:
            continue
        if offload_device != "disk" and torch.device(offload_device).type == "cpu":
            module = model.get_submodule(name)
            count += sum(1 for _ in get_tensors(module, recurse=False))

    return count


def validate_shm_segment_limit(
    model: torch.nn.Module,
    device_map: "DeviceMap",
) -> None:
    if not is_distributed():
        return

    limit = _get_shm_limit()
    if limit is None:
        return

    required = _count_required_shm_segments(model, device_map)
    if required > limit:
        raise OSError(
            f"CPU offloading requires {required} shared memory segments, but the OS "
            f"limit is {limit}. Reduce the number of CPU-offloaded modules, increase "
            f"the OS shared memory limit (e.g. `sysctl -w kernel.shmmni={required}`), "
            f"or use disk offloading (`offload_device='disk'`)."
        )
