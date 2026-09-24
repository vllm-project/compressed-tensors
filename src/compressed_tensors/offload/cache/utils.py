# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import errno
import torch
from functools import wraps

from loguru import logger


_CPU_MEMORY_KEYWORDS = (
    "defaultcpuallocator",
    "can't allocate memory",
    "cannot allocate memory",
    "failed to allocate",
    "out of memory",
    "mmap",
    "shm_open",
)

_CPU_MEMORY_REMEDIATION = (
    "CPU offloading ran out of host RAM or mmap descriptors. "
    "Switch to disk offloading (`offload_device='disk'`) or "
    "increase the OS mmap limit."
)

_PINNED_MEMORY_REMEDIATION = (
    "Pinned-memory staging ran out of host RAM or page-locked memory. "
    "This can surface as a CUDA/XPU OOM even though the failure is on the CPU side. "
    "Try disabling `pin_memory` or switching to disk offloading."
)


def _is_cpu_memory_error(exc: BaseException) -> bool:
    errno_value = getattr(exc, "errno", None)
    if errno_value == errno.ENOMEM:
        return True
    message = str(exc).lower()
    return any(kw in message for kw in _CPU_MEMORY_KEYWORDS)


def _is_pinned_memory_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        kw in message
        for kw in (
            "cuda out of memory",
            "xpu out of memory",
            "hip out of memory",
            "out of memory",
            "failed to allocate",
            "cannot allocate memory",
            "can't allocate memory",
        )
    )


def catch_cpu_mem_error(func):
    """
    Decorator to catch CPU memory errors and log a remediation warning.
    Prevents duplicate logs if nested functions also use this decorator.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (RuntimeError, OSError) as exc:
            # Prevent duplicate logs when DistributedCPUCache calls super().offload()
            if getattr(exc, "_cpu_mem_logged", False):
                raise

            if _is_cpu_memory_error(exc):
                logger.warning(_CPU_MEMORY_REMEDIATION)
                exc._cpu_mem_logged = True
            raise

    return wrapper


def catch_pinned_mem_error(func):
    """
    Decorator to catch failures from page-locked staging and log a remediation warning.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        pin_memory = kwargs.get("pin_memory", False)
        if len(args) >= 3:
            pin_memory = args[2]

        try:
            return func(*args, **kwargs)
        except (RuntimeError, OSError) as exc:
            if getattr(exc, "_cpu_mem_logged", False):
                raise

            if pin_memory and _is_pinned_memory_error(exc):
                logger.warning(_PINNED_MEMORY_REMEDIATION)
                exc._cpu_mem_logged = True
            raise

    return wrapper

def load_disk_tensor_from_offload(offloaded: dict, device: str) -> torch.Tensor:
    with safe_open(offloaded["safetensors_file"], framework="pt", device=device) as file:
        onloaded = file.get_tensor(offloaded["weight_name"])
        onloaded = to_tensor(onloaded)
        onloaded = onloaded.to(getattr(torch, offloaded["dtype"]))
        return onloaded