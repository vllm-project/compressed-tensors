# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import errno
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


def _is_cpu_memory_error(exc: BaseException) -> bool:
    errno_value = getattr(exc, "errno", None)
    if errno_value == errno.ENOMEM:
        return True
    message = str(exc).lower()
    return any(kw in message for kw in _CPU_MEMORY_KEYWORDS)


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
