# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import errno
import sys
from functools import wraps

import torch
from compressed_tensors.offload.cache.disk_utils import _opened_raw
from compressed_tensors.offload.utils import _pin_memory, send_tensors, to_tensor
from compressed_tensors.utils.safetensors_load import str_to_torch_dtype
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


def _empty_pinned(shape, dtype: torch.dtype) -> torch.Tensor:
    """Allocate page-locked storage for a direct read."""
    return torch.empty(shape, dtype=dtype, pin_memory=True)


def _load_raw(
    offloaded: dict, dtype: torch.dtype, pin_memory: bool = False
) -> torch.Tensor:
    """
    Read a disk-offloaded tensor's payload into host memory.

    A matching dtype is read directly into page-locked memory when requested;
    otherwise the payload is read in the on-disk dtype, cast, and pinned. Raw
    payloads are little-endian, so big-endian hosts must swap bytes after
    reading.
    """
    file_path = offloaded["safetensors_file"]
    name = offloaded["weight_name"]
    with _opened_raw(file_path) as (file, header, data_offset):
        info = header[name]
        disk_dtype = str_to_torch_dtype.get(info.get("dtype"))
        if disk_dtype is None:
            raise ValueError(
                f"Unsupported dtype {info.get('dtype')!r} for tensor {name} "
                f"in {file_path}"
            )

        direct = pin_memory and disk_dtype == dtype
        onloaded = (
            _empty_pinned(info["shape"], dtype)
            if direct
            else torch.empty(info["shape"], dtype=disk_dtype)
        )
        start, end = info["data_offsets"]
        if onloaded.nbytes != end - start:
            raise ValueError(f"Invalid byte length for tensor {name}")

        array = onloaded.reshape(-1).view(torch.uint8).numpy()
        remaining = memoryview(array)
        file.seek(data_offset + start)
        while remaining:
            count = file.readinto(remaining)
            if not count:
                raise EOFError(
                    f"Unexpected end of file reading tensor {name} from {file_path}"
                )
            remaining = remaining[count:]

        if sys.byteorder == "big" and onloaded.element_size() > 1:
            array.view(f"u{onloaded.element_size()}").byteswap(inplace=True)

        if not direct:
            if disk_dtype != dtype:
                onloaded = onloaded.to(dtype)
            if pin_memory:
                onloaded = _pin_memory(onloaded)
    return onloaded


def load_disk_tensor_from_offload(
    offloaded: dict,
    device: str,
    template: torch.Tensor | None = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """
    Load a tensor by reading its payload directly from the offload file.

    Matching dtypes are read into page-locked CPU memory when requested; dtype
    conversions read the on-disk dtype, cast, then pin. Raw payloads are
    little-endian, so big-endian hosts must swap bytes after reading.
    """
    dtype = getattr(torch, offloaded["dtype"])
    pin_memory = pin_memory and torch.accelerator.is_available()

    onloaded = _load_raw(offloaded, dtype, pin_memory=pin_memory)
    if torch.device(device).type != "cpu":
        onloaded = send_tensors(onloaded, device=device)

    return to_tensor(onloaded, template) if template is not None else onloaded
