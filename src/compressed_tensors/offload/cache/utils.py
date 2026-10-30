# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import errno
import sys
from functools import wraps

import torch
from compressed_tensors.offload.cache.disk_utils import _opened
from compressed_tensors.offload.utils import _pin_memory, to_tensor
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


def load_disk_tensor_from_offload(
    offloaded: dict,
    device: str,
    template: torch.Tensor | None = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """Load a tensor, reading matching dtypes directly into pinned CPU memory.

    Dtype conversions use safetensors before pinning the result. Raw payloads
    are little-endian, so big-endian hosts must swap bytes after reading.
    """
    dtype = getattr(torch, offloaded["dtype"])
    file_path = offloaded["safetensors_file"]
    pin_memory = pin_memory and torch.accelerator.is_available()
    with _opened(file_path, device, raw=pin_memory) as (
        file,
        header,
        data_offset,
    ):
        name = offloaded["weight_name"]
        if header is None:
            onloaded = file.get_tensor(name).to(dtype)
        else:
            info = header[name]
            if str_to_torch_dtype.get(info["dtype"]) != dtype:
                with _opened(file_path, device) as (source_file, _, _):
                    onloaded = _pin_memory(source_file.get_tensor(name).to(dtype))
            else:
                onloaded = torch.empty(
                    info["shape"], dtype=dtype, device="cpu", pin_memory=True
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
                        raise EOFError(f"Unexpected end of file reading tensor {name}")
                    remaining = remaining[count:]
                if sys.byteorder == "big":
                    array.view(f"u{onloaded.element_size()}").byteswap(inplace=True)
    return to_tensor(onloaded, template) if template is not None else onloaded
