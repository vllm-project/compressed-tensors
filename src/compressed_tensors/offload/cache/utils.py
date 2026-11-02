# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import errno
import os
import sys
import tempfile
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


def share_staged_tensor(tensor: torch.Tensor, memo: dict, directory: str) -> tuple:
    """Copy a staged CPU tensor to named shared memory and describe its storage."""
    source = tensor.untyped_storage()
    nbytes = source.nbytes()
    shape = tuple(tensor.shape)
    stride = tuple(tensor.stride())
    storage_offset = tensor.storage_offset()
    if nbytes == 0:
        return (None, 0, tensor.dtype, shape, stride, storage_offset)

    key = (source.data_ptr(), nbytes)
    entry = memo.get(key)
    if entry is None:
        fd, filename = tempfile.mkstemp(prefix="weight-", dir=directory)
        try:
            # Reserve tmpfs pages now. A sparse ftruncate could SIGBUS later
            # during the tensor copy if /dev/shm fills up.
            os.posix_fallocate(fd, 0, nbytes)
            os.close(fd)
            fd = -1
            shared = torch.UntypedStorage.from_file(
                filename, shared=True, nbytes=nbytes
            )
            source_bytes = torch.empty(0, dtype=torch.uint8).set_(
                source, 0, (nbytes,), (1,)
            )
            shared_bytes = torch.empty(0, dtype=torch.uint8).set_(
                shared, 0, (nbytes,), (1,)
            )
            shared_bytes.copy_(source_bytes)
        except BaseException:
            if fd != -1:
                os.close(fd)
            os.unlink(filename)
            raise
        # Keep the source mapping alive until staging finishes, so a later
        # tensor cannot reuse its virtual address and collide with this key.
        memo[key] = (filename, shared, source)
    else:
        filename, shared, _ = entry

    with torch.no_grad():
        tensor.set_(shared, storage_offset, shape, stride)
    return (
        filename,
        nbytes,
        tensor.dtype,
        shape,
        stride,
        storage_offset,
    )


def attach_staged_tensor(
    template: torch.Tensor, descriptor: tuple, memo: dict
) -> torch.Tensor:
    """Attach to the source rank's storage while retaining local tensor attributes."""
    filename, nbytes, dtype, shape, stride, storage_offset = descriptor
    if nbytes == 0:
        return to_tensor(
            torch.empty_strided(shape, stride, dtype=dtype, device="cpu"), template
        )

    storage = memo.get(filename)
    if storage is None:
        storage = torch.UntypedStorage.from_file(filename, shared=True, nbytes=nbytes)
        memo[filename] = storage

    tensor = to_tensor(torch.empty(0, dtype=dtype, device="cpu"), template)
    with torch.no_grad():
        tensor.set_(storage, storage_offset, shape, stride)
    return tensor


def register_shared_storage(tensor: torch.Tensor) -> int | None:
    """Pin shared pages in this CUDA process without allocating a second buffer."""
    storage = tensor.untyped_storage()
    if storage.nbytes() == 0:
        return None

    pointer = storage.data_ptr()
    torch.cuda.check_error(
        torch.cuda.cudart().cudaHostRegister(pointer, storage.nbytes(), 1)
    )
    return pointer


def unregister_shared_storage(pointer: int) -> None:
    """Release a process-local CUDA registration after the blocking GPU onload."""
    torch.cuda.check_error(torch.cuda.cudart().cudaHostUnregister(pointer))
