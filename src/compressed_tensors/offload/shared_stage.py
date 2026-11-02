# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared CPU storage and CUDA registration for distributed disk staging."""

import os
import tempfile
import threading

import torch
from compressed_tensors.offload.utils import to_tensor

_registration_lock = threading.Lock()
_registrations: dict[int, int] = {}


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
    with _registration_lock:
        if pointer in _registrations:
            _registrations[pointer] += 1
        else:
            # cudaHostRegisterPortable makes the registration usable by any CUDA
            # context in this process. Each rank registers its own virtual mapping.
            torch.cuda.check_error(
                torch.cuda.cudart().cudaHostRegister(pointer, storage.nbytes(), 1)
            )
            _registrations[pointer] = 1
    return pointer


def unregister_shared_storage(pointer: int) -> None:
    """Release a process-local CUDA registration after the blocking GPU onload."""
    with _registration_lock:
        count = _registrations[pointer]
        if count == 1:
            torch.cuda.check_error(torch.cuda.cudart().cudaHostUnregister(pointer))
            del _registrations[pointer]
        else:
            _registrations[pointer] = count - 1
