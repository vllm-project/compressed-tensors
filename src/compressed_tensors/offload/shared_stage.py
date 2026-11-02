# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared CPU storage and CUDA registration for distributed disk staging."""

import threading

import torch
from compressed_tensors.offload.utils import to_tensor

_registration_lock = threading.Lock()
_registrations: dict[int, int] = {}


def share_staged_tensor(tensor: torch.Tensor) -> tuple:
    """Move a staged CPU tensor to shared memory and describe its storage."""
    handle = tensor.untyped_storage()._share_filename_cpu_()
    return (
        *handle,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
    )


def attach_staged_tensor(
    template: torch.Tensor, descriptor: tuple, memo: dict
) -> torch.Tensor:
    """Attach to the source rank's storage while retaining local tensor attributes."""
    *handle, dtype, shape, stride, storage_offset = descriptor
    filename = handle[1]
    storage = memo.get(filename)
    if storage is None:
        storage = torch.UntypedStorage._new_shared_filename_cpu(*handle)
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
