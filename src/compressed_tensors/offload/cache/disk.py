# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import torch
import torch.distributed as dist
from compressed_tensors.distributed import is_source_process
from compressed_tensors.logger import logger
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.cache.disk_utils import _evict
from compressed_tensors.offload.cache.utils import (
    catch_pinned_mem_error,
    load_disk_tensor_from_offload,
)
from compressed_tensors.offload.cache.writeback import AsyncDiskWriteback
from compressed_tensors.offload.utils import _pin_memory, send_tensors
from compressed_tensors.utils import is_accelerator_type
from safetensors.torch import save_file


if TYPE_CHECKING:
    from torch._prims_common import DeviceLikeType


class DiskCache(OffloadCache):
    """
    Handles offloading and onloading tensors from/to disk.

    Tensors usually start as a key in safetensors file, converted by (TODO NAME).
    New or updated tensors are written to new safetensors files in `offload_dir`.

    Tensors are stored in memory as meta tensors. The mapping between offloaded meta
    tensors and their locations on disk is defined by `index`.
    """

    offload_device = "disk"

    # offloaded tensors -> weight info
    index: dict[torch.Tensor, dict[str, str]] = dict()

    # directory where new tensors are written to
    offload_dir: str
    async_writer: AsyncDiskWriteback | None
    _ct_file_prefix = "ct_disk_cache"

    def __init__(
        self,
        onload_device: torch.device,
        offload_device: Optional["DeviceLikeType | Literal['disk']"] = None,
        offload_dir: Optional[str] = None,
        async_writer: AsyncDiskWriteback | None = None,
    ):
        super().__init__(onload_device, offload_device=offload_device)
        if offload_device is not None:
            assert str(offload_device) == str(self.offload_device)

        if offload_dir is None:
            raise ValueError(
                "Must provide an `offload_dir` to perform disk offloading "
                "(add `offload_folder` argument to `from_pretrained`)"
            )
        # Resolve relative paths to absolute paths for symlink creation
        self.offload_dir = Path(offload_dir).resolve()
        self.async_writer = async_writer

    @catch_pinned_mem_error
    def stage(
        self,
        offloaded: torch.Tensor | None,
        pin_memory: bool = False,
    ) -> torch.Tensor | None:
        """
        Stage a disk-backed tensor for a later onload.

        :param offloaded: meta tensor to stage
        :param pin_memory: whether to use page-locked CPU memory
        :return: staged tensor
        """
        if offloaded is None:
            return None

        weight_info = self.index[offloaded]
        # if there is a pending write, we have to wait for it
        # to finish or else we get stale data
        self._wait_for_write(weight_info["safetensors_file"])

        staged = load_disk_tensor_from_offload(
            weight_info, device="cpu", template=offloaded
        )
        # direct disk --> pinned memory is a bit complicated,
        # leave this for a future change. For now, copy to cpu first
        staged = (
            _pin_memory(staged)
            if (pin_memory and self.onload_device != "cpu")
            else staged
        )
        # don't transfer to pinned if onload_device is cpu

        return staged

    def onload(self, offloaded: torch.Tensor | None) -> torch.Tensor | None:
        """
        Onload a tensor from disk/meta to device

        :param offloaded: meta tensor to onload
        :return: device tensor, read from disk
        """
        if offloaded is None:
            return None

        device = _get_safe_open_device(self.onload_device)

        if self.is_staged:
            onloaded = send_tensors(offloaded, device=device, copy=False)
        else:
            weight_info = self.index[offloaded]
            # same as in `stage`, need to wait for writes to finish
            # or else stale data
            self._wait_for_write(weight_info["safetensors_file"])
            onloaded = load_disk_tensor_from_offload(
                weight_info, device=device, template=offloaded
            )

        return onloaded

    def offload(
        self, tensor: torch.Tensor | None, offloaded: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        """
        Offload a tensor to disk by writing a new safetensors file

        :param tensor: tensor on any device
        :param offloaded: optional meta tensor used to look up an existing file
        :return: meta tensor representing the offloaded tensor
        """
        if tensor is None:
            return None

        if tensor.device.type == "meta":
            assert tensor in self.index
            return tensor

        if offloaded is None:
            offloaded = send_tensors(tensor, device="meta")

        if tensor.dtype != offloaded.dtype:
            logger.bind(log_once=True).warning(
                f"Dtype mismatch during offload: tensor dtype {tensor.dtype} "
                f"does not match offloaded meta tensor dtype {offloaded.dtype}"
            )

        file_path = self._get_ct_file_path(self.offload_dir, offloaded)
        self.index[offloaded] = {
            "safetensors_file": file_path,
            "weight_name": "weight",
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }

        assert self._is_ct_file_path(file_path), f"Attempted to write to {file_path}"
        # safetensors requires contiguous tensors; compressed/packed weights (e.g.
        # NVFP4, FP8 block) may be non-contiguous views after compression.
        if (
            self.async_writer is not None
            and self.async_writer.is_batch_active
            and not (dist.is_available() and dist.is_initialized())
        ):
            # Snapshot to pageable CPU memory before returning. This synchronizes
            # the device-to-host copy and releases the caller's accelerator tensor
            # when its module mapping is replaced, while disk serialization runs
            # independently on the writeback worker.
            cpu_tensor = tensor.detach().contiguous().to(device="cpu", copy=True)
            self.async_writer.submit(file_path, {"weight": cpu_tensor})
        else:
            save_file({"weight": tensor.contiguous()}, file_path)
        return offloaded

    def __delitem__(self, key: str):
        """
        Remove the offload associated with `key`. If a new file was created to store
        updated tensor data, that new tensor data file is deleted.

        Any references to onloaded tensors held by this class are invalidated.

        :param key: name of tensor to invalidate
        """
        offloaded = self.offloaded_values[key]
        if not self.onloading_disabled:
            file_path = self.index[offloaded]["safetensors_file"]
            if self._is_ct_file_path(file_path):
                self._wait_for_write(file_path)
                os.remove(file_path)
            del self.index[offloaded]
        super().__delitem__(key)

    def update_offload(self, offloaded: torch.Tensor, data: torch.Tensor | None):
        """
        Write new param data to file that already exists.

        :param offloaded: meta tensors representating parameter to update
        :param data: new data
        """
        # get weight info from index
        assert offloaded in self.index, "Cannot find offload to update"
        weight_info = self.index[offloaded]
        file_path = weight_info["safetensors_file"]
        self._wait_for_write(file_path)
        weight_name = weight_info["weight_name"]
        dtype = getattr(torch, weight_info["dtype"])

        # create new file if old file was a symlink to a checkpoint file
        if os.path.islink(file_path):
            assert self._is_ct_file_path(file_path), f"Attempted to remove {file_path}"
            os.unlink(file_path)
        else:
            # rewriting in place, so drop any handle still reading the old contents
            _evict(file_path)

        # save with data using original weight_name
        assert self._is_ct_file_path(file_path), f"Attempted to write to {file_path}"
        save_file({weight_name: data.reshape_as(offloaded).to(dtype=dtype)}, file_path)

    def _wait_for_write(self, file_path: str) -> None:
        if self.async_writer is not None:
            self.async_writer.wait(file_path)

    @classmethod
    def create_checkpoint_symlink(
        cls,
        offloaded: torch.Tensor,
        weight_info: dict,
        offload_dir: str | os.PathLike | None,
    ) -> None:
        assert (
            is_source_process()
        ), "Must call on rank 0 to avoid id collisions between ranks"
        if offload_dir is None:
            raise ValueError(
                "Must provide an `offload_dir` to perform disk offloading "
                "(add `offload_folder` argument to `from_pretrained`)"
            )

        # Warn if dtype mismatch between offloaded meta tensor and weight_info
        weight_info_dtype = getattr(torch, weight_info["dtype"])
        if offloaded.dtype != weight_info_dtype:
            logger.bind(log_once=True).warning(
                f"Dtype mismatch during create_checkpoint_symlink: offloaded meta "
                f"tensor dtype {offloaded.dtype} does not match weight_info dtype "
                f"{weight_info_dtype}."
            )

        # Resolve relative paths to absolute paths for symlink creation
        source_path = Path(weight_info["safetensors_file"]).resolve()
        file_path = cls._get_ct_file_path(offload_dir, offloaded)

        os.symlink(source_path, file_path)
        cls.index[offloaded] = {
            "safetensors_file": file_path,
            "weight_name": weight_info["weight_name"],
            "dtype": weight_info["dtype"],
        }

    @classmethod
    def _is_ct_file_path(cls, file_path: str) -> bool:
        """Only write and delete files that DiskCache has created"""
        return os.path.basename(file_path).startswith(cls._ct_file_prefix)

    @classmethod
    def _get_ct_file_path(cls, offload_dir: str, offloaded: torch.Tensor) -> str:
        """Create file path with a prefix marking it as modifiable"""
        file_name = f"{cls._ct_file_prefix}_{_get_rank()}_{id(offloaded)}.safetensors"
        return os.path.join(offload_dir, file_name)


def _get_safe_open_device(device: "DeviceLikeType") -> str:
    """
    `safetensors.safe_open` does not accept `torch.device` as argument, so
    we must convert from torch.device to a string, while considering accelerator
    device index resolution.

    :param device: torch device to convert
    :return: device string for `safetensors.safe_open`
    """
    device = torch.device(device)
    if is_accelerator_type(device.type):
        # TODO: check if this case can be applied for all non-index accelerators
        if device.type == "mps":
            return f"{device.type}"

        if device.index is None:
            index = torch.accelerator.current_device_index()
        else:
            index = device.index
        return f"{device.type}:{index}"
    else:
        return device.type


def _get_rank() -> int:
    """Get rank, value is zero if not distributed"""
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0
