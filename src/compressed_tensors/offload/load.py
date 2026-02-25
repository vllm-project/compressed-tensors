# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import inspect
import os
import shutil
from functools import wraps
from types import FrameType

import psutil
import torch
import torch.distributed as dist
from compressed_tensors.offload.convert import from_accelerate
from compressed_tensors.offload.dist_utils import is_distributed, is_rank0
from loguru import logger
from transformers import PreTrainedModel
from transformers.models.auto.modeling_auto import _BaseAutoModelClass


__all__ = ["load_offloaded_model"]


cls_to_patch = _BaseAutoModelClass | PreTrainedModel
CPU_RESERVE_MEMORY = 5e9


@contextlib.contextmanager
def load_offloaded_model():
    """
    Context manager used to load a transformers model with offloading implemented by
    compressed-tensors.

    The model is first loaded with accelerate's offloading, then convereted into
    offloading implemented by compressed-tensors. If a distributed environment has been
    initialized, then rank 0 loads the weights while other ranks load on the meta
    device, then the offload is shared across ranks during conversion.

    In addition to the standard `device_map` options, this context also supports
    `device_map="auto_offload"`, which means that the model will load as many parameters
    can fit onto the cpu, and any extra parameters will be loaded on disk.
    """
    frame = _get_caller_frame()

    with contextlib.ExitStack() as stack:
        for obj in frame.f_globals.values():
            if isinstance(obj, type) and issubclass(obj, cls_to_patch):
                stack.enter_context(patch_from_pretrained(obj))

        yield


@contextlib.contextmanager
def patch_from_pretrained(obj: cls_to_patch):
    original_func = obj.from_pretrained.__func__

    @wraps(original_func)
    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("device_map", None)

        # Rank 0 does loading, other ranks init on meta device
        if not is_rank0():
            kwargs["device_map"] = "meta"

        # Intercept `auto_offload`: same as "auto", but only cpu/disk are visible
        elif kwargs["device_map"] == "auto_offload":
            kwargs["device_map"] = "auto"
            if "max_memory" not in kwargs:
                kwargs["max_memory"] = _get_cpu_memory()

        # Unless the user specifies, use our memory estimates, which take into
        # account distributed setups and extra cpu reserved memory
        elif "max_memory" not in kwargs:
            kwargs["max_memory"] = _get_device_memory() | _get_cpu_memory()

        model = original_func(cls, *args, **kwargs)
        from_accelerate(model)  # rank 0 shares weights with ranks via offload/broadcast
        return model

    try:
        obj.from_pretrained = from_pretrained.__get__(obj)
        yield
    finally:
        obj.from_pretrained = original_func.__get__(obj)


def _get_device_memory() -> dict[int, int]:
    # TODO: extend to xpu, ect.
    if is_distributed():
        index = dist.get_rank()
        return {index: torch.cuda.get_device_properties(index).total_memory}
    else:
        return {
            index: torch.cuda.get_device_properties(index).total_memory
            for index in range(torch.cuda.device_count())
        }


def _get_cpu_memory() -> dict[str, int]:
    if is_distributed():
        return {"cpu": _get_shared_memory() - CPU_RESERVE_MEMORY}
    else:
        return {"cpu": psutil.virtual_memory().available - CPU_RESERVE_MEMORY}


def _get_shared_memory() -> int:
    linux_shm_path = "/dev/shm"
    if os.path.exists(linux_shm_path):
        total, _used, _free = shutil.disk_usage(linux_shm_path)
        return total

    else:
        logger.warning(
            "Could not find shared memory at `/dev/shm`. Please add platform suppport"
        )
        return psutil.virtual_memory().available


def _get_caller_frame() -> FrameType:
    frame = inspect.currentframe()
    frame = frame.f_back.f_back  # skip this function's caller's frame
    while frame is not None and "contextlib" in frame.f_code.co_filename:
        frame = frame.f_back  # skip contextlib frames

    if frame is None:
        raise RuntimeError("Could not find caller frame")

    return frame
