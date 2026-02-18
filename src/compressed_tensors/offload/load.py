# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import inspect
from functools import wraps
from types import FrameType

import psutil
import torch
import torch.distributed as dist
from compressed_tensors.offload.convert import from_accelerate
from compressed_tensors.offload.dist_utils import is_distributed, is_rank0
from transformers import PreTrainedModel
from transformers.models.auto.modeling_auto import _BaseAutoModelClass


__all__ = ["load_offloaded_model"]


cls_to_patch = _BaseAutoModelClass | PreTrainedModel


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

        # Intercept auto device map options
        match (kwargs["device_map"], is_distributed()):
            case "auto", True:
                if "max_memory" not in kwargs:
                    # only sees local device memory
                    kwargs["max_memory"] = _get_device_memory() | _get_cpu_memory()

            case "auto_offload", _:
                kwargs["device_map"] = "auto"
                if "max_memory" not in kwargs:
                    kwargs["max_memory"] = _get_cpu_memory()

        # Rank 0 does loading, other ranks init on meta device
        if not is_rank0():
            kwargs["device_map"] = "meta"
        model = original_func(cls, *args, **kwargs)

        # During conversion, rank 0 shares weights with ranks via offload/broadcast
        from_accelerate(model)
        return model

    obj.from_pretrained = from_pretrained.__get__(obj)
    yield
    obj.from_pretrained = original_func.__get__(obj)


def _get_device_memory() -> dict[int, int]:
    assert is_distributed()
    device_memory = torch.cuda.get_device_properties(dist.get_rank()).total_memory
    return {dist.get_rank(): device_memory}


def _get_cpu_memory() -> dict[str, int]:
    return {"cpu": psutil.virtual_memory().available}


def _get_caller_frame() -> FrameType:
    frame = inspect.currentframe()
    frame = frame.f_back.f_back  # skip this function's caller's frame
    while frame is not None and "contextlib" in frame.f_code.co_filename:
        frame = frame.f_back  # skip contextlib frames

    if frame is None:
        raise RuntimeError("Could not find caller frame")

    return frame
