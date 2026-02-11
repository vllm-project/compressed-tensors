# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import inspect
from functools import wraps
from types import FrameType

import psutil
import torch
from compressed_tensors.offload.convert import from_accelerate
from compressed_tensors.offload.dist_utils import is_rank0
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
    `device_map="disk"`, which means that the model will load as many parameters can
    fit onto the cpu, and any extra parameters will be loaded on disk.
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
        arguments = inspect.signature(original_func).bind(*args).arguments
        arguments.update(kwargs)

        if is_rank0():
            device_map = arguments.get("device_map", None)

            # intercept "disk"
            if device_map == "disk":
                arguments["device_map"] = "auto"
                if "max_memory" not in arguments:
                    arguments["max_memory"] = {"cpu": psutil.virtual_memory().available}

            model = original_func(cls, **arguments)

        else:
            arguments["device_map"] = "meta"
            model = original_func(cls, **arguments)

        from_accelerate(model)
        return model

    obj.from_pretrained = from_pretrained.__get__(obj)
    yield
    obj.from_pretrained = original_func.__get__(obj)


DeviceMap = dict[str, tuple[torch.device | None, torch.device | None]]


def _get_caller_frame() -> FrameType:
    frame = inspect.currentframe()
    frame = frame.f_back.f_back  # skip this function's caller's frame
    while frame is not None and "contextlib" in frame.f_code.co_filename:
        frame = frame.f_back  # skip contextlib frames

    if frame is None:
        raise RuntimeError("Could not find caller frame")

    return frame
