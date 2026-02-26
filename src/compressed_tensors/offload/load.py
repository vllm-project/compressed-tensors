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


@contextlib.contextmanager
def load_offloaded_model(extra_cpu_mem: int = 5e9):
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

    :param extra_cpu_mem: extra cpu memory to reserve for any operations not related to
        model loading (bytes). Defaults to 5Gb.
    """
    frame = _get_caller_frame()

    with contextlib.ExitStack() as stack:
        for obj in frame.f_globals.values():
            if isinstance(obj, type) and issubclass(obj, cls_to_patch):
                stack.enter_context(patch_from_pretrained(obj, extra_cpu_mem))

        yield


@contextlib.contextmanager
def patch_from_pretrained(obj: cls_to_patch, extra_cpu_mem: int):
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
                kwargs["max_memory"] = _get_cpu_memory(extra_cpu_mem)

        # Unless the user specifies, use our memory estimates, which take into
        # account distributed setups and extra cpu reserved memory
        elif "max_memory" not in kwargs:
            kwargs["max_memory"] = _get_device_memory() | _get_cpu_memory(extra_cpu_mem)


        if not is_rank0():
            # intercept model stub
            model_stub = args[0]

            # download files into tmp dir
            tmp_dir = "tmpdir"
            os.makedirs(tmp_dir, exist_ok=True)
            snapshot_download(
                repo_id=model_stub, local_dir=tmp_dir, ignore_patterns=[
                    "*.bin",
                    "*.safetensors",
                    "*.pth",
                    SAFE_WEIGHTS_INDEX_NAME,
                    WEIGHTS_INDEX_NAME,
                    "*.msgpack",
                    "*.pt",
                ]
            )

            # make an empty weights file to avoid errors
            weights_file_path = os.path.join(tmp_dir, "model.safetensors")
            save_file({}, weights_file_path, metadata={"format": "pt"})
            list(args)[0] = tmp_dir

        model = original_func(cls, *args, **kwargs)
        #from_accelerate(model)  # rank 0 shares weights with ranks via offload/broadcast
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


def _get_cpu_memory(extra_cpu_mem: int) -> dict[str, int]:
    if is_distributed():
        return {"cpu": _get_shared_memory() - extra_cpu_mem}
    else:
        return {"cpu": psutil.virtual_memory().available - extra_cpu_mem}


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



import contextlib
import logging
import os
import tempfile
from functools import wraps
from typing import Type

import torch
from compressed_tensors.offload import dispatch_model
from compressed_tensors.utils import deprecated, patch_attr
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import TORCH_INIT_FUNCTIONS
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME

__all__ = [
    "skip_weights_download",
    "patch_transformers_logger_level",
    "get_main_device",
    "dispatch_for_generation",
]


@contextlib.contextmanager
def skip_weights_download(model_class: Type[PreTrainedModel] = AutoModelForCausalLM):
    """
    Context manager under which models are initialized without having to download
    the model weight files. This differs from `init_empty_weights` in that weights are
    allocated on to assigned devices with random values, as opposed to being on the meta
    device

    :param model_class: class to patch, defaults to `AutoModelForCausalLM`
    """
    original_fn = model_class.from_pretrained
    weights_files = [
        "*.bin",
        "*.safetensors",
        "*.pth",
        SAFE_WEIGHTS_INDEX_NAME,
        WEIGHTS_INDEX_NAME,
        "*.msgpack",
        "*.pt",
    ]

    @classmethod
    def patched(cls, *args, **kwargs):
        nonlocal tmp_dir

        # intercept model stub
        model_stub = args[0] if args else kwargs.pop("pretrained_model_name_or_path")

        # download files into tmp dir
        os.makedirs(tmp_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_stub, local_dir=tmp_dir, ignore_patterns=weights_files
        )

        # make an empty weights file to avoid errors
        weights_file_path = os.path.join(tmp_dir, "model.safetensors")
        save_file({}, weights_file_path, metadata={"format": "pt"})

        # load from tmp dir
        model = original_fn(tmp_dir, **kwargs)

        # replace model_path
        model.name_or_path = model_stub
        model.config._name_or_path = model_stub

        return model

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch_attr(model_class, "from_pretrained", patched),
        skip_weights_initialize(),
        patch_transformers_logger_level(),
    ):
        yield


@contextlib.contextmanager
def skip_weights_initialize(use_zeros: bool = False):
    """
    Very similar to `transformers.model_utils.no_init_weights`, except that torch.Tensor
    initialization functions are also patched to account for tensors which are
    initialized not on the meta device
    """

    def skip(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if use_zeros:
            return tensor.fill_(0)
        return tensor

    with contextlib.ExitStack() as stack:
        for name in TORCH_INIT_FUNCTIONS.keys():
            stack.enter_context(patch_attr(torch.nn.init, name, skip))
            stack.enter_context(patch_attr(torch.Tensor, name, skip))
        yield


@contextlib.contextmanager
def patch_transformers_logger_level(level: int = logging.ERROR):
    """
    Context under which the transformers logger's level is modified

    This can be used with `skip_weights_download` to squelch warnings related to
    missing parameters in the checkpoint

    :param level: new logging level for transformers logger. Logs whose level is below
        this level will not be logged
    """
    transformers_logger = logging.getLogger("transformers.modeling_utils")
    restore_log_level = transformers_logger.getEffectiveLevel()

    transformers_logger.setLevel(level=level)
    yield
    transformers_logger.setLevel(level=restore_log_level)


def get_main_device() -> torch.device:
    rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device(f"xpu:{rank}")
    else:
        logger.warning("CUDA/XPU is not available! Compressing model on CPU instead")
        return torch.device("cpu")


@deprecated("compressed_tensors.offload::dispatch_model")
@wraps(dispatch_model)
def dispatch_for_generation(*args, **kwargs) -> PreTrainedModel:
    """
    Dispatch a model autoregressive generation. This means that modules are dispatched
    evenly across avaiable devices and kept onloaded if possible.

    :param model: model to dispatch
    :param hint_batch_size: reserve memory for batch size of inputs
    :param hint_batch_seq_len: reserve memory for sequence of length of inputs
    :param hint_model_dtype: reserve memory for model's dtype.
        Will be inferred from model if none is provided
    :param hint_extra_memory: extra memory reserved for model serving
    :param no_split_modules: names of module classes which should not be split
        across multiple devices
    :return: dispatched model
    """
    return dispatch_model(*args, **kwargs)
