# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
import shutil
from functools import wraps

import psutil
import torch
from compressed_tensors.distributed import is_distributed, is_source_process
from compressed_tensors.offload.convert import from_accelerate
from compressed_tensors.offload.utils import as_single_threaded
from compressed_tensors.utils import patch_attr
from loguru import logger
from transformers import AutoModelForCausalLM, PreTrainedModel


__all__ = ["load_offloaded_model"]


@contextlib.contextmanager
def load_offloaded_model(
    model_class: type[PreTrainedModel] = AutoModelForCausalLM, extra_cpu_mem: int = 5e9
):
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

    :param model_class: model class to patch
    :param extra_cpu_mem: extra cpu memory to reserve for any operations not related to
        model loading (bytes). Defaults to 5Gb.
    """
    original_from_pretrained = model_class.from_pretrained
    patched_fn_called = False

    @classmethod
    @wraps(original_from_pretrained)
    def patched(cls, *args, **kwargs):
        nonlocal patched_fn_called
        patched_fn_called = True

        kwargs.setdefault("device_map", None)

        # Rank 0 does loading, other ranks init on meta device
        if not is_source_process():
            kwargs["device_map"] = "meta"
            # Workaround: transformers v5 tie_weights() calls torch.equal() on
            # meta tensors which is unsupported. Since rank 0 broadcasts the real
            # weights, we can safely skip tying on non-rank workers.
            kwargs.setdefault("tie_word_embeddings", False)

        # Intercept `auto_offload`: same as "auto", but only cpu/disk are visible
        elif kwargs["device_map"] == "auto_offload":
            kwargs["device_map"] = "auto"
            if "max_memory" not in kwargs:
                num_tensors, total_bytes = 0, 0
                if is_distributed():
                    num_tensors, total_bytes = _estimate_tensor_count(
                        original_from_pretrained, *args, **kwargs
                    )
                kwargs["max_memory"] = _get_cpu_memory(
                    extra_cpu_mem, num_tensors, total_bytes
                )

        # Unless the user specifies, use our memory estimates, which take into
        # account distributed setups and extra cpu reserved memory
        elif "max_memory" not in kwargs:
            num_tensors, total_bytes = 0, 0
            if is_distributed():
                num_tensors, total_bytes = _estimate_tensor_count(
                    original_from_pretrained, *args, **kwargs
                )
            kwargs["max_memory"] = _get_device_memory() | _get_cpu_memory(
                extra_cpu_mem, num_tensors, total_bytes
            )

        # Unless the user specifies, use `offload_buffers` to avoid accelerate weirdness
        if not kwargs.get("offload_buffers", True):
            logger.warning("Loading with `offload_buffers=False` is not supported")
        kwargs["offload_buffers"] = True

        model = original_from_pretrained(*args, **kwargs)
        from_accelerate(model)  # rank 0 shares weights with ranks via offload/broadcast

        return model

    with patch_attr(model_class, "from_pretrained", patched):
        try:
            yield
        finally:
            if not patched_fn_called:
                logger.warning(
                    f"`{model_class.__name__}.from_pretrained` was never called. If "
                    "you are loading with a model class other than "
                    f"{model_class.__name__}, please pass as argument to "
                    "`load_offloaded_model`"
                )


def _get_device_memory() -> dict[int, int]:
    if is_distributed():
        index = torch.accelerator.current_device_index()
        return {index: torch.accelerator.get_memory_info(index)[1]}
    else:
        return {
            index: torch.accelerator.get_memory_info(index)[1]
            for index in range(torch.accelerator.device_count())
        }


def _get_cpu_memory(
    extra_cpu_mem: int, num_tensors: int = 0, total_model_bytes: int = 0
) -> dict[str, int]:
    if is_distributed():
        return {
            "cpu": _get_shared_memory(num_tensors, total_model_bytes) - extra_cpu_mem
        }
    else:
        return {"cpu": psutil.virtual_memory().available - extra_cpu_mem}


def _get_shared_memory(num_tensors: int = 0, total_model_bytes: int = 0) -> int:
    linux_shm_path = "/dev/shm"
    if not os.path.exists(linux_shm_path):
        logger.warning(
            "Could not find shared memory at `/dev/shm`. Please add platform suppport"
        )
        return psutil.virtual_memory().available

    shm_bytes = shutil.disk_usage(linux_shm_path).total

    if num_tensors > 0:
        max_map_count = _get_max_map_count()
        if max_map_count is not None:
            curr_maps = _get_current_map_count()
            available_maps = max(0, max_map_count - curr_maps)
            avg_tensor_bytes = total_model_bytes // num_tensors
            mmap_byte_budget = available_maps * avg_tensor_bytes

            if mmap_byte_budget < shm_bytes:
                logger.info(
                    f"Capping shared memory from {shm_bytes} to {mmap_byte_budget} "
                    f"bytes based on mmap limit {max_map_count}."
                )
            shm_bytes = min(shm_bytes, mmap_byte_budget)

    return shm_bytes


def _get_max_map_count() -> int | None:
    try:
        with open("/proc/sys/vm/max_map_count") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _get_current_map_count() -> int:
    try:
        with open("/proc/self/maps") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _estimate_tensor_count(
    original_from_pretrained, *args, **kwargs
) -> tuple[int, int]:
    """helper to load model on meta device to count tensors + total bytes"""
    meta_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ("device_map", "max_memory", "offload_folder")
    }
    meta_kwargs.setdefault("tie_word_embeddings", False)
    try:
        with as_single_threaded():
            meta_model = original_from_pretrained(*args, device_map="meta", **meta_kwargs)
    except Exception:
        logger.warning("Meta device preload failed, skipping capacity estimate.")
        return 0, 0

    num_tensors = sum(1 for _ in meta_model.parameters()) + sum(
        1 for _ in meta_model.buffers()
    )
    total_bytes = sum(param.nbytes for param in meta_model.parameters()) + sum(
        buffer.nbytes for buffer in meta_model.buffers()
    )

    return num_tensors, total_bytes
