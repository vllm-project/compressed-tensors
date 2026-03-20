# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import torch
import tqdm
from compressed_tensors.entrypoints.convert import (
    find_safetensors_index_file,
    get_checkpoint_files,
    invert_mapping,
    is_weights_file,
    update_safetensors_index,
)
from loguru import logger
from safetensors.torch import load_file, save_file


def reindex_checkpoint(
    model_stub: str,
    save_directory: str,
    get_unmatched_names: Callable[[Iterable[str]], list[str]],
    num_workers: int = 5,
):
    """
    Reindex the safetensors files of a model according to user-provided function,
    `get_unmatched_names`, such that weights exist in the same safetensors file.
    This is necessary for:

    - microscale schemes, where fused weights must exist in the same file, for example:
        1) gate_proj and up_proj -> fused gate_up_proj
        2) q_proj, k_proj and v_proj -> fused qkv_proj
    - previously compressed checkpoints, where weights and qparams must exist in the
        same safetensors file, for example:
        1) weight and weight_scale_inv must exist in the same safetensors file when
        expanding from FP8 quant method's FP8_BLOCK to bf16

    This script assumes weight locality; if a set of fused weights are not in a file,
    1. the incomplete set is the last set of weights (sorted alphabetically)
    2. the remainder of the incomplete set is the next file (sorted alphabetically)

    This assumption holds true for most model checkpoints, even in the common case where
    weights are sorted alphabetically and not numerically.

    :param model_stub: huggingface model hub or path to local weights files
    :param save_directory: output directory for reindexed weights files
    :param get_unmatched_names: function that takes a list of tensor names in current
        safetensors file, and returns the list of unmatched names, i.e. a list of names
        of the tensors that are missing one or more accompanying tensors.
    :param num_workers: number of worker threads to save files with
    """

    # read files
    model_files = get_checkpoint_files(model_stub)
    index_file = find_safetensors_index_file(model_files)
    if index_file is None:
        raise ValueError(
            "This script is used to modify safetensor file shards, but was "
            "unable to find safetensors index file. No reindexing is required."
        )

    # copy non-weight files
    for file_path, resolved_path in model_files.items():
        save_path = Path(save_directory) / file_path

        if file_path.endswith("safetensors"):
            continue
        else:
            if is_weights_file(file_path):
                logger.warning(f"Skip processing for weights file {file_path}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if str(save_path) != str(resolved_path):
                logger.debug(f"Copying {file_path} {save_path}")
                shutil.copyfile(resolved_path, save_path)

    # read index file
    with open(index_file, "r") as file:
        index_file_data = json.load(file)

    weight_map: dict[str, str] = index_file_data["weight_map"]
    final_weight_map: dict[str, str] = {}

    # set up copy executor and carry over
    writers = ThreadPoolExecutor(max_workers=num_workers)
    carry_over_tensors: dict[str, torch.Tensor] = {}

    # iterate in alphabetical order on assumption of weight-file locality
    file_map = invert_mapping(weight_map)
    file_map = sorted(file_map)
    progress = tqdm.tqdm(total=len(file_map))
    for file_name in file_map:
        file_path = model_files[file_name]
        save_path = os.path.join(save_directory, file_name)
        tensors = load_file(file_path)

        if len(carry_over_tensors) > 0:
            # add carryover
            tensors.update(carry_over_tensors)
            logger.info(f"Moved {list(carry_over_tensors.keys())} into {file_name}")
            carry_over_tensors = {}

        tensor_names = sorted(list(tensors.keys()))
        unmatched_names = get_unmatched_names(tensor_names)
        for unmatched_name in unmatched_names:
            # move to carry over
            carry_over_tensors.update({unmatched_name: tensors[unmatched_name]})

            # delete from current file
            tensor_names.remove(unmatched_name)
            del tensors[unmatched_name]

        # save tensors after modification
        writers.submit(_with_progress, save_file, tensors, save_path, progress=progress)
        final_weight_map.update({name: file_name for name in tensor_names})

    total_size = index_file_data["metadata"]["total_size"]
    update_safetensors_index(save_directory, total_size, final_weight_map)
    writers.shutdown(wait=True)


def _with_progress(fn: callable, *args, progress: tqdm.tqdm):
    ret = fn(*args)
    progress.update(1)
    return ret
