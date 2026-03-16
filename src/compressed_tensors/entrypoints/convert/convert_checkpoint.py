# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tqdm
from compressed_tensors.entrypoints.convert.convert_file import (
    convert_file,
    validate_file,
)
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.entrypoints.convert.file_utils import (
    get_checkpoint_files,
    is_weights_file,
)
from compressed_tensors.entrypoints.convert.save_utils import (
    update_config,
    update_safetensors_index,
)
from loguru import logger


__all__ = ["convert_checkpoint", "exec_jobs"]


def convert_checkpoint(
    model_stub: str | os.PathLike,
    save_directory: str | os.PathLike,
    converter: Converter,
    max_workers: int = 1,
):
    """
    Convert a model checkpoint to either:
    - its equivalent quantized format in compressed-tensors
    - the unquantized format
    without loading it up in memory, instead operating directly on the model
    safetensors files. This entrypoint operates on a model stub or folder containing
    weights saved in safetensors files, and updates the corresponding
    quantization_config field in the config.json. All additional files will be
    copied to new checkpoint.

    :param model_stub: huggingface model hub or path to local weights files
    :param save_directory: new checkpoint will be saved in this directory.
    :param max_workers: number of worker threads to process files with
    :param device: gpu device to accelerate quantization with
    :param converters: converter we wish to apply to the checkpoint,
        e.g. conversion of some layers from some format to compressed-tensors
    """
    # validate arguments
    model_files = get_checkpoint_files(model_stub)

    # 0. collect safetensors files, copy files
    validate_jobs = []
    convert_jobs = []
    for file_path, resolved_path in model_files.items():
        save_path = Path(save_directory) / file_path

        if file_path.endswith("safetensors"):
            validate_jobs.append((validate_file, resolved_path, converter))
            convert_jobs.append((convert_file, resolved_path, save_path, converter))

        else:
            if is_weights_file(file_path):
                logger.warning(f"Skip processing for weights file {file_path}")
            if str(resolved_path) != str(save_path):
                save_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Copying {file_path} {save_path}")
                shutil.copyfile(resolved_path, save_path)

    # 1. validate quantizable tensors fail fast before long-running quantization
    exec_jobs(validate_jobs, max_workers, desc="Validating")

    # 2-5. quantize and compress weights
    total_size = 0
    weight_map = dict()
    convert_results = exec_jobs(convert_jobs, max_workers, desc="Converting")
    for _total_size, _weight_map in convert_results:
        total_size += _total_size
        weight_map.update(_weight_map)

    # 5. update config and safetensors index
    update_config(save_directory, converter)
    update_safetensors_index(save_directory, total_size, weight_map)


def exec_jobs(
    jobs: list[tuple[Callable, ...]], max_workers: int = 1, desc: str = "Executing Jobs"
) -> list:
    """
    Execute jobs in parallel, using ThreadPoolExecutor

    :param jobs: list of tuples, the first entry of which is the callable,
        and the remaining elements are the inputs args to the callable
    :param max_workers: number of workers to use
    :param desc: tqdm description
    """
    results = []
    if max_workers == 1:
        for job in tqdm.tqdm(jobs, desc=desc):
            results.append([job[0](*job[1:])])
        return results

    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(*job) for job in jobs]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc=desc):
            results.append(future.result())

    return results
