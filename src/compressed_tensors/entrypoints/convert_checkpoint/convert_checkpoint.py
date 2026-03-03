# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tqdm
from compressed_tensors.convert import (
    Converter,
    convert_file,
    get_checkpoint_files,
    is_weights_file,
    update_config,
    update_safetensors_index,
    validate_file,
)
from loguru import logger


__all__ = ["convert_checkpoint"]


def convert_checkpoint(
    model_stub: str | os.PathLike,
    save_directory: str | os.PathLike,
    converter: Converter,
    max_workers: int = 1,
):
    """
    Convert a model checkpoint to compressed-tensors format without loading it up
    in memory, instead operating directly on the model safetensors files. This
    entrypoint operates on a model stub or folder containing weights saved in
    safetensors files, and updates the corresponding quantization_config field in
    the config.json. All additional files will be copied to new checkpoint.

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
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying {file_path} {save_path}")
            shutil.copyfile(resolved_path, save_path)

    with ThreadPoolExecutor(max_workers) as executor:
        # 1. validate quantizable tensors fail fast before long-running quantization
        futures = [executor.submit(*job) for job in validate_jobs]
        for future in tqdm.tqdm(
            as_completed(futures), total=len(futures), desc="Validating"
        ):
            future.result()

        # 2-5. quantize and compress weights
        total_size = 0
        weight_map = dict()
        futures = [executor.submit(*job) for job in convert_jobs]
        for future in tqdm.tqdm(
            as_completed(futures), total=len(futures), desc="Converting"
        ):
            _total_size, _weight_map = future.result()
            total_size += _total_size
            weight_map.update(_weight_map)

    # 5. update config and safetensors index
    update_config(save_directory, converter)
    update_safetensors_index(save_directory, total_size, weight_map)
