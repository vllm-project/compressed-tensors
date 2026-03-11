# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from compressed_tensors.entrypoints.convert import Converter
from safetensors.torch import load_file, save_file


__all__ = [
    "validate_file",
    "convert_file",
]


def validate_file(
    file_path: str | os.PathLike,
    converter: Converter,
):
    """
    Validate that each quantizable tensor in a safetensors file can be quantized.

    :param file_path: safetensors file to validate
    :param converter: converter we wish to apply to the checkpoint,
        e.g. conversion of some layers from some format to compressed-tensors
    """
    tensors = load_file(file_path)

    converter.validate(tensors)


def convert_file(
    file_path: str | os.PathLike,
    save_path: str | os.PathLike,
    converter: Converter,
) -> tuple[int, dict[str, str]]:
    """
    Convert tensors in a given safetensors file

    :param file_path: safetensors file to process
    :param save_path: save path of file with quantized weights
    :param converter: converter we wish to apply to the checkpoint,
        e.g. conversion of some layers from some format to compressed-tensors
    """
    tensors = load_file(file_path)

    converter.process(tensors)

    save_file(tensors, save_path)
    total_size = sum(tensor.nbytes for tensor in tensors.values())
    weight_map = {key: os.path.basename(save_path) for key in tensors.keys()}
    return total_size, weight_map
