# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from huggingface_hub import list_repo_files
from transformers.file_utils import CONFIG_NAME
from transformers.utils.hub import cached_file


__all__ = [
    "get_checkpoint_files",
    "is_weights_file",
    "find_config_path",
    "find_safetensors_index_path",
]


def is_weights_file(file_name: str) -> bool:
    return any(
        file_name.endswith(suffix)
        for suffix in [
            ".bin",
            ".safetensors",
            ".pth",
            ".msgpack",
            ".pt",
        ]
    )


def get_checkpoint_files(model_stub: str | os.PathLike) -> dict[str, str]:
    # In the future, this function can accept and pass download kwargs to cached_file

    if os.path.exists(model_stub):
        file_paths = _walk_file_paths(model_stub, ignore=[".cache", ".gitattributes"])
    else:
        file_paths = list_repo_files(model_stub)

    return {file_path: cached_file(model_stub, file_path) for file_path in file_paths}


def _walk_file_paths(root_dir: str, ignore: list[str] | str) -> list[str]:
    """
    Return all file paths relative to the root directory
    """

    if isinstance(ignore, str):
        ignore = [ignore]

    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            rel_path = os.path.relpath(os.path.join(dirpath, filename), root_dir)
            if not any([rel_path.startswith(i) for i in ignore]):
                all_files.append(rel_path)
    return all_files


def find_safetensors_index_path(save_directory: str | os.PathLike) -> str | None:
    for file_name in os.listdir(save_directory):
        if file_name.endswith("safetensors.index.json"):
            return os.path.join(save_directory, file_name)

    return None


def find_config_path(save_directory: str | os.PathLike) -> str | None:
    for file_name in os.listdir(save_directory):
        if file_name in (CONFIG_NAME, "params.json"):
            return os.path.join(save_directory, file_name)

    return None
