# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import re
import struct
from collections import defaultdict
from collections.abc import Iterable

import torch
from huggingface_hub import list_repo_files
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.file_utils import CONFIG_NAME
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file


__all__ = [
    "get_safetensors_folder",
    "get_safetensors_header",
    "match_param_name",
    "get_weight_mappings",
    "get_nested_weight_mappings",
    "get_quantization_parameter_to_path_mapping",
    "get_file_map",
    "InverseWeightMap",
    "load_tensors_from_inverse_weight_map",
    "is_quantization_param",
    "find_config_path",
    "find_safetensors_index_path",
    "find_safetensors_index_file",
    "get_weight_map",
    "update_safetensors_index",
    "is_weights_file",
    "get_checkpoint_files",
]

WeightMappingType = dict[str, str]
NestedWeightMappingType = dict[str, WeightMappingType]


def is_weights_file(file_name: str) -> bool:
    """
    Check whether a filename corresponds to a model weights file based on its
    extension.

    :param file_name: filename to check
    :return: True if the file is a recognized weights format, else False
    """
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
    """
    Given a local path or HuggingFace model stub, return a mapping from each
    file's relative path to its resolved local path. Local directories are
    walked recursively; HuggingFace stubs are resolved via the Hub API.

    :param model_stub: local path to a model directory or HuggingFace model stub
    :return: dict mapping relative file path to resolved local file path
    """
    # In the future, this function can accept and pass download kwargs to cached_file

    if os.path.exists(model_stub):
        file_paths = _walk_directory_files(
            model_stub, ignore=[".cache", ".gitattributes"]
        )
    else:
        file_paths = list_repo_files(model_stub)

    return {file_path: cached_file(model_stub, file_path) for file_path in file_paths}


def _walk_directory_files(root_dir: str, ignore: Iterable[str]) -> list[str]:
    """
    Return all file paths relative to root_dir, optionally skipping entries
    whose relative path starts with `ignore`.

    :param root_dir: root directory to walk
    :param ignore: optional path prefix to exclude from results
    :return: list of relative file paths
    """

    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            rel_path = os.path.relpath(os.path.join(dirpath, filename), root_dir)
            if not any([rel_path.startswith(i) for i in ignore]):
                all_files.append(rel_path)
    return all_files


def find_safetensors_index_path(save_directory: str | os.PathLike) -> str | None:
    """
    Search save_directory for a safetensors weight index file.

    :param save_directory: directory to search
    :return: absolute path to the index file, or None if not found
    """
    for file_name in os.listdir(save_directory):
        if file_name.endswith("safetensors.index.json"):
            return os.path.join(save_directory, file_name)
    return None


def find_config_path(save_directory: str | os.PathLike) -> str | None:
    """
    Search save_directory for a model config file (config.json or params.json).

    :param save_directory: directory to search
    :return: absolute path to the config file, or None if not found
    """
    for file_name in os.listdir(save_directory):
        if file_name in (CONFIG_NAME, "params.json"):
            return os.path.join(save_directory, file_name)
    return None


def find_safetensors_index_file(model_files: dict[str, str]) -> str | None:
    """
    Find safetensors index file from full list of model_files

    :param model_files: mapping of file relative path to absolute path, usually the
        result of `get_checkpoint_files`
    :return: absolute path to the safetensors index file, or None if not found
    """
    for file_path, resolved_path in model_files.items():
        if file_path.endswith(SAFE_WEIGHTS_INDEX_NAME):
            return resolved_path

    return None


def get_weight_map(model_files: dict[str, str]) -> dict[str, str]:
    """
    Get weight map from full list of model_files.
    If safetensors index.json file is found, weight_map can be pulled from there.
    Otherwise, it is created from the single safetensors weights file.

    :returns: weight map of form {weight name -> safetensor file name}
    """
    index_file = find_safetensors_index_file(model_files)
    if index_file is not None:
        with open(index_file, "r") as f:
            return json.load(f)["weight_map"]

    # if no index_file, use model.saftensors instead.
    if SAFE_WEIGHTS_NAME not in model_files:
        raise ValueError(
            f"File {SAFE_WEIGHTS_NAME} expected but not found in {model_files.keys()}"
        )

    # create from model.safetensors
    with safe_open(model_files[SAFE_WEIGHTS_NAME], "r") as file:
        return {tensor: SAFE_WEIGHTS_NAME for tensor in file.keys()}


def update_safetensors_index(
    save_directory: str | os.PathLike,
    total_size: int,
    weight_map: dict[str, str],
):
    """
    Write (or overwrite) the safetensors weight index file in save_directory.
    If an existing index file is found it will be replaced in-place; otherwise
    the standard model.safetensors.index.json filename is used.

    :param save_directory: directory containing the checkpoint
    :param total_size: total byte size of all shards, stored in index metadata
    :param weight_map: mapping from tensor name to shard filename
    """
    file_path = find_safetensors_index_path(save_directory)
    if file_path is None:
        file_path = os.path.join(save_directory, SAFE_WEIGHTS_INDEX_NAME)

    with open(file_path, "w") as file:
        json.dump(
            {
                "metadata": {
                    "total_size": total_size,
                },
                "weight_map": weight_map,
            },
            file,
            indent=2,
            sort_keys=True,
        )


def get_safetensors_folder(
    pretrained_model_name_or_path: str, cache_dir: str | None = None
) -> str:
    """
    Given a Hugging Face stub or a local path, return the folder containing the
    safetensors weight files

    :param pretrained_model_name_or_path: local path to model or HF stub
    :param cache_dir: optional cache dir to search through, if none is specified the
    model will be searched for in the default TRANSFORMERS_CACHE
    :return: local folder containing model data
    """
    if os.path.exists(pretrained_model_name_or_path):
        # argument is a path to a local folder
        return os.path.abspath(pretrained_model_name_or_path)

    safetensors_path = cached_file(
        pretrained_model_name_or_path,
        SAFE_WEIGHTS_NAME,
        cache_dir=cache_dir,
        _raise_exceptions_for_missing_entries=False,
    )
    index_path = cached_file(
        pretrained_model_name_or_path,
        SAFE_WEIGHTS_INDEX_NAME,
        cache_dir=cache_dir,
        _raise_exceptions_for_missing_entries=False,
    )
    if safetensors_path is not None:
        # found a single cached safetensors file
        return os.path.split(safetensors_path)[0]
    if index_path is not None:
        # found a cached safetensors weight index file
        return os.path.split(index_path)[0]

    # model weights could not be found locally or cached from HF Hub
    raise ValueError(
        "Could not locate safetensors weight or index file from "
        f"{pretrained_model_name_or_path}."
    )


def get_safetensors_header(safetensors_path: str) -> dict[str, str]:
    """
    Extracts the metadata from a safetensors file as JSON

    :param safetensors_path: path to a safetensors file
    :return: dictionary of metadata extracted from the safetensors file
    """
    with open(safetensors_path, "rb") as f:
        length_of_header = struct.unpack("<Q", f.read(8))[0]
        header_data = f.read(length_of_header)
        header = json.loads(header_data)

    return header


def match_param_name(full_name: str, param_name: str) -> str | None:
    """
    Helper function extracting the uncompressed parameterized layer name from a
    compressed name. Assumes the compressed name was merged using merge_names.

    :param full_name: full name of parameter in compressed model
    :param param_name: compression paramater name
    :return: uncompressed name of the uncompressed parameterized layer
    """
    pattern = r"^(.*)\." + param_name + r"$"
    regex = re.findall(pattern, full_name)
    if len(regex) == 0:
        return None
    return regex[0]


def get_weight_mappings(path_to_model_or_tensors: str) -> dict[str, str]:
    """
    Takes a path to a state dict saved in safetensors format and returns a mapping
    from parameterized layer name to file location.

    {
        layer.weight.bitmask: file_location,
        layer.weight.row_offsets: file_location,
        layer.weight.shape: file_location,
        layer.weight.compressed: file_location
    }

    This generalizes to cases where the model is split into multiple safetensors files

    :param path_to_model_or_tensors: path to directory that contains
        safetensors (must contain either a single file or multiple files with an index),
        or a path to a single safetensors file
    :return: mapping of parameterized layer name to file location
    """

    if os.path.isfile(path_to_model_or_tensors):
        # we have a single safetensors file to read
        header = get_safetensors_header(path_to_model_or_tensors)
        for key in header.keys():
            header[key] = path_to_model_or_tensors
        header.pop("__metadata__", None)
    else:
        # we have a directory with multiple safetensors files
        safetensors_path = os.path.join(path_to_model_or_tensors, SAFE_WEIGHTS_NAME)
        index_path = os.path.join(path_to_model_or_tensors, SAFE_WEIGHTS_INDEX_NAME)
        if os.path.exists(safetensors_path):
            # we have a single safetensors file to read
            header = get_safetensors_header(safetensors_path)
            for key in header.keys():
                header[key] = SAFE_WEIGHTS_NAME
            header.pop("__metadata__", None)
        elif os.path.exists(index_path):
            # we have multiple safetensors file, read from index
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            header = index["weight_map"]
        else:
            raise ValueError(
                "Could not find a safetensors weight "
                f"or index file at {path_to_model_or_tensors}"
            )

        # convert weight locations to full paths
        for key, value in header.items():
            header[key] = os.path.join(path_to_model_or_tensors, value)

    return header


def get_nested_weight_mappings(
    model_path: str,
    params_to_nest: Iterable[str],
    return_unmatched_params: bool = False,
) -> NestedWeightMappingType | tuple[NestedWeightMappingType, WeightMappingType]:
    """
    Takes a path to a state dict saved in safetensors format and returns a nested
    mapping from uncompressed parameterized layer names to the file locations of
    each layer's compression parameters.

    Example of the nested mapping:
    layer: {
        bitmask: file_location,
        row_offsets: file_location,
        shape: file_location,
        compressed: file_location
    }

    If other parameters are found that do not match the nested parameters, they will
    be returned in a separate dictionary only if return_unmatched_params is True.
    This dictionary may be needed for cases where compressors are stacked (e.g.,
    quantization compression followed by sparse compression).

    Example of the unmatched params mapping:
    {
        layer.weight_scale: file_location,
        layer.input_scale: file_location
    }

    This generalizes to cases where the model is split into multiple safetensors
    files.

    :param model_path: Path to the safetensors state dict, must contain either a
        single safetensors file or multiple files with an index.
    :param params_to_nest: Iterable of parameter names to nest.
    :param return_unmatched_params: If True, return a second dictionary containing
        the remaining parameters that were not matched to the params_to_nest.
    :return:
        - If return_unmatched_params is False:
            NestedWeightMappingType: A nested mapping of parameterized layer names to
            file locations of each layer's compression parameters.
        - If return_unmatched_params is True:
            Tuple[NestedWeightMappingType, WeightMappingType]: A tuple containing:
                - NestedWeightMappingType: A nested mapping of parameterized layer
                names to file locations of each layer's compression parameters.
                - WeightMappingType: A mapping of the remaining parameter names to
                their file locations that were not matched to the params_to_nest.
    """
    weight_mappings = get_weight_mappings(model_path)
    nested_weight_mappings = {}
    unmatched_params = {}

    for key, file_location in weight_mappings.items():
        matched = False
        for param_name in params_to_nest:
            module_path = match_param_name(key, param_name)
            if module_path:
                if module_path not in nested_weight_mappings:
                    nested_weight_mappings[module_path] = {}
                nested_weight_mappings[module_path][param_name] = file_location
                matched = True
        if return_unmatched_params and not matched:
            unmatched_params[key] = file_location

    if return_unmatched_params:
        return nested_weight_mappings, unmatched_params
    return nested_weight_mappings


def get_quantization_parameter_to_path_mapping(model_path: str) -> dict[str, str]:
    """
    Given a model path, return a mapping between a parameter and its path
    on disk
    """
    weight_mappings = get_weight_mappings(model_path)
    mapping = {}
    for weight_name, safe_path in weight_mappings.items():
        if is_quantization_param(weight_name):
            mapping[weight_name] = safe_path
            continue
    return mapping


def get_file_map(weight_map: WeightMappingType) -> dict[str, list[str]]:
    """
    Given a safetensors index file's weight_map, which maps weight name to safetensors
    file name, return a mapping of safetensors file name to list of weight names

    :param weight_map: mapping of weight name to safetensors file name.
        result of `get_weight_mappings`
    :returns: file_map
    """

    file_map = defaultdict(list)
    for k, v in weight_map.items():
        file_map[v].append(k)

    return dict(file_map)


InverseWeightMap = dict[str, list[str] | None]
"""
Mapping of absolute path -> list of tensors. Used to pull tensors across different
safetensors files that must be loaded/processed together. Used in conjunction
with `load_tensors_from_inverse_weight_map`
"""


def load_tensors_from_inverse_weight_map(
    inverse_weight_map: InverseWeightMap,
    device: str | torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """
    Given an inverse_weight_map, which is a dictionary of file name to list of
    tensor names, load up all listed tensor names

    :param inverse_weight_map: mapping of resolved source file path ->
        list of tensor names to load from that file. Precomputed by
        build_inverse_weight_map() in the job-building phase.
        If list is empty, all tensors are pulled
        Example: {"/path/shard0.safetensors": ["q_proj.weight"],
                  "/path/shard1.safetensors": ["k_proj.weight", "v_proj.weight"]}
    :param device: tensors will be loaded onto this device. Defaults to CPU

    :returns: mapping of tensor name to actual tensor loaded from safetensors file
        Example: {"q_proj.weight": torch.Tensor(...), "k_proj.weight: torch.Tensor(...)}
    """
    tensors: dict[str, torch.Tensor] = {}
    for source_file, tensor_names in inverse_weight_map.items():
        with safe_open(source_file, framework="pt", device=str(device)) as f:
            keys = f.keys()
            # if tensor_names is empty, pull all tensors
            if tensor_names is None or len(tensor_names) == 0:
                tensor_names = keys
            for tensor_name in tensor_names:
                if tensor_name not in keys:
                    raise ValueError(
                        f"Expected to find tensor {tensor_name} in "
                        f"{source_file}, but tensor was not found."
                    )
                tensors[tensor_name] = f.get_tensor(tensor_name)
    return tensors


def is_quantization_param(name: str) -> bool:
    """
    Checks is a parameter name is associated with a quantization parameter

    :param name: parameter name to check
    :return: True if parameter name is a quantization parameter, else False
    """
    if name.endswith("_scale"):
        return True
    if name.endswith("zero_point"):
        return True
    if name.endswith("g_idx"):
        return True

    return False


def _fetch_and_save_prefix_tensors(
    source_model: str, prefix: str, dest_dir: str, shard_name: str
) -> dict:
    """
    Extracts all tensors whose keys start with `prefix` from `source_model`
    and saves them as a new shard in `dest_dir`. This is useful when saving
    MTP layers from the original checkpoint into the quantized checkpoint,
    since MTP layers are not included in the quantized model and thus must
    be copied over as-is.

    :param source_model: local path or HuggingFace stub of the source model
    :param prefix: tensor key prefix to filter on
    :param dest_dir: destination directory to write the shard into
    :param shard_name: filename for the new shard
    :return: dict mapping tensor key to tensor

    """
    source_dir = get_safetensors_folder(source_model)
    weight_mappings = get_weight_mappings(source_dir)

    tensors = {}
    for key, filepath in weight_mappings.items():
        if key.startswith(prefix):
            with safe_open(filepath, framework="pt", device="cpu") as f:
                tensors[key] = f.get_tensor(key)

    if len(tensors) <= 0:
        raise ValueError(f"No tensors with prefix '{prefix}' found in {source_model}")

    save_file(tensors, os.path.join(dest_dir, shard_name))
    return tensors
