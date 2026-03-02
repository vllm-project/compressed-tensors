# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import operator
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import compressed_tensors
import torch
import torch.distributed as dist
import transformers
from compressed_tensors.base import (
    COMPRESSION_VERSION_NAME,
    QUANTIZATION_CONFIG_NAME,
    QUANTIZATION_METHOD_NAME,
    SPARSITY_CONFIG_NAME,
    TRANSFORM_CONFIG_NAME,
)
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.config import CompressionFormat, SparsityCompressionConfig
from compressed_tensors.config.format import (
    infer_and_set_per_module_quantization_format,
)
from compressed_tensors.linear.compressed_linear import CompressedLinear
from compressed_tensors.offload import update_offload_parameter
from compressed_tensors.quantization import (
    DEFAULT_QUANTIZATION_METHOD,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
    load_pretrained_quantization_parameters,
)
from compressed_tensors.transform import TransformConfig
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    get_safetensors_folder,
    has_offloaded_params,
    merge_names,
    patch_attr,
)
from compressed_tensors.utils.helpers import (
    fix_fsdp_module_name,
    is_compressed_tensors_config,
)
from compressed_tensors.utils.match import match_named_modules
from loguru import logger
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm
from transformers import AutoConfig
from transformers.file_utils import CONFIG_NAME


__all__ = ["ModelCompressor", "map_module_to_scheme"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # dummy type if not available from transformers
    CompressedTensorsConfig = TypeVar("CompressedTensorsConfig")


class ModelCompressor:
    """
    Handles compression and decompression of a model with a sparsity config and/or
    quantization config.

    Compression LifeCycle
        - compressor = ModelCompressor.from_pretrained_model(model)
        - compressed_state_dict = compressor.compress(model, state_dict)
            - compressor.quantization_compressor.compress(model, state_dict)
            - compressor.sparsity_compressor.compress(model, state_dict)
        - model.save_pretrained(output_dir, state_dict=compressed_state_dict)
        - compressor.update_config(output_dir)

    Decompression LifeCycle
        - compressor = ModelCompressor.from_pretrained(comp_model_path)
        - model = AutoModel.from_pretrained(comp_model_path)
        - compressor.decompress(comp_model_path, model)
            - compressor.sparsity_compressor.decompress(comp_model_path, model)
            - compressor.quantization_compressor.decompress(comp_model_path, model)

    :param sparsity_config: config specifying sparsity compression parameters
    :param quantization_config: config specifying quantization compression parameters
    """

    sparsity_config: SparsityCompressionConfig | None = None
    quantization_config: QuantizationConfig | None = None
    transform_config: TransformConfig | None = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "ModelCompressor | None":
        """
        Given a path to a model config, extract the sparsity and/or quantization
        configs and load a ModelCompressor

        :param pretrained_model_name_or_path: path to model config on disk or HF hub
        :return: compressor for the configs, or None if model is not compressed
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        compression_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)
        return cls.from_compression_config(compression_config)

    @classmethod
    def from_compression_config(
        cls,
        compression_config: "dict[str, Any] | CompressedTensorsConfig",
    ):
        """
        :param compression_config:
            A compression or quantization config

            The type is one of the following:
            1. A Dict found under either "quantization_config" or "compression_config"
                keys in the config.json
            2. A CompressedTensorsConfig found under key "quantization_config" in HF
                model config
        :return: compressor for the configs, or None if model is not compressed
        """
        if compression_config is None:
            return None

        sparsity_config = cls.parse_sparsity_config(compression_config)
        quantization_config = cls.parse_quantization_config(compression_config)
        # TODO: transform config is not support by CompressedTensorsConfig yet

        if sparsity_config is None and quantization_config is None:
            return None

        if sparsity_config is not None:
            format = sparsity_config.get("format")
            sparsity_config = SparsityCompressionConfig.load_from_registry(
                format, **sparsity_config
            )
        if quantization_config is not None:
            quantization_config = QuantizationConfig.model_validate(quantization_config)

        return cls(
            sparsity_config=sparsity_config, quantization_config=quantization_config
        )

    @classmethod
    def from_pretrained_model(
        cls,
        model: Module,
        sparsity_config_or_format: SparsityCompressionConfig | str | None = None,
        quantization_format: str | None = None,
        sparsity_config: SparsityCompressionConfig | str | None = None,
    ) -> "ModelCompressor | None":
        """
        Given a pytorch model and optional sparsity and/or quantization configs,
        load the appropriate compressors

        :param model: pytorch model to target for compression
        :param sparsity_config: a filled in sparsity config or string corresponding
            to a sparsity format
        :param quantization_format: string corresponding to a quantization
            format that should be applied to the entire model
        :return: compressor for the configs, or None if model is not compressed
        """
        if sparsity_config:
            logger.warning(
                "sparsity_config is deprecated, use sparsity_config_or_format"
            )
            sparsity_config_or_format = sparsity_config

        if sparsity_config_or_format and isinstance(
            sparsity_config_or_format, str
        ):  # we passed in a sparsity format
            sparsity_config = SparsityCompressionConfig.load_from_registry(
                sparsity_config_or_format
            )
        else:
            # otherwise, config or None
            sparsity_config = sparsity_config_or_format

        quantization_format = infer_and_set_per_module_quantization_format(
            model=model,
            sparsity_structure=(
                sparsity_config.sparsity_structure
                if sparsity_config is not None
                else None
            ),
            quantization_format=quantization_format,
        )

        quantization_config = QuantizationConfig.from_pretrained(
            model, format=quantization_format
        )

        # use config attached to model
        transform_config = getattr(model, TRANSFORM_CONFIG_NAME, None)

        if not any((quantization_config, sparsity_config, transform_config)):
            return None

        return cls(
            sparsity_config=sparsity_config,
            quantization_config=quantization_config,
            transform_config=transform_config,
            compression_formats=quantization_format,
        )

    @staticmethod
    def parse_sparsity_config(
        compression_config: "dict[str, Any] | CompressedTensorsConfig",
    ) -> dict[str, Any] | None:
        """
        Parse sparsity config from quantization/compression config. Sparsity
        config is nested inside q/c config

        :param compression_config: quantization/compression config
        :return: sparsity config
        """
        if compression_config is None:
            return None

        if is_compressed_tensors_config(compression_config):
            s_config = compression_config.sparsity_config
            return s_config.model_dump() if s_config is not None else None

        # explicitly return None if {} in config
        return compression_config.get(SPARSITY_CONFIG_NAME, None) or None

    @staticmethod
    def parse_quantization_config(
        compression_config: "dict[str, Any] | CompressedTensorsConfig",
    ) -> dict[str, Any] | None:
        """
        Parse quantization config from quantization/compression config. The
        quantization are all the fields that are not the sparsity config or
        metadata fields

        :param compression_config: quantization/compression config
        :return: quantization config without sparsity config or metadata fields
        """
        if compression_config is None:
            return None

        if is_compressed_tensors_config(compression_config):
            q_config = compression_config.quantization_config
            return q_config.model_dump() if q_config is not None else None

        quantization_config = deepcopy(compression_config)
        quantization_config.pop(SPARSITY_CONFIG_NAME, None)
        quantization_config.pop(TRANSFORM_CONFIG_NAME, None)

        # some fields are required, even if a qconfig is not present
        # pop them off and if nothing remains, then there is no qconfig
        quant_method = quantization_config.pop(QUANTIZATION_METHOD_NAME, None)
        _ = quantization_config.pop(COMPRESSION_VERSION_NAME, None)

        if len(quantization_config) == 0:
            return None

        # replace popped off values
        # note that version is discarded for now
        if quant_method is not None:
            quantization_config[QUANTIZATION_METHOD_NAME] = quant_method

        return quantization_config

    def _fetch_unique_quantization_formats(self) -> list[str]:
        """
        Get all unique compression formats present in a model.
        :return: list of quantization formats
        """
        quantization_formats = []
        for _, scheme in self.quantization_config.config_groups.items():
            if scheme.format is not None and scheme.format not in quantization_formats:
                quantization_formats.append(scheme.format)

        if (
            len(quantization_formats) == 0
            and self.quantization_config.format
            != CompressionFormat.mixed_precision.value
        ):
            quantization_formats.append(self.quantization_config.format)
        return quantization_formats

    def __init__(
        self,
        sparsity_config: SparsityCompressionConfig | None = None,
        quantization_config: QuantizationConfig | None = None,
        transform_config: TransformConfig | None = None,
        compression_formats: list[str] | None = None,
    ):
        self.sparsity_config = sparsity_config
        self.quantization_config = quantization_config
        self.transform_config = transform_config
        self.compression_formats = compression_formats
        self.compressors: dict[str, BaseCompressor] = {}

        if sparsity_config is not None:
            self.compressors[sparsity_config.format] = BaseCompressor.load_from_registry(
                sparsity_config.format,
                config=sparsity_config,
            )

        if quantization_config is not None:
            # If a list of compression_format is not provided, we resolve the
            # relevant quantization formats using the config groups from the config
            # and if those are not defined, we fall-back to the global quantization fmt
            if not self.compression_formats:
                self.compression_formats = self._fetch_unique_quantization_formats()

            for format in self.compression_formats:
                self.compressors[format] = BaseCompressor.load_from_registry(
                    format, config=quantization_config
                )

    @property
    def sparsity_compressor(self) -> BaseCompressor | None:
        if self.sparsity_config is None:
            return None
        return self.compressors.get(self.sparsity_config.format)

    @property
    def quantization_compressor(self) -> dict[str, BaseCompressor] | None:
        if self.quantization_config is None:
            return None
        formats = self.compression_formats or []
        return {format: self.compressors[format] for format in formats}

    @staticmethod
    def _get_dist_context() -> tuple[bool, int, int]:
        if not dist.is_available() or not dist.is_initialized():
            return False, 0, 1
        return True, dist.get_rank(), dist.get_world_size()

    @staticmethod
    def _is_local_index(index: int, rank: int, world_size: int) -> bool:
        return (index % world_size) == rank

    @staticmethod
    def _merge_dicts_across_ranks(
        local_value: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, dict[str, torch.Tensor]]:
        enabled, _, world_size = ModelCompressor._get_dist_context()
        if not enabled or world_size == 1:
            return local_value

        gathered: list[dict[str, dict[str, torch.Tensor]]] = [None] * world_size
        dist.all_gather_object(gathered, local_value)
        merged: dict[str, dict[str, torch.Tensor]] = {}
        for item in gathered:
            merged.update(item)
        return merged

    @staticmethod
    def _resolve_module_format(
        module: Module, compression_formats: list[str]
    ) -> str | None:
        if not hasattr(module, "quantization_scheme"):
            return None
        if (
            not hasattr(module.quantization_scheme, "format")
            or module.quantization_scheme.format is None
        ):
            if len(compression_formats) > 1:
                raise ValueError(
                    "Applying multiple compressors without defining per module "
                    "formats is not supported"
                )
            return compression_formats[0]
        return module.quantization_scheme.format

    @staticmethod
    def _module_parameter_state(prefix: str, module: Module) -> dict[str, Tensor]:
        return {
            f"{prefix}.{name}": param
            for name, param in module.named_parameters(recurse=False)
        }

    def _sparse_compression_targets(self, model: Module) -> set[str]:
        if self.sparsity_config is None:
            return set()
        return {
            module_name
            for module_name, _module in match_named_modules(
                model=model,
                targets=self.sparsity_config.targets,
                ignore=self.sparsity_config.ignore,
            )
        }

    def _candidate_modules(
        self,
        model: Module,
        module_to_scheme: dict[str, QuantizationScheme],
        sparse_targets: set[str],
    ) -> list[tuple[str, Module]]:
        return list(
            match_named_modules(
                model,
                [*sparse_targets, *module_to_scheme.keys()],
                warn_on_fail=True,
            )
        )

    def _module_quantization_compressor(self, module: Module) -> BaseCompressor | None:
        if self.quantization_config is None:
            return None
        format = self._resolve_module_format(module, self.compression_formats)
        return self.compressors.get(format)

    def _primary_quantization_compressor(self) -> BaseCompressor | None:
        if self.quantization_config is None:
            return None
        if not self.compression_formats:
            return None
        return self.compressors.get(self.compression_formats[0])

    def _compress_module_state(
        self,
        prefix: str,
        module: Module,
        module_state: dict[str, Tensor],
        module_to_scheme: dict[str, QuantizationScheme],
        sparse_targets: set[str],
        compression_device: str = "cpu",
    ) -> dict[str, Tensor]:
        if prefix in module_to_scheme:
            quant_compressor = self._module_quantization_compressor(module)
            module_state = quant_compressor.compress(
                module_state,
                names_to_scheme=module_to_scheme,
                show_progress=False,
                compression_device=compression_device,
            )

        if prefix in sparse_targets and self.sparsity_compressor is not None:
            module_state = self.sparsity_compressor.compress(
                module_state,
                compression_targets=sparse_targets,
                show_progress=False,
            )
        return module_state

    def _decompress_module_state(
        self,
        prefix: str,
        module: Module,
        module_state: dict[str, Tensor],
        module_to_scheme: dict[str, QuantizationScheme],
        sparse_targets: set[str],
    ) -> dict[str, Tensor]:
        if prefix in sparse_targets and self.sparsity_compressor is not None:
            generator = self.sparsity_compressor.decompress_from_state_dict(module_state)
            module_state = {key: value for key, value in generator}

        if prefix in module_to_scheme:
            quant_compressor = self._module_quantization_compressor(module)
            module_state = quant_compressor.decompress_module_from_state_dict(
                prefix,
                module_state,
                scheme=module_to_scheme[prefix],
            )
        return module_state

    def get_missing_module_keys(self, model: Module) -> list[str]:
        """
        Identifies the expected missing weight keys in the compressed state_dict.

        When a model undergoes sparsity or quantization compression, certain
        weight tensors may be absent from the checkpoint by virtue of compression.
        This function determines which weight keys are missing based on the
        applied compression techniques.

        :param model: The PyTorch model to check for missing keys.
        :return: A list of missing keys expected in the compressed state_dict.
        """
        missing_keys = set()

        # Determine missing keys due to sparsity compression
        if (
            self.sparsity_config is not None
            and self.sparsity_config.format != CompressionFormat.dense.value
        ):
            sparse_targets = match_named_modules(
                model=model,
                targets=self.sparsity_config.targets,
                ignore=self.sparsity_config.ignore,
            )

            missing_keys.update(
                merge_names(target_name, "weight")
                for target_name, _module in sparse_targets
            )

        # Determine missing keys due to pack quantization
        if (
            self.quantization_config is not None
            and self.quantization_config.format == CompressionFormat.pack_quantized.value
        ):
            for scheme in self.quantization_config.config_groups.values():
                quant_targets = match_named_modules(
                    model=model,
                    targets=scheme.targets,
                    ignore=self.quantization_config.ignore,
                )
                missing_keys.update(
                    merge_names(target_name, "weight")
                    for target_name, _module in quant_targets
                )

        return list(missing_keys)

    def get_unexpected_file_keys(self, model: Module) -> list[str]:
        """
        Identifies extra keys introduced by the compression process in the
        compressed state_dict that are not expected by the model graph.

        During sparsity or quantization compression, additional metadata or
        auxiliary parameters may be stored in the checkpoint, which do not
        correspond to any parameter in the original model. These keys are
        typically introduced to support the reconstruction of compressed weights.

        For example, Sparse24Bitmask compression may introduce keys such as
        'compressed', 'bitmask', and 'shape' in the checkpoint, which are
        not part of the original model parameters.

        :param model: The PyTorch model to check for unexpected keys.
        :return: A list of extra keys introduced by the compression process
                that are not expected by the model.
        """

        unexpected_keys = set()

        # Identify unexpected keys from sparsity compression
        if (
            self.sparsity_config is not None
            and self.sparsity_config.format != CompressionFormat.dense.value
        ):
            sparse_targets = match_named_modules(
                model=model,
                targets=self.sparsity_config.targets,
                ignore=self.sparsity_config.ignore,
            )
            sparse_compressor = self.compressors[self.sparsity_config.format]
            unexpected_keys.update(
                merge_names(target_name, param)
                for target_name, _module in sparse_targets
                for param in sparse_compressor.compression_param_names
            )

        # Identify unexpected keys from quantization compression
        if self.quantization_config is not None:
            for scheme in self.quantization_config.config_groups.values():
                quant_targets = match_named_modules(
                    model=model,
                    targets=scheme.targets,
                    ignore=self.quantization_config.ignore,
                )
                for format in self.compression_formats:
                    quant_compressor = self.compressors[format]
                    unexpected_keys.update(
                        merge_names(target_name, param)
                        for target_name, _module in quant_targets
                        for param in quant_compressor.compression_param_names
                        if param != "weight"
                    )

        return list(unexpected_keys)

    # ----- model memory compression/decompression pathways ----- #

    def compress_model(self, model: Module):
        """
        Compress a model in memory. Because the model structure is modified in place,
        this method is more memory-efficient than `self.compress`

        :param model: model containing parameters to compress
        """
        module_to_scheme = map_module_to_scheme(model)
        sparse_compression_targets = self._sparse_compression_targets(model)
        candidate_modules = self._candidate_modules(
            model, module_to_scheme, sparse_compression_targets
        )
        enabled, rank, world_size = self._get_dist_context()
        local_compressed: dict[str, dict[str, Tensor]] = {}

        for module_index, (prefix, module) in enumerate(
            tqdm(candidate_modules, desc="Compressing model")
        ):
            if not self._is_local_index(module_index, rank, world_size):
                continue

            if isinstance(module, CompressedLinear):
                continue

            module_device = get_execution_device(module)
            is_meta = module_device.type == "meta"

            exec_device = "meta" if is_meta else "cpu"

            with align_module_device(module, execution_device=exec_device):
                state_dict = self._module_parameter_state(prefix, module)

            state_dict = self._compress_module_state(
                prefix=prefix,
                module=module,
                module_state=state_dict,
                module_to_scheme=module_to_scheme,
                sparse_targets=sparse_compression_targets,
                compression_device=exec_device,
            )

            local_compressed[prefix] = state_dict

        compressed_by_module = (
            self._merge_dicts_across_ranks(local_compressed) if enabled else local_compressed
        )

        for prefix, module in candidate_modules:
            state_dict = compressed_by_module.get(prefix)
            if state_dict is None:
                continue

            module_device = get_execution_device(module)
            onloading_device = "meta" if module_device.type == "meta" else module_device

            for name, _ in list(module.named_parameters(recurse=False)):
                delattr(module, name)

            for name, value in state_dict.items():
                name = name.removeprefix(f"{prefix}.")
                value = value.to(onloading_device)
                param = torch.nn.Parameter(value, requires_grad=False)
                module.register_parameter(name, param)

            module.quantization_status = QuantizationStatus.COMPRESSED
        # TODO: consider sparse compression to also be compression
        if (
            self.quantization_config is not None
            and self.quantization_config.format != CompressionFormat.dense.value
        ):
            self.quantization_config.quantization_status = QuantizationStatus.COMPRESSED

    def decompress_model(self, model: Module):
        """
        Decompress a model in memory. Because the model structure is modified in place,
        this method does not require loading some compression parameters from disk

        :param model: model containing parameters to compress
        """
        module_to_scheme = map_module_to_scheme(model)
        sparse_compression_targets = self._sparse_compression_targets(model)
        candidate_modules = self._candidate_modules(
            model, module_to_scheme, sparse_compression_targets
        )
        enabled, rank, world_size = self._get_dist_context()
        local_decompressed: dict[str, dict[str, Tensor]] = {}

        for module_index, (prefix, module) in enumerate(
            tqdm(candidate_modules, desc="Decompressing model")
        ):
            if not self._is_local_index(module_index, rank, world_size):
                continue

            with align_module_device(module, execution_device="cpu"):
                state_dict = self._module_parameter_state(prefix, module)

            state_dict = self._decompress_module_state(
                prefix=prefix,
                module=module,
                module_state=state_dict,
                module_to_scheme=module_to_scheme,
                sparse_targets=sparse_compression_targets,
            )

            local_decompressed[prefix] = state_dict

        decompressed_by_module = (
            self._merge_dicts_across_ranks(local_decompressed)
            if enabled
            else local_decompressed
        )

        for prefix, module in candidate_modules:
            state_dict = decompressed_by_module.get(prefix)
            if state_dict is None:
                continue

            exec_device = get_execution_device(module)
            for name, _ in list(module.named_parameters(recurse=False)):
                delattr(module, name)

            for name, value in state_dict.items():
                name = name.removeprefix(f"{prefix}.")
                value = value.to(exec_device)
                param = torch.nn.Parameter(value, requires_grad=False)
                module.register_parameter(name, param)

            module.quantization_status = QuantizationStatus.FROZEN

    # ----- state dict compression pathways ----- #

    def compress(
        self,
        model: Module,
        state_dict: dict[str, Tensor] | None = None,
        show_progress: bool = False,
    ) -> dict[str, Tensor]:
        """
        Compresses a dense state dict or model with sparsity and/or quantization

        :param model: uncompressed model to compress
        :param state_dict: optional uncompressed state_dict to insert into model
        :return: compressed state dict
        """

        if state_dict is None:
            state_dict = model.state_dict()

        enabled, rank, world_size = self._get_dist_context()
        module_to_scheme = map_module_to_scheme(model)
        sparse_compression_targets = self._sparse_compression_targets(model)
        candidate_modules = self._candidate_modules(
            model, module_to_scheme, sparse_compression_targets
        )

        if not enabled or world_size == 1:
            for prefix, module in candidate_modules:
                module_state = {
                    key: value
                    for key, value in state_dict.items()
                    if key == prefix or key.startswith(f"{prefix}.")
                }
                if len(module_state) == 0:
                    continue
                module_state = self._compress_module_state(
                    prefix=prefix,
                    module=module,
                    module_state=module_state,
                    module_to_scheme=module_to_scheme,
                    sparse_targets=sparse_compression_targets,
                )
                state_dict = {
                    key: value
                    for key, value in state_dict.items()
                    if not (key == prefix or key.startswith(f"{prefix}."))
                }
                state_dict.update(module_state)
        else:
            local_compressed: dict[str, dict[str, Tensor]] = {}
            for module_index, (prefix, module) in enumerate(candidate_modules):
                if not self._is_local_index(module_index, rank, world_size):
                    continue

                module_state = {
                    key: value
                    for key, value in state_dict.items()
                    if key == prefix or key.startswith(f"{prefix}.")
                }
                if len(module_state) == 0:
                    continue
                local_compressed[prefix] = self._compress_module_state(
                    prefix=prefix,
                    module=module,
                    module_state=module_state,
                    module_to_scheme=module_to_scheme,
                    sparse_targets=sparse_compression_targets,
                )

            compressed_by_module = self._merge_dicts_across_ranks(local_compressed)
            compressed_prefixes = set(compressed_by_module.keys())
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if all(
                    not (key == prefix or key.startswith(f"{prefix}."))
                    for prefix in compressed_prefixes
                )
            }
            for module_state in compressed_by_module.values():
                state_dict.update(module_state)

        if (
            self.quantization_config is not None
            and self.quantization_config.format != CompressionFormat.dense.value
        ):
            self.quantization_config.quantization_status = (
                QuantizationStatus.COMPRESSED
            )

        # HACK: Override the dtype_byte_size function in transformers to
        # support float8 types. Fix is posted upstream
        # https://github.com/huggingface/transformers/pull/30488
        transformers.modeling_utils.dtype_byte_size = new_dtype_byte_size

        return state_dict

    # ----- disk decompression pathways ----- #

    def decompress(
        self, model_path: str | Path | dict[str, Tensor], model: Module
    ):
        """
        Overwrites the weights in model with weights decompressed from model_path

        :param model_path: path to compressed weights
        :param model: pytorch model to load decompressed weights into

        Note: decompress makes use of both _replace_sparsity_weights and
        _replace_weights. The variations in these methods are a result of the subtle
        variations between the sparsity and quantization compressors. Specifically,
        quantization compressors return not just the decompressed weight, but the
        quantization parameters (e.g scales, zero_point) whereas sparsity compressors
        only return the decompressed weight.

        """
        if isinstance(model_path, (str, Path)):
            model_path = get_safetensors_folder(str(model_path))
        sparse_decompressed = False
        quant_compressor = self._primary_quantization_compressor()

        if (
            self.sparsity_compressor is not None
            and self.sparsity_config.format != CompressionFormat.dense.value
        ):
            # note - decompress only supports one compressor atm
            params_to_ignore = None
            if quant_compressor is not None:
                params_to_ignore = quant_compressor.compression_param_names
            # Sparse decompression is applied on the model_path
            # The compressor will try and load any quantization parameters as well
            # params_to_skip_load will skip over quantization params from being loaded
            if isinstance(model_path, dict):
                dense_gen = self.sparsity_compressor.decompress_from_state_dict(model_path)
            else:
                dense_gen = self.sparsity_compressor.decompress(
                    model_path, params_to_skip_load=params_to_ignore
                )
            self._replace_sparsity_weights(dense_gen, model)
            setattr(model, SPARSITY_CONFIG_NAME, self.sparsity_compressor.config)
            sparse_decompressed = True

        if quant_compressor is not None:
            # Temporarily set quantization status to FROZEN to prevent
            # quantization during apply_quantization_config. This ensures
            # that the dtypes of the weights are not unintentionally updated.
            # The status is restored after quantization params are loaded.

            with patch_attr(
                self.quantization_config,
                "quantization_status",
                QuantizationStatus.FROZEN,
            ):
                apply_quantization_config(model, self.quantization_config)
                names_to_scheme: dict[str, QuantizationScheme] = {
                    name: getattr(module, "quantization_scheme")
                    for name, module in model.named_modules()
                    if getattr(module, "quantization_scheme", None) is not None
                }
                # Load activation scales/zp or any other quantization parameters
                # Conditionally load the weight quantization parameters if we have a
                # dense compressor or if a sparsity compressor has already been applied
                load_weight_qparams = (
                    sparse_decompressed
                    or quant_compressor.config.format == CompressionFormat.dense.value
                )
                if isinstance(model_path, (str, Path)):
                    load_pretrained_quantization_parameters(
                        model,
                        model_path,
                        # TODO: all weight quantization params will be moved to the
                        # compressor in a follow-up including initialization
                        load_weight_qparams=load_weight_qparams,
                    )

            model_path_or_state_dict = (
                model.state_dict() if sparse_decompressed else model_path
            )

            dense_gen = quant_compressor.decompress(
                model_path_or_state_dict, names_to_scheme=names_to_scheme
            )
            # TODO: all weight quantization params will be moved to the compressor
            # to prevent duplicate parameter updates in update_offload_parameter
            self._replace_weights(
                dense_gen, model, load_weight_qparams=not load_weight_qparams
            )

            def freeze_quantization_status(module):
                module.quantization_status = QuantizationStatus.FROZEN

            model.apply(freeze_quantization_status)
            setattr(model, QUANTIZATION_CONFIG_NAME, self.quantization_config)

    def update_config(self, save_directory: str):
        """
        Update the model config located at save_directory with compression configs
        for sparsity and/or quantization

        :param save_directory: path to a folder containing a HF model config
        """
        # this check is also done in `from_pretrained_model`,
        # but not in `from_pretrained`` or `from_compression_config``
        if not any(
            (self.quantization_config, self.sparsity_config, self.transform_config)
        ):
            return

        # write to config.json file, regardless of whether it exists already
        # overwrite previous config and version if already existing
        config_file_path = os.path.join(save_directory, CONFIG_NAME)
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as file:
                config_data = json.load(file)
        else:
            config_data = {}

        # serialize configs into json
        qconfig_data = (
            self.quantization_config.model_dump(exclude=["quant_method"])
            if self.quantization_config is not None
            else {}
        )
        sconfig_data = (
            self.sparsity_config.model_dump()
            if self.sparsity_config is not None
            else {}
        )
        tconfig_data = (
            self.transform_config.model_dump()
            if self.transform_config is not None
            else {}
        )

        # construct compression (quantization) config
        config_data[QUANTIZATION_CONFIG_NAME] = {
            COMPRESSION_VERSION_NAME: compressed_tensors.__version__,
            QUANTIZATION_METHOD_NAME: DEFAULT_QUANTIZATION_METHOD,
            SPARSITY_CONFIG_NAME: sconfig_data,
            TRANSFORM_CONFIG_NAME: tconfig_data,
            **qconfig_data,
        }

        # write results to config.json file
        with open(config_file_path, "w") as config_file:
            json.dump(config_data, config_file, indent=2, sort_keys=True)

    def _replace_sparsity_weights(self, dense_weight_generator, model: Module):
        """
        Replace the weights of the model with the
        provided dense weights.

        This method iterates over the dense_weight_generator and
        updates the corresponding weights in the model. If a parameter
        name does not exist in the model, it will be skipped.

        :param dense_weight_generator (generator): A generator that yields
            tuples of (name, data), where 'name' is the parameter name and
            'data' is the updated param data
        :param model: The model whose weights are to be updated.
        """
        for name, data in tqdm(dense_weight_generator, desc="Decompressing model"):
            split_name = name.split(".")
            prefix, param_name = ".".join(split_name[:-1]), split_name[-1]
            module = operator.attrgetter(prefix)(model)

            params_device = next(module.parameters()).device
            device = "cpu" if has_offloaded_params(module) else params_device
            delattr(module, param_name)
            requires_grad = data.dtype in (torch.float16, torch.float32, torch.bfloat16)
            param = torch.nn.Parameter(data.to(device), requires_grad=requires_grad)
            module.register_parameter(param_name, param)

    def _replace_weights(
        self, dense_weight_generator, model: Module, load_weight_qparams: bool = True
    ):
        """
        Replace the weights of the model with the
        provided dense weights.

        This method iterates over the dense_weight_generator and
        updates the corresponding weights in the model. If a parameter
        name does not exist in the model, it will be skipped.

        :param dense_weight_generator (generator): A generator that yields
            tuples of (name, data), where 'name' is the parameter name and
            'data' is the updated param data
        :param model: The model whose weights are to be updated.
        """

        for mod_path, data in tqdm(dense_weight_generator, desc="Decompressing model"):
            module = operator.attrgetter(mod_path)(model)

            params_device = next(module.parameters()).device
            device = "cpu" if has_offloaded_params(module) else params_device

            for param_name, param_data in data.items():
                if hasattr(module, param_name):
                    # If compressed, will have an incorrect dtype for transformers >4.49
                    # TODO: we can also just skip initialization of scales/zp if in
                    # decompression in init to be consistent with loading which happens
                    # later as well however, update_data does a good shape check -
                    # should be moved to the compressor

                    if param_name == "weight":
                        delattr(module, param_name)
                        requires_grad = param_data.dtype in (
                            torch.float16,
                            torch.float32,
                            torch.bfloat16,
                        )
                        param = torch.nn.Parameter(
                            param_data.to(device), requires_grad=requires_grad
                        )
                        module.register_parameter(param_name, param)
                    elif load_weight_qparams:
                        # Should already be registered to the correct device for
                        # for scales/zero-points
                        update_offload_parameter(module, param_name, param_data)


def map_module_to_scheme(model: Module) -> dict[str, QuantizationScheme]:
    """
    Returns a dictionary which maps quantized module names to their quantization
    schemes. Only includes modules with weight quantization
    """
    return {
        fix_fsdp_module_name(name): module.quantization_scheme
        for name, module in model.named_modules()
        if (
            hasattr(module, "quantization_scheme")
            and module.quantization_scheme.weights is not None
        )
    }


# HACK: Override the dtype_byte_size function in transformers to support float8 types
# Fix is posted upstream https://github.com/huggingface/transformers/pull/30488
def new_dtype_byte_size(dtype):
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)_?", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8
