# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import operator
import os
import re
from copy import deepcopy
from importlib import import_module
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
from compressed_tensors.config import SparsityCompressionConfig
from compressed_tensors.config.format import infer_and_set_per_module_quantization_format
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
    patch_attr,
)
from compressed_tensors.utils.helpers import fix_fsdp_module_name, is_compressed_tensors_config
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


_FORMAT_TO_IMPL_MODULE: dict[str, str] = {
    "dense": "compressed_tensors.compressors.dense.impl",
    "naive-quantized": "compressed_tensors.compressors.naive_quantized.impl",
    "int-quantized": "compressed_tensors.compressors.naive_quantized.impl",
    "float-quantized": "compressed_tensors.compressors.naive_quantized.impl",
    "pack-quantized": "compressed_tensors.compressors.pack_quantized.impl",
    "nvfp4-pack-quantized": "compressed_tensors.compressors.fp4_quantized.impl",
    "mxfp4-pack-quantized": "compressed_tensors.compressors.fp4_quantized.impl",
    "marlin-24": "compressed_tensors.compressors.marlin_24.impl",
    "sparse-bitmask": "compressed_tensors.compressors.sparse_bitmask.impl",
    "sparse-24-bitmask": "compressed_tensors.compressors.sparse_24_bitmask.impl",
}


def _ensure_compressor_registered(compression_format: str):
    module_name = _FORMAT_TO_IMPL_MODULE.get(compression_format)
    if module_name is not None:
        import_module(module_name)


class ModelCompressor:
    """
    Lightweight orchestrator around format compressors.

    Compression and decompression are driven by per-module quantization scheme formats.
    Sparse compression execution paths are intentionally removed; sparse config fields are
    only retained as compatibility metadata for downstream integrations.
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
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        compression_config = getattr(config, QUANTIZATION_CONFIG_NAME, None)
        return cls.from_compression_config(compression_config)

    @classmethod
    def from_compression_config(
        cls,
        compression_config: "dict[str, Any] | CompressedTensorsConfig",
    ):
        if compression_config is None:
            return None

        sparsity_config = cls.parse_sparsity_config(compression_config)
        quantization_config = cls.parse_quantization_config(compression_config)

        if sparsity_config is None and quantization_config is None:
            return None

        if sparsity_config is not None:
            fmt = sparsity_config.get("format")
            sparsity_config = SparsityCompressionConfig.load_from_registry(
                fmt, **sparsity_config
            )
        if quantization_config is not None:
            quantization_config = QuantizationConfig.model_validate(quantization_config)

        return cls(
            sparsity_config=sparsity_config,
            quantization_config=quantization_config,
        )

    @classmethod
    def from_pretrained_model(
        cls,
        model: Module,
        sparsity_config_or_format: SparsityCompressionConfig | str | None = None,
        quantization_format: str | None = None,
        sparsity_config: SparsityCompressionConfig | str | None = None,
    ) -> "ModelCompressor | None":
        if sparsity_config:
            logger.warning("sparsity_config is deprecated, use sparsity_config_or_format")
            sparsity_config_or_format = sparsity_config

        resolved_sparsity_config = None
        if isinstance(sparsity_config_or_format, str):
            resolved_sparsity_config = SparsityCompressionConfig.load_from_registry(
                sparsity_config_or_format
            )
        elif sparsity_config_or_format is not None:
            resolved_sparsity_config = sparsity_config_or_format

        # Sparsity structure is intentionally not part of format inference anymore.
        quantization_format = infer_and_set_per_module_quantization_format(
            model=model,
            sparsity_structure=None,
            quantization_format=quantization_format,
        )

        quantization_config = QuantizationConfig.from_pretrained(
            model,
            format=quantization_format,
        )

        transform_config = getattr(model, TRANSFORM_CONFIG_NAME, None)
        if not any((quantization_config, resolved_sparsity_config, transform_config)):
            return None

        return cls(
            sparsity_config=resolved_sparsity_config,
            quantization_config=quantization_config,
            transform_config=transform_config,
            compression_formats=quantization_format,
        )

    @staticmethod
    def parse_sparsity_config(
        compression_config: "dict[str, Any] | CompressedTensorsConfig",
    ) -> dict[str, Any] | None:
        if compression_config is None:
            return None

        if is_compressed_tensors_config(compression_config):
            s_config = compression_config.sparsity_config
            return s_config.model_dump() if s_config is not None else None

        return compression_config.get(SPARSITY_CONFIG_NAME, None) or None

    @staticmethod
    def parse_quantization_config(
        compression_config: "dict[str, Any] | CompressedTensorsConfig",
    ) -> dict[str, Any] | None:
        if compression_config is None:
            return None

        if is_compressed_tensors_config(compression_config):
            q_config = compression_config.quantization_config
            return q_config.model_dump() if q_config is not None else None

        quantization_config = deepcopy(compression_config)
        quantization_config.pop(SPARSITY_CONFIG_NAME, None)
        quantization_config.pop(TRANSFORM_CONFIG_NAME, None)

        quant_method = quantization_config.pop(QUANTIZATION_METHOD_NAME, None)
        _ = quantization_config.pop(COMPRESSION_VERSION_NAME, None)
        if len(quantization_config) == 0:
            return None

        if quant_method is not None:
            quantization_config[QUANTIZATION_METHOD_NAME] = quant_method

        return quantization_config

    def _fetch_unique_quantization_formats(self) -> list[str]:
        if self.quantization_config is None:
            return []

        quantization_formats: list[str] = []
        for _, scheme in self.quantization_config.config_groups.items():
            if scheme.format is not None and scheme.format not in quantization_formats:
                quantization_formats.append(scheme.format)

        if len(quantization_formats) == 0 and self.quantization_config.format is not None:
            quantization_formats.append(self.quantization_config.format)

        return quantization_formats

    def __init__(
        self,
        sparsity_config: SparsityCompressionConfig | None = None,
        quantization_config: QuantizationConfig | None = None,
        transform_config: TransformConfig | None = None,
        compression_formats: list[str] | str | None = None,
    ):
        self.sparsity_config = sparsity_config
        self.quantization_config = quantization_config
        self.transform_config = transform_config

        if compression_formats is None:
            self.compression_formats = self._fetch_unique_quantization_formats()
        elif isinstance(compression_formats, str):
            self.compression_formats = [compression_formats]
        else:
            self.compression_formats = compression_formats

        self.compressors: dict[str, BaseCompressor] = {}
        for fmt in self.compression_formats:
            _ensure_compressor_registered(fmt)
            self.compressors[fmt] = BaseCompressor.load_from_registry(
                fmt,
                config=self.quantization_config,
            )

        # Compatibility only for integrations that read
        # `ModelCompressor.sparsity_compressor.decompress_weight`.
        self._sparsity_compressor = None
        if sparsity_config is not None:
            try:
                _ensure_compressor_registered(sparsity_config.format)
                self._sparsity_compressor = BaseCompressor.load_from_registry(
                    sparsity_config.format,
                    config=sparsity_config,
                )
            except Exception:
                self._sparsity_compressor = None

    @property
    def sparsity_compressor(self):
        return self._sparsity_compressor

    @property
    def quantization_compressor(self) -> dict[str, BaseCompressor] | None:
        if self.quantization_config is None:
            return None
        return {fmt: self.compressors[fmt] for fmt in self.compression_formats}

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
        module: Module,
        compression_formats: list[str],
    ) -> str | None:
        if not hasattr(module, "quantization_scheme"):
            return None

        scheme_format = getattr(module.quantization_scheme, "format", None)
        if scheme_format is not None:
            return scheme_format

        if len(compression_formats) == 1:
            return compression_formats[0]

        if len(compression_formats) > 1:
            raise ValueError(
                "Applying multiple compressors without defining per-module formats "
                "is not supported"
            )

        return None

    @staticmethod
    def _module_parameter_state(prefix: str, module: Module) -> dict[str, Tensor]:
        return {
            f"{prefix}.{name}": param
            for name, param in module.named_parameters(recurse=False)
        }

    @staticmethod
    def _state_dict_for_prefix(
        state_dict: dict[str, Tensor],
        prefix: str,
    ) -> dict[str, Tensor]:
        return {
            key: value
            for key, value in state_dict.items()
            if key == prefix or key.startswith(f"{prefix}.")
        }

    @staticmethod
    def _replace_prefix_entries(
        state_dict: dict[str, Tensor],
        prefix: str,
        module_state: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not (key == prefix or key.startswith(f"{prefix}."))
        }
        state_dict.update(module_state)
        return state_dict

    @staticmethod
    def _module_plan(model: Module) -> tuple[dict[str, QuantizationScheme], list[tuple[str, Module]]]:
        module_to_scheme = map_module_to_scheme(model)
        candidates = list(
            match_named_modules(
                model,
                list(module_to_scheme.keys()),
                warn_on_fail=True,
            )
        )
        return module_to_scheme, candidates

    def _module_quantization_compressor(self, module: Module) -> BaseCompressor | None:
        if self.quantization_config is None:
            return None
        fmt = self._resolve_module_format(module, self.compression_formats)
        if fmt is None:
            return None
        return self.compressors.get(fmt)

    def _primary_quantization_compressor(self) -> BaseCompressor | None:
        if self.quantization_config is None or not self.compression_formats:
            return None
        return self.compressors.get(self.compression_formats[0])

    def _compress_module_state(
        self,
        prefix: str,
        module: Module,
        module_state: dict[str, Tensor],
        module_to_scheme: dict[str, QuantizationScheme],
        compression_device: str = "cpu",
    ) -> dict[str, Tensor]:
        if prefix not in module_to_scheme:
            return module_state

        compressor = self._module_quantization_compressor(module)
        if compressor is None:
            return module_state

        return compressor.compress(
            module_state,
            names_to_scheme=module_to_scheme,
            show_progress=False,
            compression_device=compression_device,
        )

    def _decompress_module_state(
        self,
        prefix: str,
        module: Module,
        module_state: dict[str, Tensor],
        module_to_scheme: dict[str, QuantizationScheme],
    ) -> dict[str, Tensor]:
        if prefix not in module_to_scheme:
            return module_state

        compressor = self._module_quantization_compressor(module)
        if compressor is None:
            return module_state

        return compressor.decompress_module_from_state_dict(
            prefix,
            module_state,
            scheme=module_to_scheme[prefix],
        )

    def get_missing_module_keys(self, model: Module) -> list[str]:
        # Keys removed by compression are format-specific and no longer tracked here.
        return []

    def get_unexpected_file_keys(self, model: Module) -> list[str]:
        # Compression metadata keys are format-specific and no longer tracked here.
        return []

    def compress_model(self, model: Module):
        module_to_scheme, candidate_modules = self._module_plan(model)
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
            exec_device = "meta" if module_device.type == "meta" else "cpu"

            with align_module_device(module, execution_device=exec_device):
                state_dict = self._module_parameter_state(prefix, module)

            state_dict = self._compress_module_state(
                prefix=prefix,
                module=module,
                module_state=state_dict,
                module_to_scheme=module_to_scheme,
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

        if self.quantization_config is not None:
            self.quantization_config.quantization_status = QuantizationStatus.COMPRESSED

    def decompress_model(self, model: Module):
        module_to_scheme, candidate_modules = self._module_plan(model)
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

    def compress(
        self,
        model: Module,
        state_dict: dict[str, Tensor] | None = None,
        show_progress: bool = False,
    ) -> dict[str, Tensor]:
        if state_dict is None:
            state_dict = model.state_dict()

        enabled, rank, world_size = self._get_dist_context()
        module_to_scheme, candidate_modules = self._module_plan(model)

        if not enabled or world_size == 1:
            iterator = candidate_modules
            if show_progress:
                iterator = tqdm(candidate_modules, desc="Compressing state dict")
            for prefix, module in iterator:
                module_state = self._state_dict_for_prefix(state_dict, prefix)
                if len(module_state) == 0:
                    continue
                module_state = self._compress_module_state(
                    prefix=prefix,
                    module=module,
                    module_state=module_state,
                    module_to_scheme=module_to_scheme,
                )
                state_dict = self._replace_prefix_entries(state_dict, prefix, module_state)
        else:
            local_compressed: dict[str, dict[str, Tensor]] = {}
            for module_index, (prefix, module) in enumerate(candidate_modules):
                if not self._is_local_index(module_index, rank, world_size):
                    continue

                module_state = self._state_dict_for_prefix(state_dict, prefix)
                if len(module_state) == 0:
                    continue
                local_compressed[prefix] = self._compress_module_state(
                    prefix=prefix,
                    module=module,
                    module_state=module_state,
                    module_to_scheme=module_to_scheme,
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

        if self.quantization_config is not None:
            self.quantization_config.quantization_status = QuantizationStatus.COMPRESSED

        # HACK: support float8 dtypes in older transformers versions.
        transformers.modeling_utils.dtype_byte_size = new_dtype_byte_size

        return state_dict

    def decompress(self, model_path: str | Path | dict[str, Tensor], model: Module):
        if isinstance(model_path, (str, Path)):
            model_path = get_safetensors_folder(str(model_path))

        quant_compressor = self._primary_quantization_compressor()
        if quant_compressor is None:
            return

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
            load_pretrained_quantization_parameters(
                model,
                model_path,
                load_weight_qparams=False,
            )

        for mod_path, data in tqdm(
            quant_compressor.decompress(model_path, names_to_scheme=names_to_scheme),
            desc="Decompressing model",
        ):
            module = operator.attrgetter(mod_path)(model)
            params_device = next(module.parameters()).device
            device = "cpu" if has_offloaded_params(module) else params_device

            for param_name, param_data in data.items():
                if not hasattr(module, param_name):
                    continue

                if param_name == "weight":
                    delattr(module, param_name)
                    requires_grad = param_data.dtype in (
                        torch.float16,
                        torch.float32,
                        torch.bfloat16,
                    )
                    param = torch.nn.Parameter(
                        param_data.to(device),
                        requires_grad=requires_grad,
                    )
                    module.register_parameter(param_name, param)
                else:
                    update_offload_parameter(module, param_name, param_data)

        def freeze_quantization_status(module):
            module.quantization_status = QuantizationStatus.FROZEN

        model.apply(freeze_quantization_status)
        setattr(model, QUANTIZATION_CONFIG_NAME, self.quantization_config)

    def update_config(self, save_directory: str):
        if not any((self.quantization_config, self.sparsity_config, self.transform_config)):
            return

        config_file_path = os.path.join(save_directory, CONFIG_NAME)
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as file:
                config_data = json.load(file)
        else:
            config_data = {}

        qconfig_data = (
            self.quantization_config.model_dump(exclude=["quant_method"])
            if self.quantization_config is not None
            else {}
        )
        sconfig_data = (
            self.sparsity_config.model_dump() if self.sparsity_config is not None else {}
        )
        tconfig_data = (
            self.transform_config.model_dump() if self.transform_config is not None else {}
        )

        config_data[QUANTIZATION_CONFIG_NAME] = {
            COMPRESSION_VERSION_NAME: compressed_tensors.__version__,
            QUANTIZATION_METHOD_NAME: DEFAULT_QUANTIZATION_METHOD,
            SPARSITY_CONFIG_NAME: sconfig_data,
            TRANSFORM_CONFIG_NAME: tconfig_data,
            **qconfig_data,
        }

        with open(config_file_path, "w") as config_file:
            json.dump(config_data, config_file, indent=2, sort_keys=True)


def map_module_to_scheme(model: Module) -> dict[str, QuantizationScheme]:
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
