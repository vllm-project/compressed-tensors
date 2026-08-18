# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from typing import Iterable, Optional

import pydantic
import torch
from compressed_tensors.compressors import BaseCompressor
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.quantization import (
    KVCacheScaleType,
    QuantizationConfig,
    QuantizationMetadata,
)
from compressed_tensors.utils.match import match_name, match_quantizable_tensors
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_quantization_config,
)
from loguru import logger
from transformers.file_utils import CONFIG_NAME
from transformers.modeling_utils import local_torch_dtype


class CompressedTensorsDequantizer(Converter):
    """
    Dequantize a checkpoint in the compressed-tensors quant format
    The resultant weights will be stored in user-provided dtype
    """

    def __init__(
        self,
        model_stub: str | os.PathLike,
        ignore: Iterable[str] = tuple(),
        dtype: Optional[torch.dtype] = None,
    ):
        # load quantization config from model_stub
        model_files = get_checkpoint_files(model_stub)
        if CONFIG_NAME in model_files:
            config_resolved_path = model_files[CONFIG_NAME]
        elif "params.json" in model_files:
            config_resolved_path = model_files["params.json"]
        else:
            raise ValueError("Could not find config.json file")

        if dtype is None:
            with open(config_resolved_path, "r") as f:
                config = json.load(f)
            dtype_str = config.get("dtype", config.get("torch_dtype", "bfloat16"))
            dtype = getattr(torch, dtype_str)
        self.dtype = dtype

        quant_config_data = get_quantization_config(config_resolved_path)
        if quant_config_data is None:
            raise ValueError("Could not find quantization_config in config.json")

        try:
            self.quant_config = QuantizationConfig.model_validate(quant_config_data)
        except pydantic.ValidationError as e:
            raise ValueError(
                "Model quantization config was found, but it does not match expected "
                "compressed-tensors quantization format"
            ) from e

        # hydrate with additional ignore and inferred scheme formats
        self.quant_config.ignore += list(ignore)
        for scheme in self.quant_config.config_groups.values():
            scheme.format = CompressionFormat(
                infer_module_format(torch.nn.Linear, scheme)
            )

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Dequantize, then assert the result carries no leftover quantization
        params on non-ignored modules. A residual qparam means the configured
        targets missed a module that still holds compressed weights. Missing
        params surface as a KeyError from process and are re-raised as a
        ValueError so CI catches them. Returns the dequantized tensors so
        chained converters observe the resulting (uncompressed) format.
        """
        try:
            tensors = self.process(tensors)
        except KeyError as e:
            raise ValueError(f"Missing expected compression param {e}") from e

        qparam_names = set(QuantizationMetadata.all_qparam_names())
        residual = [
            name
            for name in tensors
            if name.rpartition(".")[-1] in qparam_names
            and not any(
                match_name(name.rpartition(".")[0], ignore)
                for ignore in self.quant_config.ignore
            )
        ]
        if residual:
            raise ValueError(
                f"Found {len(residual)} residual quantization param(s) after "
                f"dequantization, indicating untargeted or orphan qparams: {residual}"
            )

        return tensors

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Dequantize compressed tensors to full-precision weight tensors in dtype
        provided to constructor
        """
        dequantized_tensors = {}

        with local_torch_dtype(self.dtype):
            for scheme in self.quant_config.config_groups.values():
                compressor = BaseCompressor.get_value_from_registry(scheme.format)
                param_names = compressor.compression_param_names(scheme)
                for module_name, tensor_name in match_quantizable_tensors(
                    tensors,
                    ignore=self.quant_config.ignore,
                    targets=scheme.targets,
                    param_targets=[param_names[0]],
                ):
                    # Create state dict of param_name -> torch.Tensor
                    state_dict = {
                        f"{param_name}": tensors.pop(f"{module_name}.{param_name}")
                        for param_name in param_names
                    }

                    dequantized_state_dict = compressor.decompress(state_dict, scheme)

                    # Add only weight param to dequantized tensors
                    weight = dequantized_state_dict["weight"]
                    dequantized_tensors[f"{module_name}.weight"] = weight
                    if weight.dtype != self.dtype:
                        logger.warning(
                            f"{module_name} decompressed dtype ({weight.dtype}) "
                            f"does not match model dtype ({self.dtype})"
                        )

        # Copy over any remaining ignored/untargeted tensors, skipping kv cache qparams
        kv_cache_param_names = [v.value for v in KVCacheScaleType]
        for name, tensor in tensors.items():
            if any([name.endswith(param_name) for param_name in kv_cache_param_names]):
                continue
            dequantized_tensors[name] = tensor

        return dequantized_tensors

    def update_config(
        self, config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        return None  # dequantizing removes quantization

    def get_dependencies(self, weight_name: str) -> set[str]:
        """
        Dependencies are determined by the associated compressor's
        compression_param_names. The first param name in the returned list
        is treated as the root param, and is usually "weight" or "weight_packed"

        If weight_name is untargeted or ignored, an empty set is returned
        """
        module_name, _, param_name = weight_name.rpartition(".")

        if any(
            [match_name(module_name, ignore) for ignore in self.quant_config.ignore]
        ):
            return set()

        for scheme in self.quant_config.config_groups.values():
            compressor = BaseCompressor.get_value_from_registry(scheme.format)
            compression_param_names = compressor.compression_param_names(scheme)

            if "Linear" in scheme.targets or any(
                [match_name(module_name, target) for target in scheme.targets]
            ):
                if param_name == compression_param_names[0]:
                    return set(
                        f"{module_name}.{param_name}"
                        for param_name in compression_param_names[1:]
                    )
                else:
                    return set()
        return set()
