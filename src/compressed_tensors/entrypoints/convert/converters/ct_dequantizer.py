# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from typing import Iterable

import torch
from compressed_tensors.base import QUANTIZATION_CONFIG_NAME
from compressed_tensors.compressors import BaseCompressor
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.quantization import QuantizationConfig, QuantizationMetadata
from compressed_tensors.utils.match import match_name, match_quantizable_tensors
from compressed_tensors.utils.safetensors_load import get_checkpoint_files
from transformers.file_utils import CONFIG_NAME


class CompressedTensorsDequantizer(Converter):
    """
    Dequantize a checkpoint in the compressed-tensors quant format
    The resultant weights will be stored in user-provided dtype
    """

    def __init__(
        self,
        model_stub: str | os.PathLike,
        ignore: Iterable[str] = tuple(),
        quant_config_key: str = QUANTIZATION_CONFIG_NAME,
        dtype=torch.bfloat16,
    ):
        self.dtype = dtype

        # load quantization config from model_stub
        model_files = get_checkpoint_files(model_stub)
        if CONFIG_NAME in model_files:
            resolved_path = model_files[CONFIG_NAME]
        elif "params.json" in model_files:
            resolved_path = model_files["params.json"]
        else:
            raise ValueError("Could not find config.json file")

        with open(resolved_path, "r") as f:
            config = json.load(f)
            quant_config = config
            for key_segment in quant_config_key.split("."):
                quant_config = quant_config[key_segment]
            self.quant_config = QuantizationConfig.model_validate(quant_config)

        # hydrate with additional ignore and inferred scheme formats
        self.quant_config.ignore += list(ignore)
        for scheme in self.quant_config.config_groups.values():
            scheme.format = CompressionFormat(
                infer_module_format(torch.nn.Linear, scheme)
            )

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Dequantize compressed tensors to full-precision weight tensors in dtype
        provided to constructor
        """
        dequantized_tensors = {}

        for scheme in self.quant_config.config_groups.values():
            compressor = BaseCompressor.get_value_from_registry(scheme.format)
            param_names = compressor.compression_param_names(scheme)
            for module_name, tensor_name in match_quantizable_tensors(
                tensors,
                self.quant_config.ignore,
                scheme.targets,
                allow_nonquantizable=True,
            ):
                param_name = tensor_name.rsplit(".", 1)[1]

                if param_name != param_names[0]:
                    continue

                # Create state dict of param_name -> torch.Tensor
                state_dict = {
                    f"{param_name}": tensors.pop(f"{module_name}.{param_name}")
                    for param_name in compressor.compression_param_names(scheme)
                }

                dequantized_state_dict = compressor.decompress(state_dict, scheme)

                # Add only weight param to dequantized tensors
                dequantized_tensors[f"{module_name}.weight"] = dequantized_state_dict[
                    "weight"
                ]

        # Copy over any remaining ignored/untargeted tensors,
        # skipping lingering activation and kv cache qparams
        for name, tensor in tensors.items():
            if any(
                [
                    name.endswith(param_name)
                    for param_name in QuantizationMetadata.all_qparam_names()
                ]
            ):
                continue
            dequantized_tensors[name] = tensor

        return dequantized_tensors

    def validate(self, tensors: dict[str, torch.Tensor]):
        """
        Ensure all tensor names of targeted layers are expected and no
        untargeted layers have unexpected tensor names
        """
        consumed_keys = set()
        matched_modules = set()
        for scheme in self.quant_config.config_groups.values():
            compressor = BaseCompressor.get_value_from_registry(scheme.format)
            param_names = compressor.compression_param_names(scheme)
            for module_name, _ in match_quantizable_tensors(
                tensors,
                self.quant_config.ignore,
                scheme.targets,
            ):
                matched_modules.add(module_name)
                for param_name in param_names:
                    expected_key = f"{module_name}.{param_name}"

                    if expected_key not in tensors:
                        raise ValueError(f"Expected key {expected_key} not found")

                    consumed_keys.add(expected_key)

        unconsumed_tensor_names = [
            name
            for name in tensors
            if name not in consumed_keys and name.rsplit(".", 1)[0] in matched_modules
        ]
        assert len(unconsumed_tensor_names) == 0, (
            f"Found {len(unconsumed_tensor_names)} unconsumed keys -- "
            f"{unconsumed_tensor_names}"
        )

        return

    def create_config(self) -> QuantizationConfig | None:
        return None

    def get_dependencies(self, weight_name: str) -> set[str]:
        """
        Dependencies are determined by the associated compressor's
        compression_param_names. The first param name in the returned list
        is treated as the root param, and is usually "weight" or "weight_packed"

        If weight_name is untargeted or ignored, an empty set is returned
        """
        module_name, param_name = weight_name.rsplit(".", 1)

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
