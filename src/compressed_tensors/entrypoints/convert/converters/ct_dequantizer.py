# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors import BaseCompressor
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.utils.match import match_name, match_quantizable_tensors


class CompressedTensorsDequantizer(Converter):
    """
    Dequantize a checkpoint in the compressed-tensors quant format
    The resultant weights will be stored in user-provided dtype
    """

    def __init__(
        self,
        quant_config: QuantizationConfig,
        dtype=torch.bfloat16,
    ):
        self.quant_config = quant_config
        self.dtype = dtype

    def process(self, tensors: dict[str, torch.Tensor]):
        """
        Dequantize compressed tensors to full-precision weight tensors in dtype
        provided to constructor
        """
        dequantized_tensors = {}

        for scheme in self.quant_config.config_groups.values():
            compressor = BaseCompressor.get_value_from_registry(scheme.format.value)
            for module_name, _ in match_quantizable_tensors(
                tensors,
                self.quant_config.ignore,
                scheme.targets,
            ):
                # Create state dict of param_name -> torch.Tensor
                state_dict = {
                    f"{param_name}": tensors.pop(f"{module_name}.{param_name}")
                    for param_name in compressor.compression_param_names(scheme)
                }

                dequantized_state_dict = compressor.decompress(state_dict, scheme)

                # Add to dequantized tensors
                dequantized_tensors.update(
                    {
                        f"{module_name}.{param_name}": dequantized_state_dict[
                            param_name
                        ]
                        for param_name in dequantized_state_dict
                    }
                )

        # Copy over any remaining ignored/untargeted tensors
        return dequantized_tensors | tensors

    def validate(self, tensors: dict[str, torch.Tensor]):
        """
        Ensure all tensor names of targeted layers are expected and no
        untargeted layers have unexpected tensor names
        """
        consumed_keys = set()
        for scheme in self.quant_config.config_groups.values():
            compressor = BaseCompressor.get_value_from_registry(scheme.format.value)
            param_names = compressor.compression_param_names(scheme)
            for module_name, _ in match_quantizable_tensors(
                tensors,
                self.quant_config.ignore,
                scheme.targets,
            ):
                for param_name in param_names:
                    expected_key = f"{module_name}.{param_name}"

                    if expected_key not in tensors:
                        raise ValueError(f"Expected key {expected_key} not found")

                    consumed_keys.add(expected_key)

        # Assert all targeted tensors have been consumed
        for scheme in self.quant_config.config_groups.values():
            unconsumed_tensor_names = [
                tensor_name
                for _, tensor_name in match_quantizable_tensors(
                    tensors,
                    self.quant_config.ignore,
                    scheme.targets,
                    allow_nonquantizable=True,
                )
                if tensor_name not in consumed_keys
            ]
            assert (
                len(unconsumed_tensor_names) > 0
            ), f"Found f{len(unconsumed_tensor_names)} unconsumed keys -- "
            f"{unconsumed_tensor_names}"

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

        for scheme in self.quant_config.config_groups.values():
            compressor = BaseCompressor.get_value_from_registry(scheme.format.value)
            compression_param_names = compressor.compression_param_names(scheme)
            if (
                any([match_name(module_name, target) for target in scheme.targets])
                and not any(
                    [
                        match_name(module_name, ignore)
                        for ignore in self.quant_config.ignore
                    ]
                )
                and param_name == compression_param_names[0]
            ):
                return set(
                    f"{module_name}.{param_name}"
                    for param_name in compression_param_names[1:]
                )
        return set()
