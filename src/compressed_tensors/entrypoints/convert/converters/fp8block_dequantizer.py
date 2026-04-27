# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from typing import Iterable

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.entrypoints.convert.converters.helpers import (
    merge_quantization_config,
)
from compressed_tensors.quantization import QuantizationConfig, QuantizationStatus
from compressed_tensors.quantization.utils.helpers import (
    maybe_pad_tensor_for_block_quant,
)
from compressed_tensors.utils.match import match_name, match_quantizable_tensors


class FP8BlockDequantizer(Converter):
    """
    Dequantize a checkpoint that has been block-quantized with FP8 quant_method
    The resultant weights will be stored in user-provided dtype
    """

    def __init__(
        self,
        ignore: Iterable[str] = tuple(),
        targets: Iterable[str] = tuple(),
        weight_block_size: tuple[int] = (128, 128),
        dtype=torch.bfloat16,
    ):
        warnings.warn(
            "FP8BlockDequantizer assumes the resulting model will not use "
            "FP8 quantization. The quantization config will be removed if "
            "the quantization_method is 'fp8'.",
            UserWarning,
        )
        self.ignore = ignore
        self.targets = targets
        self.weight_block_size = weight_block_size
        self.dtype = dtype

    def process(self, tensors: dict[str, torch.Tensor]):
        """
        Map the modelopt NVFP4 tensors to the appropriate compressed-tensors
        NVFP4 format.
        Some tensors require rename, some require inversion
        - 1 / input_scale -> input_global_scale
        - weight -> weight_packed
        - 1 / weight_scale_2 -> weight_global_scale
        """
        for module_name, name in match_quantizable_tensors(
            tensors, self.ignore, self.targets, allow_nonquantizable=True
        ):
            param_name = name.rsplit(".", 1)[-1]

            if param_name == "weight":
                # weight * weight_scale_inv -> dequantized weight
                tensors[f"{module_name}.weight"] = self._create_dequantized_weight(
                    tensors[f"{module_name}.weight"],
                    tensors[f"{module_name}.weight_scale_inv"],
                )
                del tensors[f"{module_name}.weight_scale_inv"]

    def validate(self, tensors: dict[str, torch.Tensor]):
        """
        Ensure all tensor names of targeted layers are expected and no
        untargeted layers have unexpected tensor names
        """
        allowed_names = ["weight", "weight_scale_inv"]

        targeted_names = [
            name
            for _, name in match_quantizable_tensors(
                tensors, self.ignore, self.targets, allow_nonquantizable=True
            )
        ]
        for name in targeted_names:
            module_name, param_name = name.rsplit(".", 1)

            if param_name not in allowed_names:
                raise ValueError(f"Found unexpected targeted tensor {name}")
            if (
                param_name == "weight"
                and f"{module_name}.weight_scale_inv" not in tensors
            ):
                raise ValueError(
                    f"Found weight without corresponding weight_scale_inv {name}"
                )
            if (
                param_name == "weight_scale_inv"
                and f"{module_name}.weight" not in tensors
            ):
                raise ValueError(
                    f"Found weight_scale_inv without corresponding weight {name}"
                )

        disallowed_names = ["weight_scale_inv"]
        untargeted_names = [
            name for name in tensors.keys() if name not in targeted_names
        ]
        for name in untargeted_names:
            param_name = name.rsplit(".", 1)[-1]

            if param_name in disallowed_names:
                raise ValueError(f"Found unexpected non-targeted tensor {name}")

    def create_config(self, config: dict) -> dict:
        # Only remove quantization config if the quantization method is fp8
        quantization_config = config.get("quantization_config")
        if quantization_config and quantization_config["quant_method"] == "fp8":
            return {}

        return config

    def get_dependencies(self, weight_name: str) -> set[str]:
        module_name, suffix = weight_name.rsplit(".", 1)
        if (
            any([match_name(module_name, target) for target in self.targets])
            and not any([match_name(module_name, ignore) for ignore in self.ignore])
            and suffix == "weight"
        ):
            return {f"{module_name}.weight_scale_inv"}
        return set()

    def _create_dequantized_weight(
        self, weight: torch.Tensor, weight_scale_inv: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert fp8 weight and fp32 weight_scale_inv tensors into
        corresponding dequantized weight tensor.
        Tensors are upscaled to fp32 before scaling

        :return: dequantized tensor in self.dtype and same shape as input weight tensor
        """
        original_shape = weight.shape
        block_height, block_width = self.weight_block_size

        # Pad tensor if dimensions are not evenly divisible by block size
        weight = maybe_pad_tensor_for_block_quant(weight, tuple(self.weight_block_size))
        padded_shape = weight.shape

        # Reshape into blocks of shape:
        # (num_rows_blocks, block_height, num_cols_blocks, block_width)
        num_rows_blocks = padded_shape[0] // block_height
        num_cols_blocks = padded_shape[1] // block_width
        weight_blocks = weight.reshape(
            num_rows_blocks,
            block_height,
            num_cols_blocks,
            block_width,
        ).transpose(
            1, 2
        )  # (num_rows_blocks, num_cols_blocks, block_height, block_width)

        # Expand scale_inv for broadcasting over block dimensions
        # weight_scale_inv shape: (num_rows_blocks, num_cols_blocks)
        # Expand to: (num_rows_blocks, num_cols_blocks, 1, 1)
        scale_inv_expanded = weight_scale_inv.unsqueeze(-1).unsqueeze(-1)

        # Dequantize: weight_bf16 = weight_fp8 * weight_scale_inv
        dequantized_blocks = (
            weight_blocks.to(torch.float32) * scale_inv_expanded.to(torch.float32)
        ).to(self.dtype)

        # Restore padded shape
        dequantized = dequantized_blocks.transpose(1, 2).reshape(padded_shape)

        # Truncate to original dimensions if padding was applied
        if original_shape != padded_shape:
            dequantized = dequantized[tuple([slice(v) for v in original_shape])]

        return dequantized
