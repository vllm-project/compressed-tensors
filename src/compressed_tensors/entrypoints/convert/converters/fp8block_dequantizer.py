# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import Iterable

import torch
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.quantization.utils.helpers import (
    maybe_pad_tensor_for_block_quant,
)
from compressed_tensors.utils.match import match_name, match_quantizable_tensors
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
)


# param names that indicate a module carries a block-quantization scale
_SCALE_PARAM_NAMES = ("weight_scale", "weight_scale_inv")


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
        raise ValueError("Use `from_pretrained`")
        self.resolved_targets = ...
        self.ignore = ignore
        self.targets = targets
        self.weight_block_size = weight_block_size
        self.dtype = dtype

        self.param_names = ["weight", "weight_scale_inv"]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | os.PathLike,
        ignore: Iterable[str] = tuple(),
        weight_block_size: tuple[int] = (128, 128),
        dtype=torch.bfloat16,
    ) -> FP8BlockDequantizer:
        """
        Build the converter by scanning the checkpoint's weight map and targeting
        every module that carries a block-quantization scale (a ``weight_scale``
        or ``weight_scale_inv`` parameter).

        :param model_name_or_path: HuggingFace stub or local checkpoint path
        :param ignore: module names (regex allowed) to exclude from dequantization
        :param weight_block_size: block dimensions used during quantization
        :param dtype: dtype to store the dequantized weights in
        """
        model_files = get_checkpoint_files(model_name_or_path)
        weight_map = get_weight_map(model_files)

        # collect the module name for every scale parameter in the checkpoint
        targets = sorted(
            {
                name.rpartition(".")[0]
                for name in weight_map
                if name.rpartition(".")[-1] in _SCALE_PARAM_NAMES
            }
        )

        instance = cls.__new__()
        instance.ignore = ignore
        ...
        return instance
        #     ignore=ignore,
        #     targets=targets,
        #     weight_block_size=weight_block_size,
        #     dtype=dtype,
        # )

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Dequantize, then assert no leftover weight_scale_inv remains on a
        non-ignored module. A residual weight_scale_inv means the configured
        targets missed a block-quantized weight. A targeted weight missing its
        weight_scale_inv surfaces as a KeyError from process and is re-raised as
        a ValueError so CI catches it. Returns the processed tensors so chained
        converters observe the resulting format.
        """
        try:
            tensors = self.process(tensors)
        except KeyError as e:
            raise ValueError(f"Missing expected weight_scale_inv {e}") from e

        residual = [
            name
            for name in tensors
            if name.rpartition(".")[-1] == "weight_scale_inv"
            and not any(match_name(name.rpartition(".")[0], ign) for ign in self.ignore)
        ]
        if residual:
            raise ValueError(
                f"Found {len(residual)} residual weight_scale_inv after "
                f"dequantization, indicating untargeted or orphan scales: {residual}"
            )

        return tensors

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Dequantize the fp8 block tensors (weight, weight_scale_inv) to full-precision
        weight tensors in dtype provided to constructor
        """
        for module_name, name in match_quantizable_tensors(
            tensors, self.ignore, self.targets, param_targets=self.param_names
        ):
            param_name = name.rpartition(".")[-1]

            if param_name == "weight":
                # weight * weight_scale_inv -> dequantized weight
                tensors[f"{module_name}.weight"] = self._create_dequantized_weight(
                    tensors[f"{module_name}.weight"],
                    tensors[f"{module_name}.weight_scale_inv"],
                )
                del tensors[f"{module_name}.weight_scale_inv"]

        return tensors

    def update_config(
        self, config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        return None  # dequantizing removes quantization

    def get_dependencies(self, weight_name: str) -> set[str]:
        module_name, _, param_name = weight_name.rpartition(".")
        if (
            any([match_name(module_name, target) for target in self.targets])
            and not any([match_name(module_name, ignore) for ignore in self.ignore])
            and param_name == "weight"
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
