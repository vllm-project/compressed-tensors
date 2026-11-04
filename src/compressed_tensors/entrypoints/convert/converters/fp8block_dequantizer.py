# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Iterable

import torch
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.quantization.utils.helpers import (
    maybe_pad_tensor_for_block_quant,
)
from compressed_tensors.utils.match import match_name
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
)


class FP8BlockDequantizer(Converter):
    """
    Dequantize a checkpoint that has been block-quantized with FP8.

    Checkpoints may store block scales as weight_scale_inv (Transformers) or
    qscale_weight (Mistral). The resultant weights use the requested dtype.
    """

    SCALE_NAMES = ("weight_scale_inv", "qscale_weight")

    def __init__(
        self,
        ignore: Iterable[str] = tuple(),
        targets: Iterable[str] = tuple(),
        weight_block_size: tuple[int] = (128, 128),
        dtype=torch.bfloat16,
        scale_name: str = "weight_scale_inv",
    ):
        self.ignore = ignore
        self.targets = targets
        self.weight_block_size = weight_block_size
        self.dtype = dtype
        self.scale_name = scale_name
        self._checkpoint_targets: frozenset[str] | None = None
        self._checkpoint_scale_names: dict[str, str] | None = None

        self.param_names = ["weight", scale_name]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | os.PathLike,
        ignore: Iterable[str] = tuple(),
        weight_block_size: tuple[int] = (128, 128),
        dtype=torch.bfloat16,
        scale_name: str | None = None,
    ) -> "FP8BlockDequantizer":
        """
        Build the converter by scanning the checkpoint weight map and targeting
        every module with a supported FP8 block scale tensor. When scale_name
        is omitted, supported names are detected automatically.

        :param model_name_or_path: HuggingFace stub or local checkpoint path
        :param ignore: module names or regexes to exclude from dequantization
        :param weight_block_size: block dimensions used during quantization
        :param dtype: dtype to store dequantized weights in
        :param scale_name: optional scale tensor suffix to use instead of auto-detection
        """
        model_files = get_checkpoint_files(model_name_or_path)
        weight_map = get_weight_map(model_files)

        scale_names = (scale_name,) if scale_name is not None else cls.SCALE_NAMES
        checkpoint_scale_names: dict[str, str] = {}
        for name in weight_map:
            module_name, _, param_name = name.rpartition(".")
            if param_name not in scale_names:
                continue
            previous_scale_name = checkpoint_scale_names.get(module_name)
            if previous_scale_name is not None and previous_scale_name != param_name:
                raise ValueError(
                    f"Found multiple FP8 block scale tensors for {module_name}: "
                    f"{previous_scale_name}, {param_name}"
                )
            checkpoint_scale_names[module_name] = param_name

        if not checkpoint_scale_names:
            expected = scale_name or "weight_scale_inv"
            raise ValueError(
                f"No {expected} tensors found in checkpoint {model_name_or_path}; "
                f"supported scale names are {scale_names}"
            )

        targets = frozenset(checkpoint_scale_names)
        converter = cls(
            ignore=ignore,
            targets=targets,
            weight_block_size=weight_block_size,
            dtype=dtype,
            scale_name=next(iter(checkpoint_scale_names.values())),
        )
        converter._checkpoint_targets = targets
        converter._checkpoint_scale_names = checkpoint_scale_names
        converter.param_names = [
            "weight",
            *sorted(set(checkpoint_scale_names.values())),
        ]
        return converter

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Dequantize, then assert no leftover FP8 block scales remain on a
        non-ignored module. A targeted weight missing its scale tensor surfaces
        as a KeyError from process and is re-raised as a ValueError. Returns
        processed tensors so chained converters observe the resulting format.
        """
        try:
            tensors = self.process(tensors)
        except KeyError as e:
            raise ValueError(f"Missing expected FP8 block scale {e}") from e

        residual = [
            name
            for name in tensors
            if name.rpartition(".")[-1] in self._supported_scale_names()
            and not any(match_name(name.rpartition(".")[0], ign) for ign in self.ignore)
        ]
        if residual:
            raise ValueError(
                f"Found {len(residual)} residual FP8 block scales after "
                f"dequantization, indicating untargeted or orphan scales: {residual}"
            )

        return tensors

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Dequantize FP8 block tensors (weight, scale) to full-precision weights.
        """
        for name in list(tensors):
            module_name, _, param_name = name.rpartition(".")
            if (
                param_name != "weight"
                or module_name.endswith("norm")
                or not self._is_targeted(module_name)
            ):
                continue

            scale_name = self._scale_name_for(module_name)
            tensors[name] = self._create_dequantized_weight(
                tensors[name],
                tensors[f"{module_name}.{scale_name}"],
            )
            del tensors[f"{module_name}.{scale_name}"]

        return tensors

    def update_config(
        self, config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        return None  # dequantizing removes quantization

    def get_dependencies(self, weight_name: str) -> set[str]:
        module_name, _, param_name = weight_name.rpartition(".")
        if param_name == "weight" and self._is_targeted(module_name):
            return {f"{module_name}.{self._scale_name_for(module_name)}"}
        return set()

    def _scale_name_for(self, module_name: str) -> str:
        if self._checkpoint_scale_names is not None:
            return self._checkpoint_scale_names.get(module_name, self.scale_name)
        return self.scale_name

    def _supported_scale_names(self) -> set[str]:
        return set(self.SCALE_NAMES) | {self.scale_name}

    def _is_targeted(self, module_name: str) -> bool:
        if any(match_name(module_name, ignore) for ignore in self.ignore):
            return False

        if self._checkpoint_targets is not None:
            return module_name in self._checkpoint_targets

        return (
            len(self.targets) == 0
            or "Linear" in self.targets
            or any(match_name(module_name, target) for target in self.targets)
        )

    def _create_dequantized_weight(
        self, weight: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert an FP8 weight and fp32 block scales into a dequantized weight.
        Tensors are upscaled to fp32 before scaling.

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

        # Expand block scales for broadcasting over block dimensions.
        # Shape: (num_rows_blocks, num_cols_blocks)
        # Expand to: (num_rows_blocks, num_cols_blocks, 1, 1)
        scale_expanded = weight_scale.unsqueeze(-1).unsqueeze(-1)

        # Dequantize: weight_bf16 = weight_fp8 * block_scale.
        dequantized_blocks = (
            weight_blocks.to(torch.float32) * scale_expanded.to(torch.float32)
        ).to(self.dtype)

        # Restore padded shape
        dequantized = dequantized_blocks.transpose(1, 2).reshape(padded_shape)

        # Truncate to original dimensions if padding was applied
        if original_shape != padded_shape:
            dequantized = dequantized[tuple([slice(v) for v in original_shape])]

        return dequantized
