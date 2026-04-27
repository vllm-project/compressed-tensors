# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example demonstrating multiple converters applied sequentially.
1. Dequantize FP8 model
2. Apply custom scaling.
"""

from typing import Iterable

import torch
from compressed_tensors.entrypoints.convert import convert_checkpoint, Converter
from compressed_tensors.utils.match import match_name


from compressed_tensors.entrypoints.convert import FP8BlockDequantizer


class ScaleBiasConverter(Converter):
    """
    Example converter that scales weights by a factor.
    This demonstrates a simple transformation that could be
    chained with other converters.
    """

    def __init__(
        self,
        targets: Iterable[str] = tuple(),
        ignore: Iterable[str] = tuple(),
        scale_factor: float = 1.0,
    ):
        self.targets = targets
        self.ignore = ignore
        self.scale_factor = scale_factor

    def process(self, tensors: dict[str, torch.Tensor]):
        """Apply scaling to targeted weight tensors."""
        for name, tensor in list(tensors.items()):
            module_name, param_name = name.rsplit(".", 1)

            if param_name == "weight":
                is_targeted = any(
                    match_name(module_name, target) for target in self.targets
                )
                is_ignored = any(
                    match_name(module_name, ignore) for ignore in self.ignore
                )

                if is_targeted and not is_ignored:
                    # Scale the weight tensor
                    tensors[name] = tensor * self.scale_factor

    def validate(self, tensors: dict[str, torch.Tensor]):
        """Validation - ensure targeted modules exist."""
        # Simple validation: check that we have some weights
        has_weights = any(name.endswith(".weight") for name in tensors.keys())
        if not has_weights:
            raise ValueError("No weight tensors found in checkpoint")

    def get_dependencies(self, weight_name: str) -> set[str]:
        """No additional dependencies needed for this converter."""
        return set()

    def update_config(self, config: dict[str, object]) -> dict[str, object]:
        """
        This converter doesn't change quantization format,
        so pass through the config unchanged.
        """
        return config

MODEL_ID = "qwen-community/Qwen3-4B-FP8"
SAVE_DIR = "Qwen3-4B-dequantized-scaled"

convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=[
        # First: dequantize from FP8
        FP8BlockDequantizer(
            targets=[
                r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
                r"re:.*self_attn.*\.(q|k|v|o)_proj$",
            ],
            weight_block_size=[128, 128],
        ),
        # Second: apply custom scaling to MLP layers
        ScaleBiasConverter(
            targets=["re:.*mlp.*"],
            scale_factor=0.99,
        ),
    ],
    max_workers=4,
)
