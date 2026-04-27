# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example demonstrating multiple converters applied sequentially.

This example shows how to:
1. Define custom converters
2. Pass multiple converters to convert_checkpoint
3. Apply transformations in sequence

In practice, you might use multiple converters to:
- Convert from one format, then apply additional transformations
- Quantize different layer types with different schemes
- Chain preprocessing steps before final conversion
"""

from typing import Iterable

import torch
from compressed_tensors.entrypoints.convert import convert_checkpoint, Converter
from compressed_tensors.utils.match import match_name


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

    def create_config(self, config: dict[str, object]) -> dict[str, object]:
        """
        This converter doesn't change quantization format,
        so pass through the config unchanged.
        """
        return config


class DtypeConverter(Converter):
    """
    Example converter that changes dtype of specific tensors.
    Demonstrates how converters can be chained to perform
    multiple transformations.
    """

    def __init__(
        self,
        targets: Iterable[str] = tuple(),
        ignore: Iterable[str] = tuple(),
        target_dtype: torch.dtype = torch.bfloat16,
    ):
        self.targets = targets
        self.ignore = ignore
        self.target_dtype = target_dtype

    def process(self, tensors: dict[str, torch.Tensor]):
        """Convert targeted tensors to target dtype."""
        for name, tensor in list(tensors.items()):
            module_name = name.rsplit(".", 1)[0]

            is_targeted = any(
                match_name(module_name, target) for target in self.targets
            )
            is_ignored = any(match_name(module_name, ignore) for ignore in self.ignore)

            if is_targeted and not is_ignored:
                # Convert dtype
                tensors[name] = tensor.to(self.target_dtype)

    def validate(self, tensors: dict[str, torch.Tensor]):
        """Validation pass."""
        pass

    def get_dependencies(self, weight_name: str) -> set[str]:
        """No additional dependencies."""
        return set()

    def create_config(self, config: dict[str, object]) -> dict[str, object]:
        """Pass through config unchanged."""
        return config


# Example 1: Single converter (traditional usage)
def example_single_converter():
    """Traditional single converter usage."""
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    SAVE_DIR = "Qwen2.5-0.5B-scaled"

    convert_checkpoint(
        model_stub=MODEL_ID,
        save_directory=SAVE_DIR,
        converter=ScaleBiasConverter(
            targets=["re:.*mlp.*"],
            scale_factor=1.1,
        ),
        max_workers=4,
    )
    print(f"✓ Converted with single converter to {SAVE_DIR}")


# Example 2: Multiple converters applied sequentially
def example_multi_converter():
    """
    Multiple converters applied in sequence.

    This applies:
    1. ScaleBiasConverter to MLP layers
    2. DtypeConverter to attention layers

    The converters process tensors in order, each seeing
    the results of previous converters.
    """
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    SAVE_DIR = "Qwen2.5-0.5B-multi-converted"

    convert_checkpoint(
        model_stub=MODEL_ID,
        save_directory=SAVE_DIR,
        converter=[
            # First converter: scale MLP weights
            ScaleBiasConverter(
                targets=["re:.*mlp.*"],
                scale_factor=1.05,
            ),
            # Second converter: convert attention layers to float32
            DtypeConverter(
                targets=["re:.*self_attn.*"],
                target_dtype=torch.float32,
            ),
        ],
        max_workers=4,
    )
    print(f"✓ Converted with multiple converters to {SAVE_DIR}")


# Example 3: Real-world scenario - combining actual converters
def example_practical_multi_converter():
    """
    Practical example: dequantize FP8 model, then apply custom scaling.

    This demonstrates a real use case where you might:
    1. First dequantize from a quantized format
    2. Then apply custom transformations
    """
    from compressed_tensors.entrypoints.convert import FP8BlockDequantizer

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
    print(f"✓ Dequantized and scaled to {SAVE_DIR}")


if __name__ == "__main__":
    print("Multi-Converter Example\n")
    print("=" * 60)

    # Choose which example to run:

    # Uncomment to run single converter example
    # example_single_converter()

    # Uncomment to run multi-converter example
    # example_multi_converter()

    # Uncomment to run practical multi-converter example
    # example_practical_multi_converter()

    print("\nℹ️  Uncomment one of the example functions above to run")
    print("   Examples use small models (0.5B-4B parameters)")
