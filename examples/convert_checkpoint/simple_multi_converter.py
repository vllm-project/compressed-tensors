# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Minimal example showing multi-converter support with Qwen2.5-0.5B.

This demonstrates passing a list of converters to convert_checkpoint,
which will be applied sequentially to the model weights.
"""

from typing import Iterable

import torch
from compressed_tensors.entrypoints.convert import convert_checkpoint, Converter
from compressed_tensors.utils.match import match_name, match_quantizable_tensors


class PrintInfoConverter(Converter):
    """
    A simple converter that prints information about tensors
    and demonstrates the converter interface.
    """

    def __init__(self, label: str = "Converter", targets: Iterable[str] = tuple()):
        self.label = label
        self.targets = targets or ["re:.*"]  # default to all layers

    def process(self, tensors: dict[str, torch.Tensor]):
        """Print info about targeted tensors."""
        matched = list(
            match_quantizable_tensors(tensors, [], self.targets, allow_nonquantizable=True)
        )
        print(f"\n[{self.label}] Processing {len(matched)} tensors")
        if matched:
            # Show first few as examples
            for module_name, tensor_name in matched[:3]:
                tensor = tensors[tensor_name]
                print(f"  {tensor_name}: shape={tensor.shape}, dtype={tensor.dtype}")

    def validate(self, tensors: dict[str, torch.Tensor]):
        """Validation - just print info."""
        print(f"[{self.label}] Validating {len(tensors)} tensors")

    def get_dependencies(self, weight_name: str) -> set[str]:
        """No dependencies."""
        return set()

    def update_config(self, config: dict[str, object]) -> dict[str, object]:
        """Pass through config unchanged."""
        return config


class ScaleConverter(Converter):
    """
    A simple converter that scales weight tensors by a constant factor.
    """

    def __init__(
        self,
        scale_factor: float = 1.0,
        targets: Iterable[str] = tuple(),
    ):
        self.scale_factor = scale_factor
        self.targets = targets or ["re:.*"]

    def process(self, tensors: dict[str, torch.Tensor]):
        """Scale targeted weight tensors."""
        count = 0
        for module_name, name in match_quantizable_tensors(
            tensors, [], self.targets, allow_nonquantizable=True
        ):
            if name.endswith(".weight"):
                tensors[name] = tensors[name] * self.scale_factor
                count += 1
        print(f"  Scaled {count} weight tensors by {self.scale_factor}")

    def validate(self, tensors: dict[str, torch.Tensor]):
        """No validation needed."""
        pass

    def get_dependencies(self, weight_name: str) -> set[str]:
        """No dependencies."""
        return set()

    def update_config(self, config: dict[str, object]) -> dict[str, object]:
        """Pass through config unchanged."""
        return config


# Minimal example with a very small model
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SAVE_DIR = "Qwen2.5-0.5B-multi-converted"

print("Multi-Converter Example with Qwen2.5-0.5B")
print("=" * 60)

# Apply multiple converters in sequence
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=[
        # First converter: print info about all tensors
        PrintInfoConverter(
            label="Step 1: Info",
            targets=["re:.*"],
        ),
        # Second converter: scale MLP weights
        ScaleConverter(
            scale_factor=1.05,
            targets=["re:.*mlp.*"],
        ),
        # Third converter: print info about attention layers
        PrintInfoConverter(
            label="Step 2: Attn Info",
            targets=["re:.*self_attn.*"],
        ),
    ],
    max_workers=2,
)

print(f"\n✓ Successfully converted model to {SAVE_DIR}")
print(f"  Each converter was applied sequentially to the model weights.")
