# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Iterable

import torch
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.quantization import QuantizationConfig
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
        self.ignore = ignore
        self.targets = targets
        self.weight_block_size = weight_block_size
        self.dtype = dtype

        self.param_names = ["weight", "scale"]

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Dequantize the fp8 block tensors (weight, scale) to full-precision
        weight tensors in dtype provided to constructor
        """
        for module_name, name in match_quantizable_tensors(
            tensors, self.ignore, self.targets, param_targets=self.param_names
        ):
            param_name = name.rpartition(".")[-1]

            if param_name == "weight":
                # weight * scale -> dequantized weight
                tensors[f"{module_name}.weight"] = self._create_dequantized_weight(
                    tensors[f"{module_name}.weight"],
                    tensors[f"{module_name}.scale"],
                )
                del tensors[f"{module_name}.scale"]

        return tensors

    def validate(self, tensors: dict[str, torch.Tensor]):
        """
        Ensure all tensor names of targeted layers are expected and no
        untargeted layers have unexpected tensor names
        """

        targeted_names = [
            name
            for _, name in match_quantizable_tensors(
                tensors, self.ignore, self.targets, param_targets=self.param_names
            )
        ]
        for name in targeted_names:
            module_name, _, param_name = name.rpartition(".")

            if (
                param_name == "weight"
                and f"{module_name}.scale" not in tensors
            ):
                raise ValueError(
                    f"Found weight without corresponding scale {name}"
                )
            if (
                param_name == "scale"
                and f"{module_name}.weight" not in tensors
            ):
                raise ValueError(
                    f"Found scale without corresponding weight {name}"
                )

        # this step is problematic for things like MTP layers
        # ideally, we add multiple converter support, support a dropping layer converter
        # and have this error point to adding that converter
        # disallowed_names = ["scale"]
        # untargeted_names = [
        #     name for name in tensors.keys() if name not in targeted_names
        # ]
        # for name in untargeted_names:
        #     param_name = name.rsplit(".", 1)[-1]

        #     if param_name in disallowed_names:
        #         raise ValueError(f"Found unexpected non-targeted tensor {name}")

    def create_config(self) -> QuantizationConfig | None:
        return None

    def get_dependencies(self, weight_name: str) -> set[str]:
        module_name, _, param_name = weight_name.rpartition(".")
        if (
            any([match_name(module_name, target) for target in self.targets])
            and not any([match_name(module_name, ignore) for ignore in self.ignore])
            and param_name == "weight"
        ):
            return {f"{module_name}.scale"}
        return set()

    def _create_dequantized_weight(
        self, weight: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert fp8 weight and fp32 scale tensors into
        corresponding dequantized weight tensor.
        Tensors are upscaled to fp32 before scaling

        :return: dequantized tensor in self.dtype and same shape as input weight tensor
        """
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize

        return Fp8Dequantize(None)._dequantize_one(weight, scale)

