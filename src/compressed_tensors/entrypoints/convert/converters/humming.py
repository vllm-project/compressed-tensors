# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Any, cast

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.utils.match import match_name
from transformers import AutoConfig


__all__ = ["HummingConverter"]


# Map GSQ's `b_dtype` strings to (num_bits, symmetric).
# GSQ stores symmetric INT values as unsigned 2's-complement-shifted ints
# (`q_unsigned = q_signed + 2**(num_bits-1)`), which matches CT's pack layout.
_HUMMING_DTYPES = {
    "uint1": (1, True),
    "uint2": (2, True),
    "uint3": (3, True),
    "uint4": (4, True),
    "uint5": (5, True),
    "uint6": (6, True),
    "uint7": (7, True),
    "uint8": (8, True),
    "int1": (1, True),
    "int2": (2, True),
    "int3": (3, True),
    "int4": (4, True),
    "int5": (5, True),
    "int6": (6, True),
    "int7": (7, True),
    "int8": (8, True),
}


class HummingConverter(Converter):
    """
    Convert "humming"/GSQ checkpoints to compressed-tensors pack-quantized.

    GSQ stores 2..8 bit symmetric INT weights packed into ``int32`` under the plain
    ``.weight`` key, alongside a ``.weight_scale`` tensor (group-scaled along the
    input dim). This layout is bit-equivalent to compressed-tensors' pack-quantized
    format: the only differences are tensor naming and the absence of a
    ``weight_shape`` tensor. This converter therefore renames ``weight`` ->
    ``weight_packed`` and emits ``weight_shape`` from the unpacked dimensions, with
    no bit manipulation on the packed values.

    Reference HF config block:
      {
        "quant_method": "humming",
        "b_dtype": "uint2",
        "weight_scale_group_size": 128,
        "weight_scale_type": "group",
        "has_zero_point": false,
        "ignore": [...]
      }
    """

    def __init__(
        self,
        bits: int,
        group_size: int,
        ignore: Iterable[str] = ("lm_head",),
        targets: Iterable[str] = ("Linear",),
    ):
        if not 1 <= bits <= 8:
            raise ValueError(f"HummingConverter supports bits in [1, 8], got {bits}")

        self.bits = bits
        self.group_size = group_size
        self.ignore = list(ignore)
        self.targets = list(targets)
        self.pack_factor = 32 // bits

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        targets: Iterable[str] = ("Linear",),
        trust_remote_code: bool = False,
    ) -> "HummingConverter":
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        humming_config = getattr(config, "quantization_config", None)
        if humming_config is None:
            raise ValueError("Model config does not contain quantization_config")

        humming_config = cast(dict[str, Any], humming_config)
        if humming_config.get("quant_method") != "humming":
            raise ValueError(
                "Model config quant_method is not 'humming', got "
                f"{humming_config.get('quant_method')!r}"
            )

        return cls.from_humming_config(humming_config, targets=targets)

    @classmethod
    def from_humming_config(
        cls,
        humming_config: dict[str, Any],
        targets: Iterable[str] = ("Linear",),
    ) -> "HummingConverter":
        b_dtype = humming_config.get("b_dtype")
        if b_dtype not in _HUMMING_DTYPES:
            raise ValueError(f"Unsupported humming b_dtype: {b_dtype!r}")
        bits, _ = _HUMMING_DTYPES[b_dtype]

        if humming_config.get("weight_scale_type", "group") != "group":
            raise ValueError(
                "HummingConverter only supports weight_scale_type='group', got "
                f"{humming_config.get('weight_scale_type')!r}"
            )

        if humming_config.get("has_zero_point", False):
            raise ValueError(
                "HummingConverter only supports symmetric quantization "
                "(has_zero_point=false)"
            )

        return cls(
            bits=bits,
            group_size=int(humming_config["weight_scale_group_size"]),
            ignore=humming_config.get("ignore") or ["lm_head"],
            targets=targets,
        )

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for name in list(tensors):
            if not name.endswith(".weight"):
                continue

            module_name = name.removesuffix(".weight")
            if not self._is_targeted(module_name):
                continue

            packed = tensors[name]
            if packed.dtype != torch.int32:
                # Non-quantized weight that happens to live on a targeted module
                # (e.g., a bias-only edge case). Leave it alone.
                continue

            scale = tensors.get(f"{module_name}.weight_scale")
            if scale is None:
                raise ValueError(
                    f"Found packed weight {name} without corresponding weight_scale"
                )

            # Recover the unpacked shape. GSQ stores no padding because group_size
            # (e.g. 128) is a multiple of pack_factor (e.g. 16 for 2-bit), so the
            # last dim is exactly `packed_last * pack_factor`. Works for 2D linears
            # and 3D MoE-fused expert weights alike.
            unpacked_shape = list(packed.shape[:-1]) + [
                packed.shape[-1] * self.pack_factor
            ]

            tensors.pop(name)
            tensors[f"{module_name}.weight_packed"] = packed.contiguous()
            tensors[f"{module_name}.weight_shape"] = torch.tensor(
                unpacked_shape, dtype=torch.int64
            )

        return tensors

    def validate(self, tensors: dict[str, torch.Tensor]):
        for name, tensor in tensors.items():
            if not name.endswith(".weight"):
                continue
            module_name = name.removesuffix(".weight")
            if not self._is_targeted(module_name):
                continue
            if tensor.dtype != torch.int32:
                continue

            scale_name = f"{module_name}.weight_scale"
            if scale_name not in tensors:
                raise ValueError(
                    f"Found packed weight {name} without corresponding {scale_name}"
                )

    def create_config(self) -> QuantizationConfig:
        weights = QuantizationArgs(
            num_bits=self.bits,
            type=QuantizationType.INT,
            symmetric=True,
            group_size=self.group_size,
            strategy=QuantizationStrategy.GROUP,
        )
        return QuantizationConfig(
            config_groups={
                "config_group_0": QuantizationScheme(
                    targets=self.targets,
                    weights=weights,
                    format=CompressionFormat.pack_quantized.value,
                )
            },
            ignore=self.ignore,
            format=CompressionFormat.pack_quantized.value,
            quantization_status=QuantizationStatus.COMPRESSED.value,
        )

    def get_dependencies(self, weight_name: str) -> set[str]:
        module_name, _, suffix = weight_name.rpartition(".")
        if suffix == "weight" and self._is_targeted(module_name):
            return {f"{module_name}.weight_scale"}
        return set()

    def _is_targeted(self, module_name: str) -> bool:
        if any(match_name(module_name, ignore) for ignore in self.ignore):
            return False
        if len(self.targets) == 0 or "Linear" in self.targets:
            return True
        return any(match_name(module_name, target) for target in self.targets)
