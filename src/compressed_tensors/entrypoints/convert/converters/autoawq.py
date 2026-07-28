# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from collections.abc import Iterable
from typing import Any, cast

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
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


__all__ = ["AutoAWQConverter"]


class AutoAWQConverter(Converter):
    """
    Convert AutoAWQ GEMM checkpoint tensors to compressed-tensors WNA16 tensors.

    AutoAWQ GEMM stores quantized linear layers as qweight/qzeros/scales. This
    converter unpacks qweight into compressed-tensors' signed integer convention
    and preserves the per-group quantization parameters as weight_scale and
    weight_zero_point.
    """

    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        version: str = "gemm",
        ignore: Iterable[str] = ("lm_head",),
        targets: Iterable[str] = ("Linear",),
    ):
        if bits != 4:
            raise ValueError("AutoAWQConverter currently supports only 4-bit weights")
        if version != "gemm":
            raise ValueError(f"Unsupported AutoAWQ version: {version}")

        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.ignore = list(ignore)
        self.targets = list(targets)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        targets: Iterable[str] = ("Linear",),
        trust_remote_code: bool = False,
    ) -> "AutoAWQConverter":

        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        autoawq_config = getattr(config, "quantization_config", None)
        if autoawq_config is None:
            raise ValueError("Model config does not contain quantization_config")

        autoawq_config = cast(dict[str, Any], autoawq_config)
        if autoawq_config.get("quant_method") != "awq":
            raise ValueError("Model config is not an AutoAWQ config")

        return cls.from_autoawq_config(
            autoawq_config,
            targets=targets,
        )

    @classmethod
    def from_autoawq_config(
        cls,
        autoawq_config: dict[str, Any],
        targets: Iterable[str] = ("Linear",),
    ) -> "AutoAWQConverter":
        ignore = ["lm_head"]
        for module in autoawq_config.get("modules_to_not_convert") or []:
            ignore.append(f"re:.*{re.escape(module)}.*")

        return cls(
            bits=autoawq_config.get("bits", 4),
            group_size=autoawq_config.get("group_size", 128),
            zero_point=autoawq_config.get("zero_point", True),
            version=autoawq_config.get("version", "gemm"),
            ignore=ignore,
            targets=targets,
        )

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for name in list(tensors):
            if not name.endswith(".qweight"):
                continue

            module_name = name.removesuffix(".qweight")
            if not self._is_targeted(module_name):
                continue

            qweight = tensors.pop(f"{module_name}.qweight")
            qzeros = tensors.pop(f"{module_name}.qzeros", None)
            scales = tensors.pop(f"{module_name}.scales")
            weight, weight_scale, weight_zero_point = self._convert_gemm_module(
                qweight, scales, qzeros
            )

            tensors[f"{module_name}.weight_scale"] = weight_scale
            tensors[f"{module_name}.weight_packed"] = pack_to_int32(weight, self.bits)
            tensors[f"{module_name}.weight_shape"] = torch.tensor(weight.shape)

            if weight_zero_point is not None:
                weight_zero_point = pack_to_int32(
                    weight_zero_point, self.bits, packed_dim=0
                ).contiguous()
                tensors[f"{module_name}.weight_zero_point"] = weight_zero_point
        return tensors

    def validate(self, tensors: dict[str, torch.Tensor]):
        for name in tensors:
            module_name, _, param_name = name.rpartition(".")

            if param_name in {"qweight", "qzeros", "scales"}:
                if not self._is_targeted(module_name):
                    raise ValueError(f"Found unexpected non-targeted tensor {name}")

            if param_name != "qweight" or not self._is_targeted(module_name):
                continue

            for dependency in self.get_dependencies(name):
                if dependency not in tensors:
                    raise ValueError(
                        f"Found qweight without corresponding {dependency}"
                    )

    def create_config(self) -> QuantizationConfig:
        weights = QuantizationArgs(
            num_bits=self.bits,
            type=QuantizationType.INT,
            symmetric=not self.zero_point,
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
        if suffix == "qweight" and self._is_targeted(module_name):
            dependencies = {f"{module_name}.scales"}
            if self.zero_point:
                dependencies.add(f"{module_name}.qzeros")

            return dependencies

        return set()

    def _convert_gemm_module(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.zero_point and qzeros is None:
            raise ValueError("Found qweight without corresponding qzeros")

        iweight, izeros = self.unpack_awq(qweight, qzeros, self.bits)
        iweight, izeros = self.reverse_awq_order(iweight, izeros, self.bits)

        iweight = torch.bitwise_and(iweight, (2**self.bits) - 1)

        quantized_weight = iweight - 2 ** (self.bits - 1)

        weight_zero_point = None
        if self.zero_point:
            assert izeros is not None
            weight_zero_point = torch.bitwise_and(izeros, (2**self.bits) - 1)
            weight_zero_point = weight_zero_point - 2 ** (self.bits - 1)
            weight_zero_point = weight_zero_point.T.contiguous()

        return (
            quantized_weight.T.contiguous(),
            scales.T.contiguous(),
            weight_zero_point,
        )

    def _is_targeted(self, module_name: str) -> bool:
        if any(match_name(module_name, ignore) for ignore in self.ignore):
            return False
        if len(self.targets) == 0 or "Linear" in self.targets:
            return True

        return any(match_name(module_name, target) for target in self.targets)

    @staticmethod
    def unpack_awq(
        qweight: torch.Tensor, qzeros: torch.Tensor | None, bits: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Unpack AutoAWQ GEMM int32-packed weights and zero-points into int8 values.
        """
        shifts = torch.arange(0, 32, bits, device=qweight.device)

        iweights = torch.bitwise_right_shift(
            qweight[:, :, None], shifts[None, None, :]
        ).to(torch.int8)
        iweights = iweights.view(iweights.shape[0], -1)

        if qzeros is None:
            return iweights, None

        izeros = torch.bitwise_right_shift(
            qzeros[:, :, None], shifts[None, None, :]
        ).to(torch.int8)
        izeros = izeros.view(izeros.shape[0], -1)

        return iweights, izeros

    @staticmethod
    def reverse_awq_order(
        iweights: torch.Tensor, izeros: torch.Tensor | None, bits: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Undo AutoAWQ's special intra-int32 packing order.
        """
        reverse_order_tensor = torch.arange(
            iweights.shape[-1],
            dtype=torch.int32,
            device=iweights.device,
        )
        reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
        reverse_order_tensor = reverse_order_tensor[
            :, AutoAWQConverter.AWQ_REVERSE_ORDER
        ]
        reverse_order_tensor = reverse_order_tensor.view(-1)

        iweights = iweights[:, reverse_order_tensor]
        if izeros is not None:
            izeros = izeros[:, reverse_order_tensor]

        return iweights, izeros
