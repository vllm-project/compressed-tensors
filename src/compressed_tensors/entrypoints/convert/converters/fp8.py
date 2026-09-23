# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
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
from compressed_tensors.utils.match import match_name, match_quantizable_tensors
from transformers import AutoConfig


__all__ = ["FP8Converter"]


class FP8Converter(Converter):
    """
    Convert a checkpoint quantized with the transformers "fp8" quant_method
    (finegrained, block-wise FP8) into the compressed-tensors format.

    The transformers "fp8" method block-quantizes weights to float8_e4m3fn and
    stores the per-block scales under the name ``weight_scale_inv``. Despite the
    ``_inv`` suffix these scales are *not* inverted relative to the
    compressed-tensors convention: dequantization is ``weight * weight_scale`` in
    both frameworks. Converting therefore mostly amounts to renaming
    ``weight_scale_inv`` to ``weight_scale``; the fp8 weights themselves are kept
    as-is.

    See the reference implementation:
    https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/finegrained_fp8.py
    """

    def __init__(
        self,
        weight_block_size: tuple[int, int] = (128, 128),
        ignore: Iterable[str] = ("lm_head",),
        targets: Iterable[str] = ("Linear",),
    ):
        self.weight_block_size = tuple(weight_block_size)
        self.ignore = list(ignore)
        self.targets = list(targets)

        self.param_names = ["weight_scale_inv"]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        targets: Iterable[str] = ("Linear",),
        trust_remote_code: bool = False,
    ) -> "FP8Converter":
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        fp8_config = getattr(config, "quantization_config", None)
        if fp8_config is None:
            raise ValueError("Model config does not contain quantization_config")

        fp8_config = cast(dict[str, Any], fp8_config)
        if fp8_config.get("quant_method") != "fp8":
            raise ValueError("Model config is not an fp8 config")

        return cls.from_fp8_config(fp8_config, targets=targets)

    @classmethod
    def from_fp8_config(
        cls,
        fp8_config: dict[str, Any],
        targets: Iterable[str] = ("Linear",),
    ) -> "FP8Converter":
        ignore = ["lm_head"]
        for module in fp8_config.get("modules_to_not_convert") or []:
            ignore.append(f"re:.*{re.escape(module)}.*")

        return cls(
            weight_block_size=tuple(fp8_config.get("weight_block_size", (128, 128))),
            ignore=ignore,
            targets=targets,
        )

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Rename ``weight_scale_inv`` to ``weight_scale`` for every targeted module.
        The fp8 weights are left untouched.
        """
        for module_name, name in match_quantizable_tensors(
            tensors, self.ignore, self.targets, param_targets=self.param_names
        ):
            param_name = name.rpartition(".")[-1]

            if param_name == "weight_scale_inv":
                tensors[f"{module_name}.weight_scale"] = tensors[name]
                del tensors[name]

        return tensors

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Rename, then assert no leftover ``weight_scale_inv`` remains on a
        non-ignored module. A residual ``weight_scale_inv`` means the configured
        targets missed a block-quantized weight. Returns the processed tensors so
        chained converters observe the resulting format.
        """
        tensors = self.process(tensors)

        residual = [
            name
            for name in tensors
            if name.rpartition(".")[-1] == "weight_scale_inv"
            and not any(match_name(name.rpartition(".")[0], ign) for ign in self.ignore)
        ]
        if residual:
            raise ValueError(
                f"Found {len(residual)} residual weight_scale_inv after "
                f"conversion, indicating untargeted or orphan scales: {residual}"
            )

        return tensors

    def _build_quant_config(self) -> QuantizationConfig:
        weights = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.BLOCK,
            symmetric=True,
            dynamic=False,
            block_structure=list(self.weight_block_size),
        )
        input_activations = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.GROUP,
            symmetric=True,
            dynamic=True,
            group_size=self.weight_block_size[-1],
        )
        return QuantizationConfig(
            config_groups={
                "config_group_0": QuantizationScheme(
                    targets=self.targets,
                    weights=weights,
                    input_activations=input_activations,
                    format=CompressionFormat.float_quantized.value,
                )
            },
            ignore=self.ignore,
            format=CompressionFormat.float_quantized.value,
            quantization_status=QuantizationStatus.COMPRESSED.value,
        )

    def update_config(
        self, config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        quant_config = self._build_quant_config()
        if config is not None:
            config.merge(quant_config)
            return config
        return quant_config

    def get_dependencies(self, weight_name: str) -> set[str]:
        # renaming weight_scale_inv is independent of any other tensor
        return set()
