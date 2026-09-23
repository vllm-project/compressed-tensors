# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from collections.abc import Iterable
from typing import Any, cast

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.entrypoints.convert.converters.fp8_helpers import (
    SCALE_PARAM_NAMES,
    classify_targets,
    encode_fp4_scale,
    encode_fp8_scale,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.utils.match import match_name
from loguru import logger
from transformers import AutoConfig


__all__ = ["FP8Converter"]


class FP8Converter(Converter):
    """
    Convert a checkpoint quantized with the transformers "fp8" quant_method
    (finegrained, block-wise) into the compressed-tensors format.

    Despite the name, the transformers "fp8" method is *mixed precision*: linear
    weights are block-quantized to ``float8_e4m3fn`` while MoE expert weights are
    frequently packed as FP4 (two ``e2m1`` values per ``int8`` byte, group-wise
    scaled). Both share the ``quant_method: "fp8"`` config and store their
    per-block/-group scale next to the weight (named ``weight_scale_inv`` in
    DeepSeek-V3-style checkpoints, or ``scale`` in newer ones). Despite the
    ``_inv`` suffix these scales are *not* inverted relative to the
    compressed-tensors convention (dequant is ``weight * weight_scale`` in both).

    This converter therefore emits up to two config groups so each precision is
    described correctly:
    - FP8 modules -> ``float-quantized`` block scheme
    - FP4 modules -> ``mxfp4-pack-quantized`` group scheme

    and renames tensors to the compressed-tensors convention:
    - fp8:  ``<mod>.scale``/``<mod>.weight_scale_inv`` -> ``<mod>.weight_scale``
    - fp4:  ``<mod>.weight`` (int8) -> ``<mod>.weight_packed`` (uint8) and the
      scale -> ``<mod>.weight_scale``

    See the reference implementation:
    https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/finegrained_fp8.py
    """

    def __init__(
        self,
        fp8_targets: Iterable[str] = ("Linear",),
        fp4_targets: Iterable[str] = tuple(),
        ignore: Iterable[str] = ("lm_head",),
        weight_block_size: tuple[int, int] = (128, 128),
        fp4_group_size: int = 32,
    ):
        self.fp8_targets = list(fp8_targets)
        self.fp4_targets = list(fp4_targets)
        self.ignore = list(ignore)
        self.weight_block_size = tuple(weight_block_size)
        self.fp4_group_size = fp4_group_size

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
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

        weight_block_size = tuple(fp8_config.get("weight_block_size") or (128, 128))

        ignore = ["lm_head"]
        for module in fp8_config.get("modules_to_not_convert") or []:
            ignore.append(f"re:.*{re.escape(module)}.*")

        # inspect the checkpoint to split modules by storage precision. "fp8"
        # checkpoints commonly mix fp8 linear weights with fp4-packed experts, so
        # we cannot assume a single scheme from config.json alone.
        fp8_targets, fp4_targets = classify_targets(model_name_or_path, ignore)
        logger.info(
            f"Detected {len(fp8_targets)} fp8 target pattern(s) and "
            f"{len(fp4_targets)} fp4 target pattern(s)"
        )

        return cls(
            fp8_targets=fp8_targets,
            fp4_targets=fp4_targets,
            ignore=ignore,
            weight_block_size=weight_block_size,
        )

    def _is_fp8(self, module_name: str) -> bool:
        return self._is_targeted(module_name, self.fp8_targets)

    def _is_fp4(self, module_name: str) -> bool:
        return self._is_targeted(module_name, self.fp4_targets)

    def _is_targeted(self, module_name: str, targets: list[str]) -> bool:
        if any(match_name(module_name, ign) for ign in self.ignore):
            return False
        if "Linear" in targets:
            return True
        return any(match_name(module_name, target) for target in targets)

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Rename fp8/fp4 tensors to the compressed-tensors convention. fp8 weights
        keep their dtype; fp4 weights are reinterpreted from int8 to uint8 and
        moved to ``weight_packed``.
        """
        for name in list(tensors):
            module_name, _, param_name = name.rpartition(".")

            if param_name in SCALE_PARAM_NAMES:
                if self._is_fp4(module_name):
                    tensors[f"{module_name}.weight_scale"] = encode_fp4_scale(
                        tensors.pop(name)
                    )
                elif self._is_fp8(module_name):
                    tensors[f"{module_name}.weight_scale"] = encode_fp8_scale(
                        tensors.pop(name)
                    )

            elif param_name == "weight" and self._is_fp4(module_name):
                # fp4 weights are two e2m1 nibbles packed per byte; compressed-
                # tensors stores these as uint8 weight_packed
                weight = tensors.pop(name)
                if weight.dtype == torch.int8:
                    weight = weight.view(torch.uint8)
                tensors[f"{module_name}.weight_packed"] = weight

        return tensors

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Process, then assert no residual scale param remains on a targeted module.
        A leftover ``scale``/``weight_scale_inv`` means the configured targets
        missed a quantized weight. Returns the processed tensors so chained
        converters observe the resulting format.
        """
        tensors = self.process(tensors)

        residual = []
        for name in tensors:
            module_name, _, param_name = name.rpartition(".")
            if param_name in SCALE_PARAM_NAMES and (
                self._is_fp8(module_name) or self._is_fp4(module_name)
            ):
                residual.append(name)
        if residual:
            raise ValueError(
                f"Found {len(residual)} residual scale param(s) after conversion, "
                f"indicating untargeted or orphan scales: {residual}"
            )

        return tensors

    def _build_quant_config(self) -> QuantizationConfig:
        config_groups: dict[str, QuantizationScheme] = {}

        if self.fp8_targets:
            config_groups["group_0_fp8"] = QuantizationScheme(
                targets=self.fp8_targets,
                weights=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.BLOCK,
                    symmetric=True,
                    dynamic=False,
                    block_structure=list(self.weight_block_size),
                ),
                input_activations=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.GROUP,
                    symmetric=True,
                    dynamic=True,
                    group_size=self.weight_block_size[-1],
                ),
                format=CompressionFormat.float_quantized.value,
            )

        if self.fp4_targets:
            config_groups["group_1_fp4"] = QuantizationScheme(
                targets=self.fp4_targets,
                weights=QuantizationArgs(
                    num_bits=4,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.GROUP,
                    symmetric=True,
                    dynamic=False,
                    group_size=self.fp4_group_size,
                    scale_dtype=torch.uint8,
                    zp_dtype=torch.uint8,
                ),
                input_activations=QuantizationArgs(
                    num_bits=4,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.GROUP,
                    symmetric=True,
                    dynamic=True,
                    group_size=self.fp4_group_size,
                    scale_dtype=torch.uint8,
                    zp_dtype=torch.uint8,
                ),
                format=CompressionFormat.mxfp4_pack_quantized.value,
            )

        if not config_groups:
            raise ValueError("No fp8 or fp4 targets configured for FP8Converter")

        # multiple compression formats -> the checkpoint is mixed-precision
        formats = {group.format for group in config_groups.values()}
        top_format = (
            formats.pop()
            if len(formats) == 1
            else CompressionFormat.mixed_precision.value
        )

        return QuantizationConfig(
            config_groups=config_groups,
            ignore=self.ignore,
            format=top_format,
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
        # every rename/repack acts on a single tensor in isolation
        return set()
