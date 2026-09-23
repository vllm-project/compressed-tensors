# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
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
from compressed_tensors.utils.match import match_name
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_safetensors_header,
)
from loguru import logger
from transformers import AutoConfig


__all__ = ["FP8Converter"]

# safetensors dtype strings (as found in the file header) for quantized weights
_FP8_WEIGHT_DTYPES = {"F8_E4M3", "F8_E5M2"}
_FP4_WEIGHT_DTYPES = {"I8", "U8"}  # two packed e2m1 values per byte
# scale param names that different fp8 checkpoints use for the per-block weight scale
_SCALE_PARAM_NAMES = ("weight_scale_inv", "scale")


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
        fp8_targets, fp4_targets = cls._classify_targets(model_name_or_path, ignore)
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

    @staticmethod
    def _classify_targets(
        model_name_or_path: str, ignore: Iterable[str]
    ) -> tuple[list[str], list[str]]:
        """
        Read the safetensors headers and bucket every quantized module (a
        ``.weight`` that has an accompanying scale) into fp8 vs fp4 targets based
        on the stored weight dtype. Module names are generalized into regex
        targets by replacing numeric indices (layer/expert ids) with ``\\d+`` so
        the resulting config stays compact.

        Only tensor headers are inspected, never the tensor data, so this stays
        cheap even for multi-hundred-GB checkpoints.
        """
        weight_dtypes, all_names = FP8Converter._read_weight_dtypes(model_name_or_path)

        fp8_patterns: set[str] = set()
        fp4_patterns: set[str] = set()
        for weight_name, dtype in weight_dtypes.items():
            module_name = weight_name[: -len(".weight")]
            if not any(f"{module_name}.{s}" in all_names for s in _SCALE_PARAM_NAMES):
                continue  # unquantized weight (e.g. norms), skip
            if any(match_name(module_name, ign) for ign in ignore):
                continue
            if dtype in _FP8_WEIGHT_DTYPES:
                fp8_patterns.add(FP8Converter._generalize_name(module_name))
            elif dtype in _FP4_WEIGHT_DTYPES:
                fp4_patterns.add(FP8Converter._generalize_name(module_name))

        return sorted(fp8_patterns), sorted(fp4_patterns)

    @staticmethod
    def _read_weight_dtypes(
        model_name_or_path: str,
    ) -> tuple[dict[str, str], set[str]]:
        """
        Return ``(weight_name -> safetensors dtype string, set of all tensor
        names)`` by reading only safetensors headers. Local directories are read
        file-by-file; Hub stubs use the metadata API to avoid downloading weights.
        """
        weight_dtypes: dict[str, str] = {}
        all_names: set[str] = set()

        if os.path.exists(model_name_or_path):
            for rel_path, path in get_checkpoint_files(model_name_or_path).items():
                if not rel_path.endswith(".safetensors"):
                    continue
                for name, info in get_safetensors_header(path).items():
                    if name == "__metadata__":
                        continue
                    all_names.add(name)
                    if name.endswith(".weight"):
                        weight_dtypes[name] = info["dtype"]
        else:
            from huggingface_hub import HfApi

            metadata = HfApi().get_safetensors_metadata(model_name_or_path)
            for file_metadata in metadata.files_metadata.values():
                for name, tensor in file_metadata.tensors.items():
                    all_names.add(name)
                    if name.endswith(".weight"):
                        weight_dtypes[name] = tensor.dtype

        return weight_dtypes, all_names

    @staticmethod
    def _generalize_name(module_name: str) -> str:
        """Turn a concrete module name into a regex target, replacing numeric
        indices with ``\\d+``, e.g. ``model.layers.3.experts.7.w1`` ->
        ``re:.*layers\\.\\d+\\.experts\\.\\d+\\.w\\d+$``.
        """
        pattern = re.sub(r"\d+", lambda _m: r"\d+", module_name).replace(".", r"\.")
        return f"re:.*{pattern}$"

    def _is_fp8(self, module_name: str) -> bool:
        return self._is_targeted(module_name, self.fp8_targets)

    def _is_fp4(self, module_name: str) -> bool:
        return self._is_targeted(module_name, self.fp4_targets)

    def _is_targeted(self, module_name: str, targets: Iterable[str]) -> bool:
        if any(match_name(module_name, ign) for ign in self.ignore):
            return False
        targets = list(targets)
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

            if param_name in _SCALE_PARAM_NAMES:
                if self._is_fp4(module_name):
                    tensors[f"{module_name}.weight_scale"] = self._fp4_scale(
                        tensors.pop(name)
                    )
                elif self._is_fp8(module_name):
                    tensors[f"{module_name}.weight_scale"] = self._fp8_scale(
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

    @staticmethod
    def _fp8_scale(scale: torch.Tensor) -> torch.Tensor:
        """
        The ``float-quantized`` block compressor multiplies the weight by the
        scale directly and has no ``float8_e8m0fnu`` kernel, so UE8M0 scales must
        be promoted to fp32 (a lossless power-of-2 cast). fp32 scales (V3-style)
        pass through unchanged.
        """
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if e8m0 is not None and scale.dtype == e8m0:
            return scale.to(torch.float32)
        return scale

    @staticmethod
    def _fp4_scale(scale: torch.Tensor) -> torch.Tensor:
        """
        Compressed-tensors' MXFP4 compressor stores weight_scale as the raw E8M0
        biased-exponent byte (uint8) and decodes it as ``2 ** (byte - 127)``. The
        checkpoint's ``float8_e8m0fnu`` scale already holds exactly that byte in
        its bit pattern, so reinterpret rather than value-cast (which would build
        the wrong integer from the float value).
        """
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if e8m0 is not None and scale.dtype == e8m0:
            return scale.view(torch.uint8)
        if scale.dtype == torch.uint8:
            return scale
        # float32 scales (V3-style): encode to E8M0 biased exponent
        return (torch.log2(scale).floor().to(torch.int32) + 127).clamp(0, 255).to(
            torch.uint8
        )

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Process, then assert no residual scale param remains on a targeted module.
        A leftover ``scale``/``weight_scale_inv`` means the configured targets
        missed a quantized weight. Returns the processed tensors so chained
        converters observe the resulting format.
        """
        tensors = self.process(tensors)

        residual = [
            name
            for name in tensors
            if name.rpartition(".")[-1] in _SCALE_PARAM_NAMES
            and (
                self._is_fp8(name.rpartition(".")[0])
                or self._is_fp4(name.rpartition(".")[0])
            )
        ]
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
