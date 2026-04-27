# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Iterable

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from compressed_tensors.quantization.quant_scheme import FP8
from compressed_tensors.utils.match import match_name, match_quantizable_tensors


class FP8CompressedTensorsConverter(Converter):
    """
    Convert checkpoint from fp8 quant_method format to compressed-tensors format.

    The fp8 format stores weights in float8_e4m3fn with per-tensor scalar scales
    (weight_scale), and optionally static per-tensor input scales (input_scale)
    for activation quantization. Tensor names already match compressed-tensors
    conventions, so the main task is creating the QuantizationConfig.

    :param ignore: layer name patterns to exclude from conversion
    :param targets: layer name patterns to target for conversion
    :param kv_cache_scheme: optional quantization args for KV cache scales
    """

    def __init__(
        self,
        ignore: Iterable[str] = tuple(),
        targets: Iterable[str] = tuple(),
        kv_cache_scheme: QuantizationArgs | None = None,
    ):
        self.ignore = ignore
        self.targets = targets
        self.kv_cache_scheme = kv_cache_scheme

    def process(self, tensors: dict[str, torch.Tensor]):
        """
        Process tensors from fp8 format. Tensor names and values already match
        compressed-tensors conventions, so only KV cache scale dtype conversion
        is performed when a kv_cache_scheme is provided.
        """
        for module_name, name in match_quantizable_tensors(
            tensors, self.ignore, self.targets, allow_nonquantizable=True
        ):
            param_name = name.rsplit(".", 1)[-1]

            match param_name:
                case "k_scale" | "v_scale":
                    if self.kv_cache_scheme is not None:
                        tensors[name] = tensors[name].to(
                            self.kv_cache_scheme.scale_dtype or torch.bfloat16
                        )

    def validate(self, tensors: dict[str, torch.Tensor]):
        """
        Ensure targeted layers only contain expected fp8 tensor names,
        and untargeted layers do not contain fp8-specific tensors.
        """
        allowed_names = ["weight", "weight_scale", "input_scale"]
        if self.kv_cache_scheme is not None:
            allowed_names += ["k_scale", "v_scale"]

        targeted_names = [
            name
            for _, name in match_quantizable_tensors(
                tensors, self.ignore, self.targets, allow_nonquantizable=True
            )
        ]
        for name in targeted_names:
            module_name, param_name = name.rsplit(".", 1)

            if param_name not in allowed_names:
                raise ValueError(f"Hit unexpected targeted tensor {name}")

            if param_name == "weight" and f"{module_name}.weight_scale" not in tensors:
                raise ValueError(
                    f"Found weight without corresponding weight_scale {name}"
                )

        disallowed_names = ["weight_scale", "input_scale"]
        untargeted_names = [
            name for name in tensors.keys() if name not in targeted_names
        ]
        for name in untargeted_names:
            param_name = name.rsplit(".", 1)[-1]

            if param_name in disallowed_names:
                raise ValueError(f"Hit unexpected non-targeted tensor {name}")

    def get_dependencies(self, weight_name: str) -> set[str]:
        module_name, suffix = weight_name.rsplit(".", 1)
        if (
            any([match_name(module_name, target) for target in self.targets])
            and not any([match_name(module_name, ignore) for ignore in self.ignore])
            and suffix == "weight"
        ):
            deps = {f"{module_name}.weight_scale"}

            if self.kv_cache_scheme:
                if module_name.endswith("k_proj"):
                    deps.add(f"{module_name}.k_scale")
                if module_name.endswith("v_proj"):
                    deps.add(f"{module_name}.v_scale")

            return deps

        return set()

    def create_config(self, config: dict[str, object]) -> dict[str, object]:
        quant_config = QuantizationConfig(
            config_groups={
                "config_group_0": QuantizationScheme(
                    **FP8,
                    targets=self.targets,
                    format=CompressionFormat.float_quantized.value,
                )
            },
            ignore=self.ignore,
            kv_cache_scheme=self.kv_cache_scheme,
            format=CompressionFormat.float_quantized.value,
            quantization_status=QuantizationStatus.COMPRESSED.value,
        )
        config.update(quant_config.model_dump())
        return config


def merge_ct_configs(
    config_a: dict[str, object], config_b: dict[str, object]
) -> dict[str, object]:
    """
    Merge two compressed-tensors quantization config dicts into one.

    :param config_a: first quantization config dict
    :param config_b: second quantization config dict
    :return: merged config dict
    """
    merged = {}

    # config_groups: concatenate with unique keys
    groups_a = config_a.get("config_groups", {})
    groups_b = config_b.get("config_groups", {})
    overlapping = set(groups_a) & set(groups_b)
    if overlapping:
        raise ValueError(
            f"config_groups share overlapping keys: {overlapping}"
        )
    merged["config_groups"] = {**groups_a, **groups_b}

    # ignore: concatenate
    ignore_a = config_a.get("ignore") or []
    ignore_b = config_b.get("ignore") or []
    merged["ignore"] = ignore_a + ignore_b

    # kv_cache_scheme: error if both provided and non-identical
    kv_a = config_a.get("kv_cache_scheme")
    kv_b = config_b.get("kv_cache_scheme")
    if kv_a is not None and kv_b is not None and kv_a != kv_b:
        raise ValueError(
            "Cannot merge configs with different kv_cache_scheme values: "
            f"{kv_a} != {kv_b}"
        )
    merged["kv_cache_scheme"] = kv_a if kv_a is not None else kv_b

    # format: mixed_precision if both provided and non-identical
    format_a = config_a.get("format")
    format_b = config_b.get("format")
    if format_a is not None and format_b is not None and format_a != format_b:
        merged["format"] = CompressionFormat.mixed_precision.value
    else:
        merged["format"] = format_a if format_a is not None else format_b

    # quantization_status: error if both provided and non-identical
    status_a = config_a.get("quantization_status")
    status_b = config_b.get("quantization_status")
    if status_a is not None and status_b is not None and status_a != status_b:
        raise ValueError(
            "Cannot merge configs with different quantization_status values: "
            f"{status_a} != {status_b}"
        )
    merged["quantization_status"] = status_a if status_a is not None else status_b

    # quant_method: preserve if consistent
    method_a = config_a.get("quant_method")
    method_b = config_b.get("quant_method")
    merged["quant_method"] = method_a if method_a is not None else method_b

    return merged
