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
from compressed_tensors.quantization.quant_scheme import NVFP4
from compressed_tensors.utils.match import match_name, match_quantizable_tensors


class ModelOptNvfp4Converter(Converter):
    """
    Convert params from modelopt NVFP4 to CT NVFP4 convention,
    and optionally the kv_cache_scheme
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
        Map the modelopt NVFP4 tensors to the appropriate compressed-tensors
        NVFP4 format.
        Some tensors require rename, some require inversion
        - 1 / input_scale -> input_global_scale
        - weight -> weight_packed
        - 1 / weight_scale_2 -> weight_global_scale
        """
        for module_name, name in match_quantizable_tensors(
            tensors, self.ignore, self.targets, allow_nonquantizable=True
        ):
            param_name = name.rsplit(".", 1)[-1]

            match param_name:
                # input_scale -> input_global_scale F32
                case "input_scale":
                    # convert modelopt input_scale x -> 1/x
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1070-L1073
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1134
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L190
                    tensors[f"{module_name}.input_global_scale"] = 1 / tensors[name]
                    del tensors[name]
                # weight -> weight_packed U8
                case "weight":
                    tensors[f"{module_name}.weight_packed"] = tensors[name]
                    del tensors[name]
                # weight_scale -> weight_scale F8_E4M3
                case "weight_scale":
                    pass
                # weight_scale_2 -> weight_global_scale F32
                case "weight_scale_2":
                    # convert modelopt weight_scale_2 x -> 1/x
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1066-L1068
                    # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L163-L166
                    tensors[f"{module_name}.weight_global_scale"] = 1 / tensors[name]
                    del tensors[name]
                case "k_scale" | "v_scale":
                    # convert kv cache scales to appropriate dtype
                    # often F32 in modelopt, defaults BF16 in compressed-tensors
                    tensors[name] = tensors[name].to(
                        self.kv_cache_scheme.scale_dtype or torch.bfloat16
                    )

    def validate(self, tensors: dict[str, torch.Tensor]):
        """
        Ensure all tensor names of targeted layers are expected and no
        untargeted layers have unexpected tensor names
        """
        allowed_names = ["input_scale", "weight", "weight_scale", "weight_scale_2"]
        if self.kv_cache_scheme is not None:
            allowed_names += ["k_scale", "v_scale"]

        targeted_names = [
            name
            for _, name in match_quantizable_tensors(
                tensors, self.ignore, self.targets, allow_nonquantizable=True
            )
        ]
        for name in targeted_names:
            param_name = name.rsplit(".", 1)[-1]

            if param_name not in allowed_names:
                raise ValueError(f"Hit unexpected targeted tensor {name}")

        disallowed_names = ["input_scale", "weight_scale", "weight_scale_2"]
        untargeted_names = [
            name for name in tensors.keys() if name not in targeted_names
        ]
        for name in untargeted_names:
            param_name = name.rsplit(".", 1)[-1]

            if param_name in disallowed_names:
                raise ValueError(f"Hit unexpected non-targeted tensor {name}")

    def get_dependencies(self, weight_name: str) -> dict[str, bool]:
        module_name, suffix = weight_name.rsplit(".", 1)
        if (
            any([match_name(module_name, target) for target in self.targets])
            and not any([match_name(module_name, ignore) for ignore in self.ignore])
            and suffix == "weight"
        ):
            deps = set(
                f"{module_name}.input_scale",
                f"{module_name}.weight_scale",
                f"{module_name}.weight_scale_2",
            )

            if self.kv_cache_scheme:
                if module_name.endswith("k_proj"):
                    deps |= {f"{module_name}.k_scale"}
                if module_name.endswith("v_proj"):
                    deps |= {f"{module_name}.v_scale"}

            return deps

        return {}

    def create_config(self) -> QuantizationConfig:
        return QuantizationConfig(
            config_groups={
                "config_group_0": QuantizationScheme(
                    **NVFP4,
                    targets=self.targets,
                    format=CompressionFormat.nvfp4_pack_quantized.value,
                )
            },
            ignore=self.ignore,
            kv_cache_scheme=self.kv_cache_scheme,
            format=CompressionFormat.nvfp4_pack_quantized.value,
            quantization_status=QuantizationStatus.COMPRESSED.value,
        )
