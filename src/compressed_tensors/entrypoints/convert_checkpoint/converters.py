# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Iterable, Protocol

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert_checkpoint.helpers import (
    match_quantizable_tensors,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from compressed_tensors.quantization.quant_scheme import NVFP4


class Converter(Protocol):
    """
    Converter interface, to modify safetensors files based
    on tensor name and pointer to torch.Tensor
    """

    def process(self, tensors: dict[str, torch.Tensor]):
        pass

    def validate(self, tensors: dict[str, torch.Tensor]):
        pass

    def create_config(self) -> QuantizationConfig:
        pass


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
        allowed_names = ["input_scale", "weight", "weight_scale", "weight_scale_2"]
        if self.kv_cache_scheme is not None:
            allowed_names += ["k_scale", "v_scale"]

        for _, name in match_quantizable_tensors(
            tensors, self.ignore, self.targets, allow_nonquantizable=True
        ):
            param_name = name.rsplit(".", 1)[-1]

            if param_name not in allowed_names:
                raise RuntimeError(f"Hit unexpected tensor {name}")

    def create_config(self) -> QuantizationConfig:
        return QuantizationConfig(
            config_groups={
                "group_0": QuantizationScheme(
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


# TODO implement
class AutoAWQConverter(Converter):
    """
    Convert params from AutoAWQ W4A16 to CT W4A16 convention
    """

    pass
