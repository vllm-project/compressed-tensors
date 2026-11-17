# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import (
    COMPRESSIBLE_MODULE_TYPES,
    BaseCompressor,
)
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    GlobalScaleType,
    QuantizationArgs,
    QuantizationScheme,
    QuantizationType,
    ScaleFormat,
)
from compressed_tensors.quantization.utils.ue5m3 import (
    ue5m3_decode,
    ue5m3_encode,
)
from compressed_tensors.utils import getattr_chain


__all__ = [
    "NVFP4E5M3PackedCompressor",
    "NVFP4E5M3GSPackedCompressor",
]


def _can_compress_e5m3(
    module_type: type,
    scheme: QuantizationScheme,
    global_scale: GlobalScaleType,
) -> bool:
    weights = scheme.weights
    return (
        issubclass(module_type, COMPRESSIBLE_MODULE_TYPES)
        and weights is not None
        and weights.num_bits == 4
        and weights.type == QuantizationType.FLOAT.value
        and weights.group_size == 16
        and weights.scale_format == ScaleFormat.UE5M3.value
        and weights.global_scale == global_scale.value
    )


@BaseCompressor.register(name=CompressionFormat.nvfp4_e5m3_pack_quantized.value)
class NVFP4E5M3PackedCompressor(NVFP4PackedCompressor):
    """
    Compressor for NVFP4+ without per-tensor global scales.
    """

    @classmethod
    def compression_param_names(cls, scheme: QuantizationScheme) -> tuple[str]:
        param_names = ("weight_packed", "weight_scale")
        if not getattr_chain(scheme, "weights.symmetric", True):
            param_names += ("weight_zero_point",)
        return param_names

    @classmethod
    def _compress_scale(
        cls, scale: torch.Tensor, weights: QuantizationArgs
    ) -> torch.Tensor:
        return ue5m3_encode(scale)

    @classmethod
    def _decompress_scale(cls, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return ue5m3_decode(scale, dtype=dtype)

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        return _can_compress_e5m3(
            module_type,
            scheme,
            GlobalScaleType.NONE,
        )


@BaseCompressor.register(name=CompressionFormat.nvfp4_e5m3_gs_pack_quantized.value)
class NVFP4E5M3GSPackedCompressor(NVFP4E5M3PackedCompressor):
    """
    Compressor for NVFP4+ with per-tensor FP32 global scales.
    """

    @classmethod
    def compression_param_names(cls, scheme: QuantizationScheme) -> tuple[str]:
        param_names = ("weight_packed", "weight_scale", "weight_global_scale")
        if not getattr_chain(scheme, "weights.symmetric", True):
            param_names += ("weight_zero_point",)
        param_names += ("input_global_scale",)
        return param_names

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        return _can_compress_e5m3(
            module_type,
            scheme,
            GlobalScaleType.PER_TENSOR,
        )
