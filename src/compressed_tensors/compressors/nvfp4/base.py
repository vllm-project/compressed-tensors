# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.nvfp4.helpers import (
    pack_fp4_to_uint8,
    unpack_fp4_from_uint8,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.utils import TensorStateDict, getattr_chain


__all__ = ["NVFP4PackedCompressor"]


@BaseCompressor.register(name=CompressionFormat.nvfp4_pack_quantized.value)
class NVFP4PackedCompressor(BaseCompressor):
    """
    Compressor for FP4 quantized models.

    Weights of each quantized layer are packed into uint8. Only supports
    symmetric weight compression.
    """

    @classmethod
    def compression_param_names(cls, scheme: QuantizationScheme) -> tuple[str]:
        param_names = (
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
        )
        if not getattr_chain(scheme, "weights.symmetric", True):
            param_names += ("weight_zero_point",)
        return param_names

    @classmethod
    def _compress_scale(
        cls, scale: torch.Tensor, weights: QuantizationArgs
    ) -> torch.Tensor:
        scale_dtype = weights.scale_dtype or torch.float8_e4m3fn
        return scale.to(scale_dtype)

    @classmethod
    def _decompress_scale(cls, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return scale.to(dtype)

    @classmethod
    def compress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Compress a per-module state dict.

        Quantizes the weight and packs into uint8 as ``weight_packed``.
        Compresses the scale according to ``scheme.weights.scale_dtype``.
        Removes the raw ``weight``.

        :param state_dict: local-name state dict (weight, weight_scale, …)
        :param scheme: quantization scheme for the weight
        :return: compressed state dict
        """
        state_dict = state_dict.copy()
        weight = state_dict.pop("weight")
        scale = state_dict.pop("weight_scale")
        global_scale = state_dict.get("weight_global_scale", None)
        zero_point = state_dict.get("weight_zero_point", None)
        weights = scheme.weights

        quantized_weight = quantize(
            x=weight,
            scale=scale,
            global_scale=global_scale,
            zero_point=zero_point,
            args=weights,
        )
        state_dict["weight_packed"] = pack_fp4_to_uint8(quantized_weight)
        state_dict["weight_scale"] = cls._compress_scale(scale, weights)
        state_dict = cls._remove_symmetric_zp(state_dict, scheme)

        return state_dict

    @classmethod
    def decompress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Decompress a per-module state dict.

        Unpacks ``weight_packed`` back to fp4 values and dequantizes.
        Converts ``weight_scale`` back to float for dequantization.

        :param state_dict: local-name state dict (weight_packed, weight_scale, …)
        :param scheme: quantization scheme for the weight
        :return: decompressed state dict with weight in float dtype
        """
        state_dict = state_dict.copy()
        packed = state_dict.pop("weight_packed")
        scale = state_dict.get("weight_scale")
        global_scale = state_dict.get("weight_global_scale", None)

        m, n = packed.shape
        unpacked = unpack_fp4_from_uint8(packed, m, n * 2)

        scale_float = cls._decompress_scale(scale, unpacked.dtype)

        state_dict["weight"] = dequantize(
            x_q=unpacked,
            scale=scale_float,
            global_scale=global_scale,
            dtype=unpacked.dtype,
        )
        state_dict["weight_scale"] = torch.nn.Parameter(
            scale_float, requires_grad=False
        )

        return state_dict

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """NVFP4 matches FP4 with group_size != 32 (or None)."""
        return (
            module_type == torch.nn.Linear
            and scheme.weights is not None
            and scheme.weights.num_bits == 4
            and scheme.weights.type == QuantizationType.FLOAT.value
            and scheme.weights.group_size == 16
        )
