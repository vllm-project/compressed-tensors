# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.utils import (
    dequantize_lut_b,
    is_lut_b_quantization,
    quantize_lut_b,
)
from compressed_tensors.utils import TensorStateDict


__all__ = ["LutBCompressor"]


@BaseCompressor.register(name=CompressionFormat.lut_b.value)
class LutBCompressor(BaseCompressor):
    """Compressor for the canonical LUT-B checkpoint representation."""

    @classmethod
    def compression_param_names(cls, scheme: QuantizationScheme) -> tuple[str]:
        return ("weight_packed", "weight_codebook")

    @classmethod
    def compress(
        cls,
        state_dict: TensorStateDict,
        scheme: QuantizationScheme,
    ) -> TensorStateDict:
        """Compress a weight to packed 3-bit indices and E4M3 codebooks."""
        state_dict = state_dict.copy()
        weight = state_dict.pop("weight")
        codebooks = state_dict.pop("weight_codebook", None)

        packed, codebooks = quantize_lut_b(weight, codebooks)
        state_dict["weight_packed"] = packed
        state_dict["weight_codebook"] = codebooks
        state_dict = cls._remove_symmetric_zp(state_dict, scheme)
        return state_dict

    @classmethod
    def decompress(
        cls,
        state_dict: TensorStateDict,
        scheme: QuantizationScheme,
    ) -> TensorStateDict:
        """Decompress canonical LUT-B tensors to a BF16 logical weight."""
        state_dict = state_dict.copy()
        packed = state_dict.pop("weight_packed")
        codebooks = state_dict["weight_codebook"]
        state_dict["weight"] = dequantize_lut_b(
            packed,
            codebooks,
            dtype=torch.bfloat16,
        )
        return state_dict

    @classmethod
    def can_compress(
        cls,
        module_type: type,
        scheme: QuantizationScheme,
    ) -> bool:
        return (
            module_type is torch.nn.Linear
            and scheme.weights is not None
            and is_lut_b_quantization(scheme.weights)
        )
