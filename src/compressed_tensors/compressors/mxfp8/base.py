# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.mx_utils import (
    compress_mx_scale,
    decompress_mx_scale,
)
from compressed_tensors.compressors.naive_quantized.base import (
    NaiveQuantizationCompressor,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationType,
)
from compressed_tensors.utils import TensorStateDict


__all__ = ["MXFP8QuantizationCompressor"]


@BaseCompressor.register(name=CompressionFormat.mxfp8_quantized.value)
class MXFP8QuantizationCompressor(NaiveQuantizationCompressor):
    """
    Compressor for MXFP8 quantized models.

    Weights are stored as float8_e4m3fn with E8M0 (power-of-2) scales
    stored as uint8 exponents. Extends NaiveQuantizationCompressor by
    converting float scales to/from E8M0 exponent format during
    compression/decompression.
    """

    @classmethod
    def _compress_scale(
        cls, scale: torch.Tensor, weights: QuantizationArgs
    ) -> torch.Tensor:
        assert weights.scale_dtype is not None
        return compress_mx_scale(scale, weights.scale_dtype)

    @classmethod
    def _decompress_scale(cls, scale: torch.Tensor) -> torch.Tensor:
        return decompress_mx_scale(scale)

    @classmethod
    def compress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Compress a per-module state dict for MXFP8 format.

        Quantizes the weight via the parent class, then converts the
        float scale to E8M0 exponent format (uint8) for storage.

        :param state_dict: local-name state dict (weight, weight_scale, ...)
        :param scheme: quantization scheme for the weight
        :return: compressed state dict with E8M0 scales
        """
        state_dict = NaiveQuantizationCompressor.compress(state_dict, scheme)

        # Convert float scale to E8M0 exponent format (uint8) for storage
        scale = state_dict["weight_scale"]
        state_dict["weight_scale"] = cls._compress_scale(scale, scheme.weights)

        return state_dict

    @classmethod
    def decompress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Decompress a per-module state dict for MXFP8 format.

        Converts E8M0 exponent scales back to float, then dequantizes
        the weight via the parent class.

        :param state_dict: local-name state dict (weight, weight_scale, ...)
        :param scheme: quantization scheme for the weight
        :return: decompressed state dict with weight in float dtype
        """
        state_dict = state_dict.copy()

        # Convert E8M0 scale back to bfloat16 for consistency with model dtype
        scale = state_dict["weight_scale"]
        state_dict["weight_scale"] = cls._decompress_scale(scale)

        return NaiveQuantizationCompressor.decompress(state_dict, scheme)

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """MXFP8 matches FP8 with group_size=32 and uint8 scale_dtype."""
        return (
            module_type == torch.nn.Linear
            and scheme.weights is not None
            and scheme.weights.num_bits == 8
            and scheme.weights.type == QuantizationType.FLOAT.value
            and scheme.weights.group_size == 32
            and scheme.weights.scale_dtype == torch.uint8
        )
