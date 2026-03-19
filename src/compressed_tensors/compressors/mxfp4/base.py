# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationType,
)
from compressed_tensors.utils import TensorStateDict


__all__ = ["MXFP4PackedCompressor"]


@BaseCompressor.register(name=CompressionFormat.mxfp4_pack_quantized.value)
class MXFP4PackedCompressor(NVFP4PackedCompressor):
    """
    Compressor for MXFP4 quantized models.

    Overrides scale compression to use log2 encoding (bias-127 exponent).
    Decompression is not implemented for this format.
    """

    @classmethod
    def _compress_scale(
        cls, scale: torch.Tensor, weights: QuantizationArgs
    ) -> torch.Tensor:
        assert weights.scale_dtype is not None
        scale_exp = 127 + torch.floor(torch.log2(scale)).to(torch.int32)
        return scale_exp.to(weights.scale_dtype)

    @classmethod
    def decompress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Decompress a per-module state dict.

        MXFP4 decompression is currently not supported.

        :param state_dict: local-name state dict (weight_packed, weight_scale, …)
        :param scheme: quantization scheme for the weight
        :return: decompressed state dict with weight in float dtype
        """
        raise NotImplementedError("MXFP4 decompression is currently not supported")

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """MXFP4 matches FP4 with group_size=32."""
        return (
            module_type == torch.nn.Linear
            and scheme.weights is not None
            and scheme.weights.num_bits == 4
            and scheme.weights.type == QuantizationType.FLOAT.value
            and scheme.weights.group_size == 32
        )
