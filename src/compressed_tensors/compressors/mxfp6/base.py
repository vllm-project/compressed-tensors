# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import (
    COMPRESSIBLE_MODULE_TYPES,
    BaseCompressor,
)
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
from compressed_tensors.compressors.mxfp6.helpers import (
    pack_fp6_to_uint8,
    unpack_fp6_from_uint8,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationScheme, QuantizationType
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.utils import TensorStateDict


__all__ = ["MXFP6PackedCompressor"]


@BaseCompressor.register(name=CompressionFormat.mxfp6_pack_quantized.value)
class MXFP6PackedCompressor(MXFP4PackedCompressor):
    """
    Compressor for MXFP6 (E2M3) quantized models.

    Packs four FP6 values into three bytes and stores scales as E8M0 exponents.
    """

    @classmethod
    def compress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        state_dict = state_dict.copy()
        weight = state_dict.pop("weight")
        scale = state_dict.pop("weight_scale")

        quantized = quantize(
            x=weight, scale=scale, zero_point=None, args=scheme.weights
        )
        state_dict["weight_packed"] = pack_fp6_to_uint8(quantized)
        state_dict["weight_scale"] = cls._compress_scale(scale, scheme.weights)
        return cls._remove_symmetric_zp(state_dict, scheme)

    @classmethod
    def decompress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        state_dict = state_dict.copy()
        packed = state_dict.pop("weight_packed")

        m, n = packed.shape
        unpacked = unpack_fp6_from_uint8(packed, m, n * 4 // 3)
        scale = cls._decompress_scale(state_dict["weight_scale"], unpacked.dtype)

        state_dict["weight"] = dequantize(
            x_q=unpacked, scale=scale, args=scheme.weights, dtype=unpacked.dtype
        )
        state_dict["weight_scale"] = torch.nn.Parameter(scale, requires_grad=False)
        return state_dict

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """MXFP6 matches FP6 with group_size=32."""
        return (
            issubclass(module_type, COMPRESSIBLE_MODULE_TYPES)
            and scheme.weights is not None
            and scheme.weights.num_bits == 6
            and scheme.weights.type == QuantizationType.FLOAT.value
            and scheme.weights.group_size == 32
        )
