# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Packed compressor for MixFP4 weight-only quantization.

MixFP4 keeps the NVFP4 W4A16 tensor layout but uses the redundant sign bit of
per-group float8_e4m3fn scales as a block-format flag: 0 selects FP4 E2M1 and
1 selects signed INT4. No extra persistent side metadata is introduced.
"""

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.mixfp4.helpers import (
    pack_mixfp4_to_uint8,
    unpack_mixfp4_from_uint8,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.utils import TensorStateDict


__all__ = ["MixFP4PackedCompressor"]


_MIXFP4_FORMATS = {
    CompressionFormat.mixfp4_pack_quantized.value,
    "mixed-fp4-int4-pack-quantized",
}
_MIXFP4_OBSERVERS = {"mixfp4", "mixed_fp4_int4"}


@BaseCompressor.register(
    name=CompressionFormat.mixfp4_pack_quantized.value,
    alias="mixed-fp4-int4-pack-quantized",
)
class MixFP4PackedCompressor(BaseCompressor):
    """
    Compressor for MixFP4 quantized models.

    The observer chooses FP4 or INT4 per 16-element block and stores that choice
    in the sign bit of ``weight_scale``. This compressor preserves the scale
    tensor as the only flag carrier and emits one packed 4-bit code per weight.
    """

    @classmethod
    def _compress_scale(
        cls, scale: torch.Tensor, weights: QuantizationArgs
    ) -> torch.Tensor:
        """Cast flagged per-group scales to the configured storage dtype."""
        scale_dtype = weights.scale_dtype or torch.float8_e4m3fn
        return scale.to(scale_dtype)

    @classmethod
    def _decompress_scale(cls, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Cast stored scales to the requested dense validation dtype."""
        return scale.to(dtype)

    @classmethod
    def _validate_scheme(cls, scheme: QuantizationScheme) -> QuantizationArgs:
        """Validate that the scheme matches the MixFP4 W4A16 contract."""
        weights = scheme.weights
        if weights is None:
            raise ValueError("MixFP4 compression requires weight quantization args")
        if not cls._is_mixfp4_weight_args(weights):
            raise ValueError(
                "MixFP4 requires symmetric TENSOR_GROUP float4 weights with "
                "group_size=16 and float8_e4m3fn scales"
            )
        return weights

    @classmethod
    def compress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """Pack dense weights while preserving flags in ``weight_scale``."""
        state_dict = state_dict.copy()
        weight = state_dict.pop("weight")
        scale = state_dict.pop("weight_scale")
        global_scale = state_dict.get("weight_global_scale", None)
        weights = cls._validate_scheme(scheme)

        if global_scale is None:
            raise ValueError("MixFP4 compression requires weight_global_scale")

        state_dict["weight_packed"] = pack_mixfp4_to_uint8(
            weight=weight,
            scale=scale,
            global_scale=global_scale,
            group_size=weights.group_size,
        )
        state_dict["weight_scale"] = cls._compress_scale(scale, weights)
        state_dict = cls._remove_symmetric_zp(state_dict, scheme)
        return state_dict

    @classmethod
    def decompress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """Restore dense BF16 weights from packed MixFP4 checkpoint tensors."""
        state_dict = state_dict.copy()
        packed = state_dict.pop("weight_packed")
        scale = state_dict.get("weight_scale")
        global_scale = state_dict.get("weight_global_scale", None)
        weights = cls._validate_scheme(scheme)

        if scale is None or global_scale is None:
            raise ValueError(
                "MixFP4 decompression requires weight_scale and global_scale"
            )

        scale_float = cls._decompress_scale(scale, torch.bfloat16)
        state_dict["weight"] = unpack_mixfp4_from_uint8(
            weight_packed=packed,
            scale=scale,
            global_scale=global_scale,
            group_size=weights.group_size,
            dtype=torch.bfloat16,
        )
        state_dict["weight_scale"] = torch.nn.Parameter(
            scale_float, requires_grad=False
        )
        return state_dict

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """Match only explicit MixFP4 schemes, not ordinary NVFP4 W4A16."""
        weights = scheme.weights
        if module_type != torch.nn.Linear or not cls._is_mixfp4_weight_args(weights):
            return False

        scheme_format = getattr(scheme.format, "value", scheme.format)
        observer = getattr(weights, "observer", None)
        return scheme_format in _MIXFP4_FORMATS or observer in _MIXFP4_OBSERVERS

    @staticmethod
    def _is_mixfp4_weight_args(weights: QuantizationArgs | None) -> bool:
        """Return whether weight args describe symmetric group-16 W4A16 FP4."""
        return (
            weights is not None
            and weights.num_bits == 4
            and weights.type == QuantizationType.FLOAT.value
            and weights.strategy == QuantizationStrategy.TENSOR_GROUP.value
            and weights.symmetric is True
            and weights.group_size == 16
            and weights.scale_dtype in (None, torch.float8_e4m3fn)
        )
