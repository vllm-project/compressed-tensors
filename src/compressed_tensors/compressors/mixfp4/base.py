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
from compressed_tensors.compressors.nvfp4.helpers import FLOAT_TO_E2M1
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.utils import TensorStateDict


__all__ = [
    "MixFP4PackedCompressor",
    "pack_mixfp4_to_uint8",
    "unpack_mixfp4_from_uint8",
]


_E2M1_CODEBOOK = torch.tensor(FLOAT_TO_E2M1, dtype=torch.float32)
_FP4_MAX = 6.0
_INT4_MAX = 7
_FP8_FLAG_BIT = 0x80
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
        scale_dtype = weights.scale_dtype or torch.float8_e4m3fn
        return scale.to(scale_dtype)

    @classmethod
    def _decompress_scale(cls, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return scale.to(dtype)

    @classmethod
    def _validate_scheme(cls, scheme: QuantizationScheme) -> QuantizationArgs:
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
        return (
            weights is not None
            and weights.num_bits == 4
            and weights.type == QuantizationType.FLOAT.value
            and weights.strategy == QuantizationStrategy.TENSOR_GROUP.value
            and weights.symmetric is True
            and weights.group_size == 16
            and weights.scale_dtype in (None, torch.float8_e4m3fn)
        )


def _split_scale(
    scale: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (is_int4 flag, effective dequant scale) from flagged FP8 scales."""
    if global_scale.numel() != 1:
        raise ValueError("MixFP4 expects a scalar weight_global_scale")

    if scale.dtype != torch.float8_e4m3fn:
        is_int4 = torch.signbit(scale)
        mag_fp8 = scale.abs().to(torch.float8_e4m3fn)
        mag_f32 = mag_fp8.to(torch.float32)
    else:
        raw = scale.contiguous().view(torch.uint8)
        is_int4 = (raw & _FP8_FLAG_BIT).ne(0)
        mag_fp8 = (raw & 0x7F).view(torch.float8_e4m3fn)
        mag_f32 = mag_fp8.to(torch.float32)

    gs = global_scale.to(torch.float32).reshape(-1)[0]
    if not global_scale.is_meta:
        if not torch.isfinite(gs).item() or gs.item() <= 0:
            raise ValueError("MixFP4 expects a finite positive weight_global_scale")
    eff_scale = (mag_f32 / gs).clamp(min=torch.finfo(torch.float32).tiny)
    return is_int4, eff_scale


def pack_mixfp4_to_uint8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Pack a dense weight matrix using per-group MixFP4 scale flags."""
    if weight.ndim != 2:
        raise ValueError("MixFP4 weight must be a rank-2 tensor")
    if group_size <= 0:
        raise ValueError("MixFP4 group_size must be positive")
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"weight columns {k} must be divisible by {group_size}")
    if k % 2 != 0:
        raise ValueError("MixFP4 packing requires an even number of columns")

    groups = k // group_size
    if scale.shape != (n, groups):
        raise ValueError(
            f"MixFP4 scale shape must be {(n, groups)}, got {tuple(scale.shape)}"
        )
    is_int4, eff_scale = _split_scale(
        scale.to(weight.device), global_scale.to(weight.device)
    )

    scaled = (
        weight.to(torch.float32).view(n, groups, group_size) / eff_scale.unsqueeze(-1)
    )
    nibble_fp4 = _encode_fp4_nibble(_round_to_e2m1(scaled))
    q_int4 = scaled.round().clamp(-_INT4_MAX, _INT4_MAX).to(torch.int32)
    nibble_int4 = _encode_int4_nibble(q_int4)

    nibbles = torch.where(is_int4.unsqueeze(-1), nibble_int4, nibble_fp4).view(n, k)
    low = nibbles[:, 0::2]
    high = nibbles[:, 1::2]
    return ((high << 4) | low).to(torch.uint8)


def unpack_mixfp4_from_uint8(
    weight_packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Unpack and dequantize MixFP4 weights for validation and decompression."""
    if weight_packed.dtype != torch.uint8:
        raise ValueError("MixFP4 packed weights must be uint8")
    if weight_packed.ndim != 2:
        raise ValueError("MixFP4 packed weights must be a rank-2 tensor")
    if group_size <= 0:
        raise ValueError("MixFP4 group_size must be positive")

    n, k_half = weight_packed.shape
    k = k_half * 2
    if k % group_size != 0:
        raise ValueError(f"packed weight columns {k} must be divisible by {group_size}")
    groups = k // group_size
    if scale.shape != (n, groups):
        raise ValueError(
            f"MixFP4 scale shape must be {(n, groups)}, got {tuple(scale.shape)}"
        )
    is_int4, eff_scale = _split_scale(
        scale.to(weight_packed.device), global_scale.to(weight_packed.device)
    )

    low = (weight_packed & 0x0F).to(torch.uint8)
    high = ((weight_packed >> 4) & 0x0F).to(torch.uint8)
    nibbles = torch.empty(n, k, dtype=torch.uint8, device=weight_packed.device)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high

    val_fp4 = _decode_fp4_nibble(nibbles)
    val_int4 = _decode_int4_nibble(nibbles)
    is_int4_elem = is_int4.unsqueeze(-1).expand(n, groups, group_size).reshape(n, k)
    values = torch.where(is_int4_elem, val_int4, val_fp4)
    eff_scale_elem = eff_scale.unsqueeze(-1).expand(n, groups, group_size).reshape(n, k)
    return (values * eff_scale_elem).to(dtype)


def _round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    codebook = _E2M1_CODEBOOK.to(x.device)
    sign = torch.sign(x)
    mag = x.abs().clamp(max=_FP4_MAX)
    idx = (mag.unsqueeze(-1) - codebook).abs().argmin(dim=-1)
    return sign * codebook[idx]


def _encode_fp4_nibble(q_fp4: torch.Tensor) -> torch.Tensor:
    codebook = _E2M1_CODEBOOK.to(q_fp4.device)
    sign_bit = torch.signbit(q_fp4).to(torch.uint8) << 3
    idx = (q_fp4.abs().unsqueeze(-1) == codebook).to(torch.uint8).argmax(dim=-1)
    return sign_bit | idx.to(torch.uint8)


def _encode_int4_nibble(q_int4: torch.Tensor) -> torch.Tensor:
    sign_bit = (q_int4 < 0).to(torch.uint8) << 3
    mag = q_int4.abs().to(torch.uint8).clamp(max=_INT4_MAX)
    return sign_bit | mag


def _decode_fp4_nibble(nibble: torch.Tensor) -> torch.Tensor:
    codebook = _E2M1_CODEBOOK.to(nibble.device)
    sign = ((nibble >> 3) & 1).to(torch.bool)
    mag = codebook[(nibble & 0x07).to(torch.long)]
    return torch.where(sign, -mag, mag)


def _decode_int4_nibble(nibble: torch.Tensor) -> torch.Tensor:
    sign = ((nibble >> 3) & 1).to(torch.bool)
    mag = (nibble & 0x07).to(torch.float32)
    return torch.where(sign, -mag, mag)
