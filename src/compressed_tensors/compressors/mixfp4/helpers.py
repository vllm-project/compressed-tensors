# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.nvfp4.helpers import FLOAT_TO_E2M1


__all__ = ["pack_mixfp4_to_uint8", "unpack_mixfp4_from_uint8"]


_E2M1_CODEBOOK = torch.tensor(FLOAT_TO_E2M1, dtype=torch.float32)
_FP4_MAX = 6.0
_INT4_MAX = 7
_FP8_FLAG_BIT = 0x80


def _split_scale(
    scale: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (is_int4 flag, effective dequant scale) from flagged FP8 scales."""
    if scale.dtype != torch.float8_e4m3fn:
        raise ValueError("MixFP4 expects float8_e4m3fn weight_scale")

    raw = scale.contiguous().view(torch.uint8)
    is_int4 = (raw & _FP8_FLAG_BIT).ne(0)
    mag_fp8 = (raw & 0x7F).view(torch.float8_e4m3fn)
    mag_f32 = mag_fp8.to(torch.float32)

    eff_scale = _apply_global_scale(mag_f32, global_scale)
    return is_int4, eff_scale


def _apply_global_scale(scale: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Apply NVFP4-style global scale broadcasting."""
    global_scale = global_scale.to(torch.float32)
    if not global_scale.is_meta:
        if not torch.isfinite(global_scale).all().item() or not (
            global_scale > 0
        ).all().item():
            raise ValueError("MixFP4 expects a finite positive weight_global_scale")
    if global_scale.ndim == 1 and global_scale.numel() == scale.size(0):
        global_scale = global_scale.reshape(-1, *([1] * (scale.ndim - 1)))
    return scale / global_scale


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

    zero_scale = eff_scale == 0
    safe_eff_scale = torch.where(zero_scale, torch.ones_like(eff_scale), eff_scale)
    scaled = (
        weight.to(torch.float32).view(n, groups, group_size)
        / safe_eff_scale.unsqueeze(-1)
    )
    nibble_fp4 = _encode_fp4_nibble(_round_to_e2m1(scaled))
    q_int4 = scaled.round().clamp(-_INT4_MAX, _INT4_MAX).to(torch.int32)
    nibble_int4 = _encode_int4_nibble(q_int4)

    nibbles = torch.where(is_int4.unsqueeze(-1), nibble_int4, nibble_fp4)
    nibbles = torch.where(zero_scale.unsqueeze(-1), torch.zeros_like(nibbles), nibbles)
    nibbles = nibbles.view(n, k)
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
    """Round values to the nearest unsigned E2M1 magnitude with sign."""
    codebook = _E2M1_CODEBOOK.to(x.device)
    sign = torch.sign(x)
    mag = x.abs().clamp(max=_FP4_MAX)
    idx = (mag.unsqueeze(-1) - codebook).abs().argmin(dim=-1)
    return sign * codebook[idx]


def _encode_fp4_nibble(q_fp4: torch.Tensor) -> torch.Tensor:
    """Encode signed E2M1 FP4 values as one 4-bit code per element."""
    codebook = _E2M1_CODEBOOK.to(q_fp4.device)
    sign_bit = torch.signbit(q_fp4).to(torch.uint8) << 3
    idx = (q_fp4.abs().unsqueeze(-1) - codebook).abs().argmin(dim=-1)
    return sign_bit | idx.to(torch.uint8)


def _encode_int4_nibble(q_int4: torch.Tensor) -> torch.Tensor:
    """Encode signed INT4 values using sign-magnitude nibbles."""
    sign_bit = (q_int4 < 0).to(torch.uint8) << 3
    mag = q_int4.abs().to(torch.uint8).clamp(max=_INT4_MAX)
    return sign_bit | mag


def _decode_fp4_nibble(nibble: torch.Tensor) -> torch.Tensor:
    """Decode one E2M1 FP4 sign-magnitude nibble per element."""
    codebook = _E2M1_CODEBOOK.to(nibble.device)
    sign = ((nibble >> 3) & 1).to(torch.bool)
    mag = codebook[(nibble & 0x07).to(torch.long)]
    return torch.where(sign, -mag, mag)


def _decode_int4_nibble(nibble: torch.Tensor) -> torch.Tensor:
    """Decode one signed INT4 sign-magnitude nibble per element."""
    sign = ((nibble >> 3) & 1).to(torch.bool)
    mag = (nibble & 0x07).to(torch.float32)
    return torch.where(sign, -mag, mag)
