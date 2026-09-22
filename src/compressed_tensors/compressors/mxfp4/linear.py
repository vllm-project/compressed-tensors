# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Backend implementations for the MXFP4 linear forward pass.

Three implementations back ``MXFP4PackedCompressor.forward``, dispatched by
``ImplBackend`` in priority order:

1. ``fp4`` (Blackwell only): a real MXFP4 matmul that feeds packed FP4 weights
   (and, for weight+activation schemes, packed FP4 activations) directly to the
   FP4 tensor cores via ``tl.dot_scaled``.
2. ``emulation`` (any Triton GPU): a fused dequantize-and-matmul Triton kernel.
   Weights are unpacked and scaled inline, then multiplied in bf16 -- the actual
   matmul happens in higher precision (emulation).
3. eager fallback (the ``forward`` entrypoint itself, defined on the compressor):
   ``forward_quantize`` for activations, unpack + ``decompress_mx_scale`` for
   weights, then a dense ``F.linear``.

Weights are stored as packed FP4 (E2M1) nibbles (two per uint8, first element in
the low nibble) with an E8M0 (bias-127) group scale per 32 input columns -- a
layout that is directly consumable as MX-format operands.
"""

import torch
from compressed_tensors.compressors.mx_utils import (
    compress_mx_scale,
    decompress_mx_scale,
)
from compressed_tensors.compressors.nvfp4.helpers import (
    pack_fp4_to_uint8,
    unpack_fp4_from_uint8,
)
from compressed_tensors.quantization.lifecycle.forward import forward_quantize, quantize
from compressed_tensors.quantization.utils.helpers import compute_dynamic_scales_and_zp
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import HAS_TRITON, tl, triton
from functools import lru_cache


__all__ = ["dequantize_mxfp4_weight"]


# Number of input columns that share a single MX group scale.
MX_GROUP_SIZE = 32
# Packed uint8 columns per group scale (two FP4 nibbles per byte).
_PACKED_PER_GROUP = MX_GROUP_SIZE // 2


# ---------------------------------------------------------------------------
# Backend selection requirements
# ---------------------------------------------------------------------------


@lru_cache
def _is_blackwell(device: torch.device) -> bool:
    """FP4 tensor cores require CUDA compute capability >= 10.0 (Blackwell)."""
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 10


def _emulation_req(module: torch.nn.Module, input: torch.Tensor) -> bool:
    return HAS_TRITON and (input.is_cuda or input.is_xpu)


def _fp4_req(module: torch.nn.Module, input: torch.Tensor) -> bool:
    return _emulation_req(module, input) and _is_blackwell(input.device)


# ---------------------------------------------------------------------------
# Shared weight dequantization (eager)
# ---------------------------------------------------------------------------


def dequantize_mxfp4_weight(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Unpack and dequantize MXFP4 weights into a dense float tensor.

    :param weight_packed: packed fp4 weights ``[N, K // 2]`` (uint8)
    :param weight_scale: E8M0 (bias-127) group scales ``[N, K // 32]`` (uint8)
    :param dtype: output dtype
    :return: dense weight ``[N, K]``
    """
    n, kp = weight_packed.shape
    k = kp * 2
    unpacked = unpack_fp4_from_uint8(weight_packed, n, k, dtype=dtype)
    scale = decompress_mx_scale(weight_scale).to(dtype)

    weight = unpacked.unflatten(-1, (k // MX_GROUP_SIZE, MX_GROUP_SIZE))
    weight = (weight * scale.unsqueeze(-1)).flatten(-2)
    return weight.to(dtype)


# ---------------------------------------------------------------------------
# Emulation backend: fused dequant + bf16 matmul
# ---------------------------------------------------------------------------


@triton.jit
def _fp4_nibble_to_float(nibble):
    """Decode a 4-bit E2M1 nibble (magnitude in bits 0-2, sign in bit 3)."""
    mag = nibble & 0x07
    sign = (nibble & 0x08) != 0
    val = tl.where(
        mag == 0,
        0.0,
        tl.where(
            mag == 1,
            0.5,
            tl.where(
                mag == 2,
                1.0,
                tl.where(
                    mag == 3,
                    1.5,
                    tl.where(
                        mag == 4,
                        2.0,
                        tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where(sign, -val, val)


@triton.jit
def _mxfp4_emulation_kernel(
    x_ptr,  # [M, K] activations (already fake-quantized)
    w_ptr,  # [N, K // 2] packed fp4 weights (uint8)
    s_ptr,  # [N, K // 32] E8M0 group scales (uint8)
    out_ptr,  # [M, N] output
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wj,
    stride_sn,
    stride_sg,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KP: tl.constexpr,
    PACKED_PER_GROUP: tl.constexpr,
):
    """
    Fused dequantize-and-matmul: out = x @ dequant(w).T

    Each byte of ``w`` holds two fp4 weights for consecutive input columns
    (even column in the low nibble, odd column in the high nibble), so the
    contraction is split into even/odd halves that are accumulated separately.
    The dequantized weights are cast to bf16 and multiplied in bf16 -- this is
    an emulation of the fp4 matmul, not a real fp4 tensor-core operation.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kp = tl.arange(0, BLOCK_KP)

    kp = K // 2  # number of packed columns
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kt in range(0, tl.cdiv(kp, BLOCK_KP)):
        cur_kp = kt * BLOCK_KP + offs_kp  # [BLOCK_KP]
        kp_mask = cur_kp < kp
        wn_mask = kp_mask[:, None] & (offs_n[None, :] < N)

        # load packed weights [BLOCK_KP, BLOCK_N] and split into nibbles
        w_ptrs = w_ptr + cur_kp[:, None] * stride_wj + offs_n[None, :] * stride_wn
        wbytes = tl.load(w_ptrs, mask=wn_mask, other=0)
        w_even = _fp4_nibble_to_float(wbytes & 0x0F)  # even input columns
        w_odd = _fp4_nibble_to_float((wbytes >> 4) & 0x0F)  # odd input columns

        # load group scale [BLOCK_KP, BLOCK_N] (shared by the even/odd pair)
        sg = cur_kp // PACKED_PER_GROUP
        s_ptrs = s_ptr + sg[:, None] * stride_sg + offs_n[None, :] * stride_sn
        s_exp = tl.load(s_ptrs, mask=wn_mask, other=127).to(tl.int32)
        s_val = tl.exp2((s_exp - 127).to(tl.float32))

        w_even = (w_even * s_val).to(tl.bfloat16)
        w_odd = (w_odd * s_val).to(tl.bfloat16)

        # load matching activation halves [BLOCK_M, BLOCK_KP]
        xm_mask = (offs_m[:, None] < M) & kp_mask[None, :]
        x_even_ptrs = (
            x_ptr + offs_m[:, None] * stride_xm + (2 * cur_kp)[None, :] * stride_xk
        )
        x_odd_ptrs = (
            x_ptr + offs_m[:, None] * stride_xm + (2 * cur_kp + 1)[None, :] * stride_xk
        )
        x_even = tl.load(x_even_ptrs, mask=xm_mask, other=0.0).to(tl.bfloat16)
        x_odd = tl.load(x_odd_ptrs, mask=xm_mask, other=0.0).to(tl.bfloat16)

        acc += tl.dot(x_even, w_even)
        acc += tl.dot(x_odd, w_odd)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def _mxfp4_emulation_matmul(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    n, kp = weight_packed.shape
    k = kp * 2
    m = x.shape[0]

    out = torch.empty((m, n), dtype=torch.float32, device=x.device)

    BLOCK_M, BLOCK_N, BLOCK_KP = 32, 64, 64
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    with torch.get_device_module().device(x.device):
        _mxfp4_emulation_kernel[grid](
            x,
            weight_packed,
            weight_scale,
            out,
            m,
            n,
            k,
            x.stride(0),
            x.stride(1),
            weight_packed.stride(0),
            weight_packed.stride(1),
            weight_scale.stride(0),
            weight_scale.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_KP=BLOCK_KP,
            PACKED_PER_GROUP=_PACKED_PER_GROUP,
        )

    out = out.to(x.dtype)
    if bias is not None:
        out += bias
    return out


@ImplBackend.register("mxfp4_forward", _emulation_req, 1)
def mxfp4_forward_emulation(
    module: torch.nn.Module, input: torch.Tensor
) -> torch.Tensor:
    """Triton emulation backend: fake-quantize activations, fused dequant matmul."""
    scheme = module.quantization_scheme
    if scheme.input_activations is not None:
        input = forward_quantize(module, input, "input", scheme.input_activations)

    original_shape = input.shape
    input_2d = input.reshape(-1, original_shape[-1])

    output = _mxfp4_emulation_matmul(
        input_2d,
        module.weight_packed,
        module.weight_scale,
        getattr(module, "bias", None),
    )
    return output.reshape(*original_shape[:-1], output.shape[-1])


# ---------------------------------------------------------------------------
# FP4 tensor-core backend: real MXFP4 matmul via tl.dot_scaled
# ---------------------------------------------------------------------------


@triton.jit
def _mxfp4_scaled_kernel(
    a_ptr,  # lhs: [M, K // 2] packed fp4 (uint8) or [M, K] bf16
    a_scale_ptr,  # [M, K // 32] E8M0 (uint8), unused when lhs is bf16
    b_ptr,  # rhs: [K // 2, N] packed fp4 (uint8)
    b_scale_ptr,  # [N, K // 32] E8M0 (uint8)
    c_ptr,  # [M, N] output
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_asm,
    stride_asg,
    stride_bk,
    stride_bn,
    stride_bsn,
    stride_bsg,
    stride_cm,
    stride_cn,
    LHS_FP4: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Microscaled matmul on FP4 tensor cores: out = x @ dequant(w).T

    ``b`` is the packed weight transposed to ``[K // 2, N]`` so it represents
    ``weight.T`` in MX ``e2m1`` format. ``tl.dot_scaled`` consumes the packed
    fp4 nibbles and E8M0 scales natively on Blackwell.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    PACK_K: tl.constexpr = BLOCK_K // 2
    SCALE_K: tl.constexpr = BLOCK_K // 32

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        offs_bk = kt * PACK_K + tl.arange(0, PACK_K)
        b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_bk[:, None] < (K // 2)) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0)

        offs_sg = kt * SCALE_K + tl.arange(0, SCALE_K)
        bs_ptrs = (
            b_scale_ptr + offs_n[:, None] * stride_bsn + offs_sg[None, :] * stride_bsg
        )
        bs_mask = (offs_n[:, None] < N) & (offs_sg[None, :] < (K // 32))
        b_scale = tl.load(bs_ptrs, mask=bs_mask, other=127)

        if LHS_FP4:
            offs_ak = kt * PACK_K + tl.arange(0, PACK_K)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_ak[None, :] * stride_ak
            a_mask = (offs_m[:, None] < M) & (offs_ak[None, :] < (K // 2))
            a = tl.load(a_ptrs, mask=a_mask, other=0)

            as_ptrs = (
                a_scale_ptr
                + offs_m[:, None] * stride_asm
                + offs_sg[None, :] * stride_asg
            )
            as_mask = (offs_m[:, None] < M) & (offs_sg[None, :] < (K // 32))
            a_scale = tl.load(as_ptrs, mask=as_mask, other=127)

            acc = tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1", acc=acc)
        else:
            offs_ak = kt * BLOCK_K + tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_ak[None, :] * stride_ak
            a_mask = (offs_m[:, None] < M) & (offs_ak[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.bfloat16)

            acc = tl.dot_scaled(a, None, "bf16", b, b_scale, "e2m1", acc=acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _mxfp4_scaled_matmul(
    a: torch.Tensor,
    a_scale: torch.Tensor | None,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    n, kp = weight_packed.shape
    k = kp * 2
    m = a.shape[0]
    lhs_fp4 = a_scale is not None

    # rhs = weight.T in packed e2m1 layout: [K // 2, N]
    b = weight_packed.t().contiguous()
    c = torch.empty((m, n), dtype=torch.float32, device=a.device)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 128
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    with torch.get_device_module().device(a.device):
        _mxfp4_scaled_kernel[grid](
            a,
            a_scale if lhs_fp4 else a,  # dummy pointer when lhs is bf16
            b,
            weight_scale,
            c,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            a_scale.stride(0) if lhs_fp4 else 0,
            a_scale.stride(1) if lhs_fp4 else 0,
            b.stride(0),
            b.stride(1),
            weight_scale.stride(0),
            weight_scale.stride(1),
            c.stride(0),
            c.stride(1),
            LHS_FP4=lhs_fp4,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

    c = c.to(out_dtype)
    if bias is not None:
        c += bias
    return c


def _quantize_activations_to_mxfp4(
    input_2d: torch.Tensor, module: torch.nn.Module, args
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize activations into packed fp4 + E8M0 group scales."""
    scale, zero_point = compute_dynamic_scales_and_zp(
        value=input_2d, args=args, module=module
    )
    quantized = quantize(x=input_2d, scale=scale, zero_point=zero_point, args=args)
    packed = pack_fp4_to_uint8(quantized)
    scale_e8m0 = compress_mx_scale(scale, torch.uint8)
    return packed, scale_e8m0


@ImplBackend.register("mxfp4_forward", _fp4_req, 0)
def mxfp4_forward_fp4(module: torch.nn.Module, input: torch.Tensor) -> torch.Tensor:
    """FP4 tensor-core backend: real MXFP4 matmul via ``tl.dot_scaled``."""
    scheme = module.quantization_scheme
    original_shape = input.shape
    input_2d = input.reshape(-1, original_shape[-1])
    bias = getattr(module, "bias", None)

    if scheme.input_activations is not None:
        # weight + activation fp4: quantize activations to packed fp4
        a, a_scale = _quantize_activations_to_mxfp4(
            input_2d, module, scheme.input_activations
        )
    else:
        # weight-only: keep activations in bf16, weights stay fp4
        a, a_scale = input_2d.to(torch.bfloat16), None

    output = _mxfp4_scaled_matmul(
        a,
        a_scale,
        module.weight_packed,
        module.weight_scale,
        bias,
        out_dtype=input.dtype,
    )
    return output.reshape(*original_shape[:-1], output.shape[-1])
