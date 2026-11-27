# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Compressed forward pass for linear modules with packed INT weights.

All supported schemes are served by a single Triton kernel,
`_pack_quantized_linear_kernel`, which unpacks the int32-packed weights
in-register, dequantizes them to the activation dtype, and multiplies them
with a dense ``tl.dot``. Element ``k`` of a packed row lives at global bit
``k * num_bits``, and values are stored biased by ``2 ** (num_bits - 1)``; the
kernel supports bit widths for which ``32 % num_bits == 0``, so that no value
is split across int32 words.

Weight scales are indexed by ``[n, k // group_size]``, with zero strides along
dimensions which share a scale (tensor, channel, and group strategies), so
that a single code path covers all of them. Asymmetric group and channel zero
points, which the compressor packs along the output dimension, are unpacked
within the kernel as well.

These are weight-only schemes: activations are multiplied in their original
dtype. Group activation ordering, bit widths which span int32 words, group
sizes which are not powers of two of at least 16, and asymmetric tensor
strategies (whose zero points are not packed) fall back to dequantizing the
weight in torch.
"""

import math
from functools import partial
from typing import Callable

import torch
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.offload import get_execution_device
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import (
    dequantize,
    forward_quantize,
)
from compressed_tensors.utils.impl_backend import ENFORCE_EAGER
from compressed_tensors.utils.triton import HAS_TRITON, tl, triton


__all__ = ["get_pack_quantized_linear_forward"]


# strategies for which the compressor packs the zero point along the output
# dimension; other asymmetric strategies keep an unpacked zero point
_PACKED_ZP_STRATEGIES = (
    QuantizationStrategy.GROUP,
    QuantizationStrategy.CHANNEL,
)


def get_pack_quantized_linear_forward(
    module: torch.nn.Module,
) -> Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None:
    """
    Resolve the forward function of a compressed linear module with packed INT
    weights. The Triton kernel is used when possible, otherwise the weight is
    dequantized on the fly.

    :param module: compressed module
    :return: forward function, or None if the module is not a pack-quantized
        linear module
    """
    scheme: QuantizationScheme = module.quantization_scheme
    weights = scheme.weights
    if not (
        isinstance(module, torch.nn.Linear)
        and weights is not None
        and weights.type == QuantizationType.INT
    ):
        return None

    device = get_execution_device(module)
    if (
        ENFORCE_EAGER
        or not HAS_TRITON
        or not isinstance(device, torch.device)
        or device.type not in ("cuda", "xpu")
        or getattr(module, "weight_g_idx", None) is not None  # actorder
    ):
        return _pack_quantized_linear_eager

    config = _get_kernel_config(weights)
    if config is None:
        return _pack_quantized_linear_eager

    num_bits, symmetric, group_size = config
    return partial(
        _pack_quantized_linear_triton,
        num_bits=num_bits,
        symmetric=symmetric,
        group_size=group_size,
    )


def _get_kernel_config(
    weights: QuantizationArgs,
) -> tuple[int, bool, int | None] | None:
    """
    :return: the kernel constexprs ``(num_bits, symmetric, group_size)`` for
    the weight quantization args, where a group size of None means that the
    scale is shared across the whole row. Returns None if the scheme is not
    supported by the kernel
    """
    if 32 % weights.num_bits != 0:
        # values would be split across int32 words
        return None

    strategy = weights.strategy
    if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        if not weights.symmetric and strategy != QuantizationStrategy.GROUP:
            return None  # zero point is not packed
        if not _is_dot_size(weights.group_size):
            return None
        return (weights.num_bits, weights.symmetric, weights.group_size)

    if strategy == QuantizationStrategy.CHANNEL:
        return (weights.num_bits, weights.symmetric, None)

    if strategy == QuantizationStrategy.TENSOR and weights.symmetric:
        return (weights.num_bits, weights.symmetric, None)

    return None


def _is_dot_size(size: int | None) -> bool:
    return size is not None and size >= 16 and math.log2(size).is_integer()


def _dequantize_weight(module: torch.nn.Linear, dtype: torch.dtype) -> torch.Tensor:
    """Unpack and dequantize the compressed weight of a linear module"""
    weights: QuantizationArgs = module.quantization_scheme.weights
    weight_packed = module.weight_packed
    n = weight_packed.shape[0]
    k = module.in_features
    scale = module.weight_scale
    zero_point = getattr(module, "weight_zero_point", None)

    if zero_point is not None and weights.strategy in _PACKED_ZP_STRATEGIES:
        # asymmetric group and channel zero points are packed along the
        # output dimension, one column per scale group
        zero_point = unpack_from_int32(
            zero_point, weights.num_bits, (n, scale.shape[-1]), packed_dim=0
        )

    unpacked = unpack_from_int32(weight_packed, weights.num_bits, (n, k))
    return dequantize(
        x_q=unpacked,
        scale=scale,
        zero_point=zero_point,
        args=weights,
        dtype=dtype,
    )


def _pack_quantized_linear_eager(
    module: torch.nn.Linear, input: torch.Tensor
) -> torch.Tensor:
    """Dequantize the weight and perform a dense linear"""
    scheme: QuantizationScheme = module.quantization_scheme

    if scheme.input_activations is not None:
        input = forward_quantize(module, input, "input", scheme.input_activations)

    weight = _dequantize_weight(module, input.dtype)
    output = torch.nn.functional.linear(input, weight, module.bias)

    if scheme.output_activations is not None:
        output = forward_quantize(module, output, "output", scheme.output_activations)

    return output


def _expand_scale(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """
    Expand a tensor, channel, group, or block scale to shape ``[rows, cols]``,
    where dimensions which share a scale have a stride of zero
    """
    if scale.ndim <= 1:
        scale = scale.reshape(1, 1)

    return scale.expand(rows, cols)


def _pack_quantized_linear_triton(
    module: torch.nn.Linear,
    input: torch.Tensor,
    num_bits: int,
    symmetric: bool,
    group_size: int | None,
) -> torch.Tensor:
    scheme: QuantizationScheme = module.quantization_scheme
    weight_packed = module.weight_packed
    weight_scale = module.weight_scale
    bias = module.bias
    n, k = weight_packed.shape[0], module.in_features

    if scheme.input_activations is not None:
        input = forward_quantize(module, input, "input", scheme.input_activations)

    input_2d = input.reshape(-1, k)
    m = input_2d.shape[0]
    output = torch.empty((m, n), dtype=input.dtype, device=input.device)

    elements_per_word = 32 // num_bits
    if m <= 512:
        # memory bound: narrow tiles along the output dimension keep enough
        # programs in flight to stream the packed weights
        block_m = min(max(triton.next_power_of_2(m), 16), 256)
        block_n, block_k = 32, 128
        num_warps, num_stages = (4 if m <= 64 else 8), 3
    elif m < 2048:
        block_m, block_n, block_k, num_warps, num_stages = 512, 64, 64, 8, 3
    else:
        block_m, block_n, block_k, num_warps, num_stages = 512, 128, 64, 8, 3
    # a tile of packed values must not be split across int32 words
    block_n = max(block_n, elements_per_word)
    block_k = max(block_k, elements_per_word)

    # column count of the expanded scale; for channel and tensor scales the
    # column stride is zero, so that the column index is a dummy
    if group_size is not None:
        num_scale_cols = weight_scale.shape[-1]
    else:
        num_scale_cols = triton.cdiv(k, block_k)
    weight_scale = _expand_scale(weight_scale, n, num_scale_cols)
    if symmetric:
        zero_point = weight_scale  # unused
    else:
        zero_point = _expand_scale(
            module.weight_zero_point, module.weight_zero_point.shape[0], num_scale_cols
        )

    # group size seen by the kernel: scales which are shared across the whole
    # row are indexed by the tile index instead of a group index
    kernel_group_size = group_size or block_k
    groups_per_tile = max(1, block_k // kernel_group_size)

    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    with torch.get_device_module(input.device).device(input.device):
        _pack_quantized_linear_kernel[grid](
            input_2d,
            weight_packed,
            weight_scale,
            zero_point,
            bias if bias is not None else output,  # unused if no bias
            output,
            m,
            n,
            k,
            *input_2d.stride(),
            *weight_packed.stride(),
            *weight_scale.stride(),
            *zero_point.stride(),
            *output.stride(),
            NUM_BITS=num_bits,
            SYMMETRIC=symmetric,
            GROUP_SIZE=kernel_group_size,
            GROUPS_PER_TILE=groups_per_tile,
            HAS_BIAS=bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    output = output.reshape(*input.shape[:-1], n)
    if scheme.output_activations is not None:
        output = forward_quantize(module, output, "output", scheme.output_activations)

    return output


@triton.jit
def _pack_quantized_linear_kernel(
    x_ptr,  # [M, K] activations
    w_ptr,  # [N, cdiv(K * NUM_BITS, 32)] int32-packed weights
    ws_ptr,  # [N, cdiv(K, GROUP_SIZE)] weight scales
    zp_ptr,  # [cdiv(N * NUM_BITS, 32), cdiv(K, GROUP_SIZE)] packed zero points
    bias_ptr,  # [N]
    out_ptr,  # [M, N]
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_wsn,
    stride_wsk,
    stride_zpn,
    stride_zpg,
    stride_om,
    stride_on,
    NUM_BITS: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_TILE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    out = x @ dequant(w).T + bias, where the packed weights are unpacked and
    dequantized to the activation dtype within each contraction tile
    """
    # sub-elements per int32 word, and the mask and bias of the packed values
    E: tl.constexpr = 32 // NUM_BITS
    MASK: tl.constexpr = (1 << NUM_BITS) - 1
    OFFSET: tl.constexpr = 1 << (NUM_BITS - 1)
    # int32 words per tile of the weight and zero point packings
    WORDS_PER_TILE: tl.constexpr = BLOCK_K * NUM_BITS // 32
    ZP_WORDS_PER_TILE: tl.constexpr = BLOCK_N * NUM_BITS // 32
    # columns which share a scale within a tile
    ELEMS_PER_GROUP: tl.constexpr = BLOCK_K // GROUPS_PER_TILE
    shifts = tl.arange(0, E) * NUM_BITS

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_w = tl.arange(0, WORDS_PER_TILE)
    offs_zw = pid_n * ZP_WORDS_PER_TILE + tl.arange(0, ZP_WORDS_PER_TILE)
    m_mask = offs_m < M
    n_mask = offs_n < N
    zp_row_mask = offs_zw < tl.cdiv(N, E)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_wk
    ws_n_ptrs = ws_ptr + offs_n[:, None] * stride_wsn
    zp_row_ptrs = zp_ptr + offs_zw[:, None] * stride_zpn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (kt * BLOCK_K + offs_k) < K
        w_mask = (kt * WORDS_PER_TILE + offs_w) < tl.cdiv(K, E)

        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_words = tl.load(w_ptrs, mask=n_mask[:, None] & w_mask[None, :], other=0)

        # element k of a packed row lives at global bit k * NUM_BITS, so the
        # sub-elements of the loaded words make up the columns w * E + e
        u = (w_words[:, :, None] >> shifts[None, None, :]) & MASK
        u = tl.reshape(u, (BLOCK_N, BLOCK_K))

        # dequantize (u - OFFSET - zp) * ws as u * ws - c, folding the
        # quantization offset and zero point into a per-group constant c.
        # Padding bits decode to arbitrary values, which is fine because the
        # activations are zero outside of [0, K)
        offs_g = (kt * BLOCK_K) // GROUP_SIZE + tl.arange(0, GROUPS_PER_TILE)
        g_mask = offs_g * GROUP_SIZE < K
        ws = tl.load(
            ws_n_ptrs + offs_g[None, :] * stride_wsk,
            mask=n_mask[:, None] & g_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if SYMMETRIC:
            c = ws * OFFSET
        else:
            # zero points are packed along the output dimension: element n of
            # a packed column lives at bit n * NUM_BITS
            zp_words = tl.load(
                zp_row_ptrs + offs_g[None, :] * stride_zpg,
                mask=zp_row_mask[:, None] & g_mask[None, :],
                other=0,
            )
            uz = (zp_words[:, None, :] >> shifts[None, :, None]) & MASK
            zp_q = tl.reshape(uz, (BLOCK_N, GROUPS_PER_TILE)) - OFFSET
            c = (zp_q.to(tl.float32) + OFFSET) * ws

        ws = tl.reshape(
            tl.broadcast_to(
                ws[:, :, None], (BLOCK_N, GROUPS_PER_TILE, ELEMS_PER_GROUP)
            ),
            (BLOCK_N, BLOCK_K),
        )
        c = tl.reshape(
            tl.broadcast_to(c[:, :, None], (BLOCK_N, GROUPS_PER_TILE, ELEMS_PER_GROUP)),
            (BLOCK_N, BLOCK_K),
        )
        w = (u.to(tl.float32) * ws - c).to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, tl.trans(w))

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += WORDS_PER_TILE * stride_wk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias.to(tl.float32)[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )
