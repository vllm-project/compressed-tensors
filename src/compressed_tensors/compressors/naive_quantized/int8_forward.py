# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Compressed forward pass for linear modules with INT weights (W8A8, W4A8, ...).

All supported schemes are served by a single Triton kernel, `_int8_linear_kernel`.
INT weights of any bit width are stored as int8 values in range, and are multiplied
with int8-quantized activations by an int8 `tl.dot` with an int32 accumulator.
Weight scales are indexed by ``[n // W_GROUP_N, k // BLOCK_K]``, with zero strides
along dimensions which share a scale (tensor, channel, and group strategies), so
that the weight scale of each tile is a vector over output columns which is applied
to the accumulator.

Activations which are quantized to INT with a dynamic symmetric per-token scale
are quantized within the kernel: a pre-pass over K computes the per-row amax and
its scale, then each tile is clamped and rounded to int8. Otherwise, the eager
fallback fake quantizes the activations in torch and dequantizes the weight on the
fly.
"""

import math
from functools import partial
from typing import Callable

import torch
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


__all__ = ["get_int8_linear_forward"]


# contraction tile size, when not constrained by the quantization group size
DEFAULT_BLOCK_K = 128


def get_int8_linear_forward(
    module: torch.nn.Module,
) -> Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None:
    """
    Resolve the forward function of a compressed linear module with INT weights.
    The Triton kernel is used when possible, otherwise the weight is dequantized
    on the fly.

    :param module: compressed module
    :return: forward function, or None if the module is not an INT linear module
    """
    scheme: QuantizationScheme = module.quantization_scheme
    weights = scheme.weights
    if not (
        isinstance(module, torch.nn.Linear)
        and weights is not None
        and weights.type == QuantizationType.INT
    ):
        return None

    weight_block = _get_weight_block(weights)
    device = get_execution_device(module)
    if (
        ENFORCE_EAGER
        or not HAS_TRITON
        or not isinstance(device, torch.device)
        or device.type != "cuda"
        or weights.num_bits > 8  # weights are stored as int16 or int32
        or not weights.symmetric
        or weight_block is None
        or not _is_dynamic_token_quant(scheme.input_activations)
    ):
        return _int8_linear_eager

    w_group_n, block_k = weight_block
    x_qmax = 2 ** (scheme.input_activations.num_bits - 1) - 1
    return partial(
        _int8_linear_triton,
        w_group_n=w_group_n,
        block_k=block_k or DEFAULT_BLOCK_K,
        x_qmax=x_qmax,
    )


def _get_weight_block(weights: QuantizationArgs) -> tuple[int, int | None] | None:
    """
    :return: number of weight rows and columns which share a scale, where None
        columns means that the scale is shared across all columns. Returns None if
        the weight strategy is not supported by the kernel
    """
    if weights.strategy in (QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL):
        return (1, None)
    if weights.strategy == QuantizationStrategy.GROUP and weights.actorder is None:
        if _is_dot_size(weights.group_size):
            return (1, weights.group_size)

    return None


def _is_dynamic_token_quant(args: QuantizationArgs | None) -> bool:
    """
    :return: True if activations are quantized to INT with a dynamic symmetric
        per-token scale, in which case they are quantized within the kernel.
        Otherwise the eager fallback is used
    """
    return (
        args is not None
        and args.type == QuantizationType.INT
        and args.num_bits in (4, 8)
        and args.strategy == QuantizationStrategy.TOKEN
        and args.dynamic is True
        and args.symmetric
        and args.scale_dtype is None
    )


def _is_dot_size(size: int | None) -> bool:
    # the K dimension of an int8 `tl.dot` tile must be at least 32
    return size is not None and size >= 32 and math.log2(size).is_integer()


def _int8_linear_eager(module: torch.nn.Linear, input: torch.Tensor) -> torch.Tensor:
    """Dequantize the weight and perform a dense linear"""
    scheme: QuantizationScheme = module.quantization_scheme

    if scheme.input_activations is not None:
        input = forward_quantize(module, input, "input", scheme.input_activations)

    weight = dequantize(
        x_q=module.weight,
        scale=module.weight_scale,
        zero_point=getattr(module, "weight_zero_point", None),
        args=scheme.weights,
        dtype=input.dtype,
    )
    output = torch.nn.functional.linear(input, weight, module.bias)

    if scheme.output_activations is not None:
        output = forward_quantize(module, output, "output", scheme.output_activations)

    return output


def _int8_linear_triton(
    module: torch.nn.Linear,
    input: torch.Tensor,
    w_group_n: int,
    block_k: int,
    x_qmax: int,
) -> torch.Tensor:
    scheme: QuantizationScheme = module.quantization_scheme
    weight = module.weight
    weight_scale = module.weight_scale
    bias = module.bias
    n, k = weight.shape
    num_k_tiles = triton.cdiv(k, block_k)

    input_2d = input.reshape(-1, k)
    m = input_2d.shape[0]
    weight_scale = _expand_scale(weight_scale, triton.cdiv(n, w_group_n), num_k_tiles)
    output = torch.empty((m, n), dtype=input.dtype, device=input.device)

    # the int8 `tl.dot` does not use the tensor cores efficiently on current
    # GPUs, so small tiles with many warps are fastest (see the benchmark in
    # the PR description)
    block_m = 16
    if m <= 16:
        block_n, num_warps, num_stages = 32, 8, 4
    elif m <= 64:
        block_n, num_warps, num_stages = 64, 8, 3
    else:
        block_n, num_warps, num_stages = 128, 8, 3
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    with torch.get_device_module(input.device).device(input.device):
        _int8_linear_kernel[grid](
            input_2d,
            weight,
            weight_scale,
            bias if bias is not None else output,  # unused if no bias
            output,
            m,
            n,
            k,
            *input_2d.stride(),
            *weight.stride(),
            *weight_scale.stride(),
            *output.stride(),
            W_GROUP_N=w_group_n,
            X_QMAX=x_qmax,
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


def _expand_scale(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """
    Expand a tensor, channel, or group scale to shape ``[rows, cols]``, where
    dimensions which share a scale have a stride of zero
    """
    if scale.ndim <= 1:
        scale = scale.reshape(1, 1)

    return scale.expand(rows, cols)


@triton.jit
def _int8_scale(amax, dtype: tl.constexpr, X_QMAX: tl.constexpr):
    """Compute a dynamic symmetric INT scale from the absolute max, rounded to
    `dtype` (mirrors `calculate_qparams` for symmetric INT quantization)"""
    scale = (amax / (X_QMAX + 0.5)).to(dtype).to(tl.float32)
    return tl.where(scale == 0.0, 1.0, scale)


@triton.jit
def _int8_linear_kernel(
    x_ptr,  # [M, K] activations
    w_ptr,  # [N, K] int8 weights
    ws_ptr,  # [cdiv(N, W_GROUP_N), cdiv(K, BLOCK_K)] weight scales
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
    stride_om,
    stride_on,
    W_GROUP_N: tl.constexpr,
    X_QMAX: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    out = (x_q * xs) @ (w_q * ws).T + bias, where x_q = round(x / xs) are the
    activations quantized to int8 with a dynamic symmetric per-row scale xs.
    Each scale is constant over the contraction dimension of a tile and is
    applied to the int32 accumulator
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    ws_ptrs = ws_ptr + (offs_n // W_GROUP_N) * stride_wsn

    # dynamic per-row activation scale from the row amax
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (kt * BLOCK_K + offs_k) < K
        x = tl.load(
            x_ptrs + kt * BLOCK_K * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        amax = tl.maximum(amax, tl.max(tl.abs(x.to(tl.float32)), axis=1))
    xs = _int8_scale(amax, x_ptr.dtype.element_ty, X_QMAX)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (kt * BLOCK_K + offs_k) < K
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        ws = tl.load(ws_ptrs, mask=n_mask, other=0.0).to(tl.float32)

        # clamp and round to int8 (round to nearest, ties to even)
        x = tl.clamp(x.to(tl.float32) / xs[:, None], -X_QMAX - 1, X_QMAX)
        x = tl.extra.cuda.libdevice.rint(x).to(tl.int8)
        acc += (
            tl.dot(x, w, out_dtype=tl.int32).to(tl.float32) * xs[:, None] * ws[None, :]
        )

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        ws_ptrs += stride_wsk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias.to(tl.float32)[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )
