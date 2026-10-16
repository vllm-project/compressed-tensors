# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Compressed forward pass for linear modules with FP8 (E4M3) weights.

All supported schemes are served by a single Triton kernel, `_fp8_linear_kernel`.
Weight scales are indexed by ``[n // W_GROUP_N, k // BLOCK_K]``, with zero strides
along dimensions which share a scale (tensor, channel, group, and block strategies),
so that the weight scale of each tile is a vector over output columns which is
applied to the fp32 accumulator.

Activations which are quantized to FP8 with a scale that is constant within each
``[1, BLOCK_K]`` tile are quantized within the kernel and multiplied with a native
FP8 matmul. Dynamic token and group scales are computed within the kernel, while
tensor scales are passed to the kernel. Otherwise, activations are fake quantized
in torch, and the weights are upcast to the activation dtype within the kernel.
"""

import math
from enum import IntEnum
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
from compressed_tensors.quantization.utils import (
    compute_dynamic_scales_and_zp,
    should_generate_mx_scales,
)
from compressed_tensors.utils.impl_backend import ENFORCE_EAGER
from compressed_tensors.utils.triton import HAS_TRITON, tl, triton


__all__ = ["get_fp8_linear_forward"]


# contraction tile size, when not constrained by the quantization group size
DEFAULT_BLOCK_K = 128
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


class InputQuant(IntEnum):
    """How activations are quantized by `_fp8_linear_kernel`"""

    NONE = 0  # activations are multiplied in their original dtype
    TENSOR = 1  # quantized with a tensor scale passed to the kernel
    GROUP = 2  # quantized with a dynamic scale per row of each [1, BLOCK_K] tile
    TOKEN = 3  # quantized with a dynamic scale per row


def get_fp8_linear_forward(
    module: torch.nn.Module,
) -> Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None:
    """
    Resolve the forward function of a compressed linear module with FP8 weights.
    The Triton kernel is used when possible, otherwise the weight is dequantized
    on the fly.

    :param module: compressed module
    :return: forward function, or None if the module is not an FP8 linear module
        (excluding MXFP8)
    """
    scheme: QuantizationScheme = module.quantization_scheme
    weights = scheme.weights
    if not (
        isinstance(module, torch.nn.Linear)
        and weights is not None
        and weights.type == QuantizationType.FLOAT
        and weights.num_bits == 8
        and not should_generate_mx_scales(weights)
    ):
        return None

    weight_block = _get_weight_block(weights)
    device = get_execution_device(module)
    if (
        ENFORCE_EAGER
        or not HAS_TRITON
        or not isinstance(device, torch.device)
        or device.type not in ("cuda", "xpu")
        or not weights.symmetric
        or weight_block is None
    ):
        return _fp8_linear_eager

    w_group_n, block_k = weight_block
    input_quant = _get_input_quant(scheme.input_activations, block_k, device)
    if input_quant == InputQuant.GROUP:
        block_k = scheme.input_activations.group_size

    return partial(
        _fp8_linear_triton,
        w_group_n=w_group_n,
        block_k=block_k or DEFAULT_BLOCK_K,
        input_quant=input_quant,
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
    if weights.strategy == QuantizationStrategy.BLOCK:
        block_n, block_k = weights.block_structure
        if _is_dot_size(block_k):
            return (block_n, block_k)

    return None


def _get_input_quant(
    args: QuantizationArgs | None, block_k: int | None, device: torch.device
) -> InputQuant:
    """
    :return: how activations can be quantized within the kernel. Activations which
        cannot be quantized within the kernel are fake quantized in torch
    """
    if (
        args is None
        or args.type != QuantizationType.FLOAT
        or args.num_bits != 8
        or not args.symmetric
        or args.scale_dtype is not None
        or not _supports_fp8_dot(device)
    ):
        return InputQuant.NONE

    if args.strategy == QuantizationStrategy.TENSOR:
        return InputQuant.TENSOR
    if args.strategy == QuantizationStrategy.TOKEN and args.dynamic is True:
        return InputQuant.TOKEN
    if (
        args.strategy == QuantizationStrategy.GROUP
        and args.dynamic is True
        and _is_dot_size(args.group_size)
        and block_k in (None, args.group_size)
    ):
        return InputQuant.GROUP

    return InputQuant.NONE


def _is_dot_size(size: int | None) -> bool:
    return size is not None and size >= 16 and math.log2(size).is_integer()


def _supports_fp8_dot(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    return torch.get_device_module(device).get_device_capability(device) >= (8, 9)


def _fp8_linear_eager(module: torch.nn.Linear, input: torch.Tensor) -> torch.Tensor:
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


def _fp8_linear_triton(
    module: torch.nn.Linear,
    input: torch.Tensor,
    w_group_n: int,
    block_k: int,
    input_quant: InputQuant,
) -> torch.Tensor:
    scheme: QuantizationScheme = module.quantization_scheme
    weight = module.weight
    weight_scale = module.weight_scale
    bias = module.bias
    n, k = weight.shape
    num_k_tiles = triton.cdiv(k, block_k)

    if input_quant == InputQuant.TENSOR:
        input_scale = _get_tensor_scale(module, input, scheme.input_activations)
    else:
        if input_quant == InputQuant.NONE and scheme.input_activations is not None:
            input = forward_quantize(module, input, "input", scheme.input_activations)
        input_scale = weight_scale  # unused

    input_2d = input.reshape(-1, k)
    m = input_2d.shape[0]
    weight_scale = _expand_scale(weight_scale, triton.cdiv(n, w_group_n), num_k_tiles)
    output = torch.empty((m, n), dtype=input.dtype, device=input.device)

    # wider tiles amortize the cost of quantizing activations within the kernel
    block_m = min(max(triton.next_power_of_2(m), 16), 64)
    if m <= 16:
        block_n, num_warps = 64, 4
    elif block_m >= 64 and input_quant != InputQuant.NONE:
        block_n, num_warps = 256, 8
    else:
        block_n, num_warps = 128, 4
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    with torch.get_device_module(input.device).device(input.device):
        _fp8_linear_kernel[grid](
            input_2d,
            weight,
            input_scale,
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
            INPUT_QUANT=input_quant.value,
            FP8_MAX=FP8_E4M3_MAX,
            HAS_BIAS=bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=3,
        )

    output = output.reshape(*input.shape[:-1], n)
    if scheme.output_activations is not None:
        output = forward_quantize(module, output, "output", scheme.output_activations)

    return output


def _get_tensor_scale(
    module: torch.nn.Linear, input: torch.Tensor, args: QuantizationArgs
) -> torch.Tensor:
    """Get the static or dynamic tensor scale of the activations"""
    if args.dynamic:
        scale, _ = compute_dynamic_scales_and_zp(value=input, args=args, module=module)
        return scale
    else:
        return module.input_scale


def _expand_scale(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """
    Expand a tensor, channel, group, or block scale to shape ``[rows, cols]``,
    where dimensions which share a scale have a stride of zero
    """
    if scale.ndim <= 1:
        scale = scale.reshape(1, 1)

    return scale.expand(rows, cols)


@triton.jit
def _fp8_scale(amax, dtype: tl.constexpr, FP8_MAX: tl.constexpr):
    """Compute a dynamic scale from the absolute max, rounded to `dtype`"""
    scale = (amax / FP8_MAX).to(dtype).to(tl.float32)
    return tl.where(scale == 0.0, 1.0, scale)


@triton.jit
def _fp8_linear_kernel(
    x_ptr,  # [M, K] activations
    w_ptr,  # [N, K] fp8 weights
    xs_ptr,  # [1] activation tensor scale, if INPUT_QUANT == TENSOR
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
    INPUT_QUANT: tl.constexpr,
    FP8_MAX: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    out = (x * xs) @ (w * ws).T + bias, where each scale is constant over the
    contraction dimension of a tile and is therefore applied to the accumulator
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

    # activation scales which are constant over the contraction dimension
    if INPUT_QUANT == 1:  # TENSOR
        xs = tl.load(xs_ptr).to(tl.float32) + tl.zeros((BLOCK_M,), dtype=tl.float32)
    elif INPUT_QUANT == 3:  # TOKEN
        amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for kt in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = (kt * BLOCK_K + offs_k) < K
            x = tl.load(
                x_ptrs + kt * BLOCK_K * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            amax = tl.maximum(amax, tl.max(tl.abs(x.to(tl.float32)), axis=1))
        xs = _fp8_scale(amax, x_ptr.dtype.element_ty, FP8_MAX)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (kt * BLOCK_K + offs_k) < K
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        ws = tl.load(ws_ptrs, mask=n_mask, other=0.0).to(tl.float32)

        if INPUT_QUANT == 0:  # NONE
            acc += tl.dot(x, w.to(x.dtype)) * ws[None, :]
        else:
            x = x.to(tl.float32)
            if INPUT_QUANT == 2:  # GROUP
                xs = _fp8_scale(
                    tl.max(tl.abs(x), axis=1), x_ptr.dtype.element_ty, FP8_MAX
                )
            x = tl.clamp(x / xs[:, None], -FP8_MAX, FP8_MAX).to(w.dtype)
            acc += tl.dot(x, w) * xs[:, None] * ws[None, :]

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
