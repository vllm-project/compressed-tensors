# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Compressed forward pass for linear modules with NVFP4 weights.

All supported schemes are served by a single Triton kernel, `_nvfp4_linear_kernel`,
which unpacks the packed FP4 (E2M1) weights and decodes the FP8 (E4M3) group
scales in-kernel, then multiplies them with the activations via `tl.dot`.
`tl.dot_scaled` cannot be used here, because it only supports E8M0 scale formats
while NVFP4 group scales are E4M3 (combined with a per-tensor global scale).

For weight-only schemes, activations are multiplied in bf16 or fp16. For NVFP4
activations, activations are quantized and packed with the fused
`quantize_and_pack_fp4` kernel before the matmul.
"""

from functools import partial
from typing import Callable

import torch
from compressed_tensors.compressors.nvfp4.helpers import (
    quantize_and_pack_fp4,
    unpack_fp4_from_uint8,
)
from compressed_tensors.offload import get_execution_device
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.quantization.utils import (
    compute_dynamic_scales_and_zp,
    should_generate_mx_scales,
)
from compressed_tensors.utils.impl_backend import ENFORCE_EAGER
from compressed_tensors.utils.triton import HAS_TRITON, tl, triton


__all__ = ["get_nvfp4_linear_forward"]


NVFP4_GROUP_SIZE = 16


def get_nvfp4_linear_forward(
    module: torch.nn.Module,
) -> Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None:
    """
    Resolve the forward function of a compressed linear module with NVFP4 weights.
    The Triton kernel is used when possible, otherwise the weight is dequantized
    on the fly.

    :param module: compressed module
    :return: forward function, or None if the module is not an NVFP4 linear module
        (excluding MXFP4 and MXFP6, which use other scale formats)
    """
    scheme: QuantizationScheme = module.quantization_scheme
    weights = scheme.weights
    if not (
        isinstance(module, torch.nn.Linear)
        and weights is not None
        and weights.num_bits == 4
        and weights.type == QuantizationType.FLOAT
        and weights.symmetric
        and weights.group_size == NVFP4_GROUP_SIZE
        and weights.scale_dtype in (None, torch.float8_e4m3fn)
        and not should_generate_mx_scales(weights)
    ):
        return None

    device = get_execution_device(module)
    if (
        ENFORCE_EAGER
        or not HAS_TRITON
        or not isinstance(device, torch.device)
        or device.type != "cuda"
        or module.in_features % NVFP4_GROUP_SIZE != 0
    ):
        return _nvfp4_linear_eager

    input_fp4 = _is_nvfp4(scheme.input_activations)
    has_w_global = getattr(module, "weight_global_scale", None) is not None
    return partial(_nvfp4_linear_triton, input_fp4=input_fp4, has_w_global=has_w_global)


def _is_nvfp4(args: QuantizationArgs | None) -> bool:
    """
    :return: True if activations are dynamically quantized to NVFP4, in which case
        they are quantized and packed before the matmul. Otherwise activations are
        fake quantized in torch
    """
    return (
        args is not None
        and args.type == QuantizationType.FLOAT
        and args.num_bits == 4
        and args.symmetric
        and args.strategy
        in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP)
        and args.group_size == NVFP4_GROUP_SIZE
        and args.dynamic in (True, DynamicType.LOCAL)
        and args.scale_dtype in (None, torch.float8_e4m3fn)
    )


def _dequantize_weight(module: torch.nn.Linear, dtype: torch.dtype) -> torch.Tensor:
    weight_packed = module.weight_packed
    n, k = weight_packed.shape[0], weight_packed.shape[1] * 2
    weight = unpack_fp4_from_uint8(weight_packed, n, k, dtype=dtype)
    scale = module.weight_scale.to(dtype)

    global_scale = getattr(module, "weight_global_scale", None)
    if global_scale is not None:
        scale = scale / global_scale

    weight = weight.unflatten(-1, (-1, NVFP4_GROUP_SIZE)) * scale.unsqueeze(-1)
    return weight.flatten(-2).to(dtype)


def _nvfp4_linear_eager(module: torch.nn.Linear, input: torch.Tensor) -> torch.Tensor:
    """Dequantize the weight and perform a dense linear"""
    scheme: QuantizationScheme = module.quantization_scheme

    if scheme.input_activations is not None:
        input = forward_quantize(module, input, "input", scheme.input_activations)

    weight = _dequantize_weight(module, input.dtype)
    output = torch.nn.functional.linear(input, weight, module.bias)

    if scheme.output_activations is not None:
        output = forward_quantize(module, output, "output", scheme.output_activations)

    return output


def _nvfp4_linear_triton(
    module: torch.nn.Linear,
    input: torch.Tensor,
    input_fp4: bool,
    has_w_global: bool,
) -> torch.Tensor:
    scheme: QuantizationScheme = module.quantization_scheme
    weight_packed = module.weight_packed
    weight_scale = module.weight_scale
    weight_global_scale = getattr(module, "weight_global_scale", None)
    bias = module.bias
    n, k = weight_packed.shape[0], weight_packed.shape[1] * 2
    output_dtype = input.dtype

    if input_fp4:
        input_2d, input_scale, input_global_scale = _quantize_input(
            module, input.reshape(-1, k), scheme.input_activations
        )
    else:
        if scheme.input_activations is not None:
            input = forward_quantize(module, input, "input", scheme.input_activations)
        input_2d = input.reshape(-1, k)
        input_scale = weight_scale  # unused
        input_global_scale = None

    m = input_2d.shape[0]
    output = torch.empty((m, n), dtype=output_dtype, device=input.device)

    # the unpacked operands are replicated over the tiles of the other operand,
    # so weight-only schemes prefer tall tiles (fewer weight unpacks) while NVFP4
    # activations prefer wide tiles (fewer activation unpacks)
    block_m = min(max(triton.next_power_of_2(m), 16), 128 if input_fp4 else 256)
    if m <= 16:
        block_n, block_k = (32, 256) if input_fp4 else (16, 128)
        num_warps, num_stages = 4, 3
    elif m <= 128:
        block_n, block_k, num_warps, num_stages = 64, 64, 4, 3
    elif input_fp4:
        block_n, block_k, num_warps, num_stages = 128, 64, 8, 3
    else:
        block_n, block_k, num_warps, num_stages = 64, 64, 8, 3
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    with torch.get_device_module(input.device).device(input.device):
        _nvfp4_linear_kernel[grid](
            input_2d,
            input_scale,
            input_global_scale
            if input_global_scale is not None
            else input_2d,  # unused if no global scale
            weight_packed,
            weight_scale,
            weight_global_scale
            if weight_global_scale is not None
            else weight_packed,  # unused if no global scale
            bias if bias is not None else output,  # unused if no bias
            output,
            m,
            n,
            k,
            *input_2d.stride(),
            *input_scale.stride(),
            *weight_packed.stride(),
            *weight_scale.stride(),
            *output.stride(),
            INPUT_FORMAT="fp16" if output_dtype == torch.float16 else "bf16",
            INPUT_FP4=input_fp4,
            HAS_X_GLOBAL=input_global_scale is not None,
            HAS_W_GLOBAL=has_w_global,
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


def _quantize_input(
    module: torch.nn.Linear, input: torch.Tensor, args: QuantizationArgs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Quantize and pack activations to NVFP4, using the same qparams as
    `forward_quantize`

    :return: packed activations of shape ``[M, K // 2]``, group scales of shape
        ``[M, K // 16]``, and the global scale (or None if there is none)
    """
    global_scale = getattr(module, "input_global_scale", None)
    scale, _ = compute_dynamic_scales_and_zp(
        value=input, args=args, module=module, global_scale=global_scale
    )
    input_packed = quantize_and_pack_fp4(
        input, scale, global_scale=global_scale, group_size=args.group_size
    )
    return input_packed, scale, global_scale


@triton.jit
def _decode_fp4(c):
    """
    Decode a tile of E2M1 codes (nibbles, 0-15) to fp16, by placing the code's
    exponent and mantissa bits into an fp16 bit pattern
    """
    mag = (c & 0x7).to(tl.uint16)
    bits = (((mag >> 1) + 14) << 10) | ((mag & 1) << 9)
    bits = tl.where(mag == 1, 14 << 10, bits)  # subnormal code 0.5
    bits = tl.where(mag == 0, 0, bits)
    bits = bits | ((c & 8).to(tl.uint16) << 12)
    return bits.to(tl.float16, bitcast=True)


@triton.jit
def _expand_scale(s, R: tl.constexpr, C: tl.constexpr, GROUP: tl.constexpr):
    """Expand a ``[R, C]`` group scale tile to ``[R, C * GROUP]``"""
    return tl.reshape(tl.broadcast_to(s[:, :, None], (R, C, GROUP)), (R, C * GROUP))


@triton.jit
def _nvfp4_linear_kernel(
    x_ptr,  # [M, K] activations, or [M, K // 2] packed fp4 if INPUT_FP4
    xs_ptr,  # [M, K // 16] activation group scales, if INPUT_FP4
    xgs_ptr,  # [1] activation global scale, if INPUT_FP4 and HAS_X_GLOBAL
    w_ptr,  # [N, K // 2] packed fp4 weights
    ws_ptr,  # [N, K // 16] fp8 weight group scales
    wgs_ptr,  # [1] weight global scale, if HAS_W_GLOBAL
    bias_ptr,  # [N]
    out_ptr,  # [M, N]
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_xsm,
    stride_xsk,
    stride_wn,
    stride_wk,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    INPUT_FORMAT: tl.constexpr,
    INPUT_FP4: tl.constexpr,
    HAS_X_GLOBAL: tl.constexpr,
    HAS_W_GLOBAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    out = x @ dequant(w).T + bias, where the packed fp4 weights are unpacked and
    scaled by ``ws / wgs`` within the kernel. If INPUT_FP4, the packed fp4
    activations are unpacked and scaled by ``xs / xgs`` as well
    """
    GROUP: tl.constexpr = 16
    PACKED_K: tl.constexpr = BLOCK_K // 2
    SCALE_K: tl.constexpr = BLOCK_K // GROUP
    X_K: tl.constexpr = PACKED_K if INPUT_FP4 else BLOCK_K

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_xk = tl.arange(0, X_K)
    offs_wk = tl.arange(0, PACKED_K)
    offs_sk = tl.arange(0, SCALE_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_xk[None, :] * stride_xk
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + offs_sk[None, :] * stride_xsk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_wk[None, :] * stride_wk
    ws_ptrs = ws_ptr + offs_n[:, None] * stride_wsn + offs_sk[None, :] * stride_wsk

    dot_dtype = tl.float16 if INPUT_FORMAT == "fp16" else tl.bfloat16
    if HAS_W_GLOBAL:
        wgs = tl.load(wgs_ptr).to(tl.float32)
    if INPUT_FP4 and HAS_X_GLOBAL:
        xgs = tl.load(xgs_ptr).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        x_mask = (kt * X_K + offs_xk) < (K // 2 if INPUT_FP4 else K)
        w_mask = (kt * PACKED_K + offs_wk) < K // 2
        s_mask = (kt * SCALE_K + offs_sk) < K // GROUP

        # unpack the weight tile and scale it by its group scales: [BLOCK_N, BLOCK_K]
        w = tl.load(w_ptrs, mask=n_mask[:, None] & w_mask[None, :], other=0)
        w = tl.interleave(_decode_fp4(w & 0xF), _decode_fp4(w >> 4))
        ws = tl.load(ws_ptrs, mask=n_mask[:, None] & s_mask[None, :], other=0.0).to(
            tl.float32
        )
        if HAS_W_GLOBAL:
            ws = ws / wgs
        ws = _expand_scale(ws, BLOCK_N, SCALE_K, GROUP)
        w = (w.to(tl.float32) * ws).to(dot_dtype)

        if INPUT_FP4:
            # unpack the activation tile and scale it by its group scales
            x = tl.load(x_ptrs, mask=m_mask[:, None] & x_mask[None, :], other=0)
            x = tl.interleave(_decode_fp4(x & 0xF), _decode_fp4(x >> 4))
            xs = tl.load(xs_ptrs, mask=m_mask[:, None] & s_mask[None, :], other=0.0).to(
                tl.float32
            )
            if HAS_X_GLOBAL:
                xs = xs / xgs
            xs = _expand_scale(xs, BLOCK_M, SCALE_K, GROUP)
            x = (x.to(tl.float32) * xs).to(dot_dtype)
        else:
            x = tl.load(x_ptrs, mask=m_mask[:, None] & x_mask[None, :], other=0.0)
            x = x.to(dot_dtype)

        acc = tl.dot(x, tl.trans(w), acc=acc)

        x_ptrs += X_K * stride_xk
        xs_ptrs += SCALE_K * stride_xsk
        w_ptrs += PACKED_K * stride_wk
        ws_ptrs += SCALE_K * stride_wsk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias.to(tl.float32)[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )
