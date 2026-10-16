# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Compressed forward pass for linear modules with MXFP6 weights.

All supported schemes are served by a single Triton kernel, `_mxfp6_linear_kernel`,
which unpacks the packed FP6 (E2M3) weights and E8M0 scales directly within the
matmul and multiplies the dequantized weight tiles with `tl.dot`. Unlike MXFP4,
`tl.dot_scaled` cannot be used for the weights, because Triton does not support
the e2m3 element format.

For weight-only schemes, activations are multiplied in bf16 or fp16. For MXFP6
schemes with MXFP8 activations, the activations are quantized to E4M3 with E8M0
group scales, using the same qparams as `forward_quantize`, and multiplied with
`tl.dot_scaled`, with the dequantized weight operand supplied as unscaled bf16.
"""

from functools import partial
from typing import Callable

import torch
from compressed_tensors.compressors.mx_utils import (
    compress_mx_scale,
    decompress_mx_scale,
)
from compressed_tensors.compressors.mxfp6.helpers import unpack_fp6_from_uint8
from compressed_tensors.offload import get_execution_device
from compressed_tensors.quantization import (
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


__all__ = ["get_mxfp6_linear_forward"]


MX_GROUP_SIZE = 32
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def get_mxfp6_linear_forward(
    module: torch.nn.Module,
) -> Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None:
    """
    Resolve the forward function of a compressed linear module with MXFP6 weights.
    The Triton kernel is used when possible, otherwise the weight is dequantized
    on the fly.

    :param module: compressed module
    :return: forward function, or None if the module is not an MXFP6 linear module
    """
    scheme: QuantizationScheme = module.quantization_scheme
    weights = scheme.weights
    if not (
        isinstance(module, torch.nn.Linear)
        and weights is not None
        and weights.num_bits == 6
        and weights.type == QuantizationType.FLOAT
        and should_generate_mx_scales(weights)
    ):
        return None

    device = get_execution_device(module)
    if (
        ENFORCE_EAGER
        or not HAS_TRITON
        or not isinstance(device, torch.device)
        or device.type != "cuda"
        or module.in_features % MX_GROUP_SIZE != 0
    ):
        return _mxfp6_linear_eager

    input_fp8 = _is_mxfp8(scheme.input_activations)
    return partial(_mxfp6_linear_triton, input_fp8=input_fp8)


def _is_mxfp8(args: QuantizationArgs | None) -> bool:
    """
    :return: True if activations are dynamically quantized to MXFP8, in which case
        they are quantized to E4M3 with E8M0 scales before the matmul. Otherwise
        activations are fake quantized in torch
    """
    return (
        args is not None
        and args.type == QuantizationType.FLOAT
        and args.num_bits == 8
        and args.strategy == QuantizationStrategy.GROUP
        and args.dynamic is True
        and args.symmetric
        and should_generate_mx_scales(args)
    )


def _dequantize_weight(module: torch.nn.Linear, dtype: torch.dtype) -> torch.Tensor:
    weight_packed = module.weight_packed
    n, k = weight_packed.shape[0], weight_packed.shape[1] * 4 // 3
    weight = unpack_fp6_from_uint8(weight_packed, n, k, dtype=dtype)
    scale = decompress_mx_scale(module.weight_scale).to(dtype)

    weight = weight.unflatten(-1, (-1, MX_GROUP_SIZE)) * scale.unsqueeze(-1)
    return weight.flatten(-2)


def _mxfp6_linear_eager(module: torch.nn.Linear, input: torch.Tensor) -> torch.Tensor:
    """Dequantize the weight and perform a dense linear"""
    scheme: QuantizationScheme = module.quantization_scheme

    if scheme.input_activations is not None:
        input = forward_quantize(module, input, "input", scheme.input_activations)

    weight = _dequantize_weight(module, input.dtype)
    output = torch.nn.functional.linear(input, weight, module.bias)

    if scheme.output_activations is not None:
        output = forward_quantize(module, output, "output", scheme.output_activations)

    return output


def _mxfp6_linear_triton(
    module: torch.nn.Linear, input: torch.Tensor, input_fp8: bool
) -> torch.Tensor:
    scheme: QuantizationScheme = module.quantization_scheme
    weight_packed = module.weight_packed
    weight_scale = module.weight_scale
    bias = module.bias
    n, k = weight_packed.shape[0], weight_packed.shape[1] * 4 // 3
    output_dtype = input.dtype

    if input_fp8:
        input_2d, input_scale = _quantize_input(
            module, input.reshape(-1, k), scheme.input_activations
        )
    else:
        if scheme.input_activations is not None:
            input = forward_quantize(module, input, "input", scheme.input_activations)
        input_2d = input.reshape(-1, k)
        input_scale = weight_scale  # unused

    m = input_2d.shape[0]
    output = torch.empty((m, n), dtype=output_dtype, device=input.device)

    block_m = min(max(triton.next_power_of_2(m), 16), 256)
    if m <= 16:
        block_n, block_k, num_warps, num_stages = 64, 256, 8, 3
    else:
        block_n, block_k, num_warps, num_stages = 128, 128, 8, 2
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    with torch.get_device_module(input.device).device(input.device):
        _mxfp6_linear_kernel[grid](
            input_2d,
            input_scale,
            weight_packed,
            weight_scale,
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
            INPUT_FP8=input_fp8,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activations to MXFP8, using the same qparams as `forward_quantize`

    :return: E4M3 activations of shape ``[M, K]`` and E8M0 scales of shape
        ``[M, K // 32]``
    """
    scale, _ = compute_dynamic_scales_and_zp(value=input, args=args, module=module)
    # the scale is a power of 2, so the E8M0 exponent is recovered exactly
    xs = compress_mx_scale(scale, torch.uint8)
    grouped = input.float().unflatten(-1, (-1, args.group_size))
    q = (grouped / scale.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return q.to(torch.float8_e4m3fn).flatten(-2), xs


@triton.jit
def _decode_e2m3(code):
    """Decode a 6-bit E2M3 code to a float value, with the sign in bit 5"""
    negative = (code & 0x20) != 0
    magnitude = code & 0x1F
    exponent = magnitude >> 3
    mantissa = magnitude & 7
    value = tl.where(
        exponent == 0,
        mantissa.to(tl.float32) * 0.125,
        (1.0 + mantissa.to(tl.float32) / 8.0) * tl.exp2(exponent.to(tl.float32) - 1.0),
    )
    return tl.where(negative, -value, value)


@triton.jit
def _mxfp6_linear_kernel(
    x_ptr,  # [M, K] activations, or [M, K] e4m3 if INPUT_FP8
    xs_ptr,  # [M, K // 32] e8m0 activation scales, if INPUT_FP8
    w_ptr,  # [N, 3K // 4] packed fp6 weights
    ws_ptr,  # [N, K // 32] e8m0 weight scales
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
    stride_wp,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    INPUT_FP8: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """out = x @ dequant(w).T + bias, with the fp6 weights unpacked in-kernel"""
    PACKED_K: tl.constexpr = BLOCK_K // 4  # packed bytes per tile
    SCALE_K: tl.constexpr = BLOCK_K // 32  # scale groups per tile

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_g = tl.arange(0, PACKED_K)  # index of each group of four codes
    offs_sk = tl.arange(0, SCALE_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + offs_sk[None, :] * stride_xsk
    # each group of four codes is packed little-endian into three bytes
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_g[None, :] * 3 * stride_wp
    ws_ptrs = ws_ptr + offs_n[:, None] * stride_wsn + offs_sk[None, :] * stride_wsk

    # the weight tile is dequantized as [BLOCK_N, BLOCK_K], so the activations
    # are the right-hand operand and the accumulator is transposed
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (kt * BLOCK_K + offs_k) < K
        g_mask = (kt * PACKED_K + offs_g) < K // 4
        s_mask = (kt * SCALE_K + offs_sk) < K // 32

        # assemble the 24-bit little-endian word of each group of four codes
        w_mask = n_mask[:, None] & g_mask[None, :]
        b0 = tl.load(w_ptrs, mask=w_mask, other=0).to(tl.int32)
        b1 = tl.load(w_ptrs + stride_wp, mask=w_mask, other=0).to(tl.int32)
        b2 = tl.load(w_ptrs + 2 * stride_wp, mask=w_mask, other=0).to(tl.int32)
        word = b0 | b1 << 8 | b2 << 16

        # decode the E8M0 scale of each group of 32, then expand it to the
        # eight groups of four codes which share it
        ws = tl.load(ws_ptrs, mask=n_mask[:, None] & s_mask[None, :], other=127)
        ws = tl.exp2(ws.to(tl.float32) - 127.0)
        ws = tl.reshape(
            tl.broadcast_to(ws[:, :, None], (BLOCK_N, SCALE_K, 8)),
            (BLOCK_N, PACKED_K),
        )

        if INPUT_FP8:
            w_dtype = tl.bfloat16
        else:
            w_dtype = x_ptr.dtype.element_ty
        # dequantize each of the four codes of a group, then restore the
        # [BLOCK_N, BLOCK_K] layout by interleaving along the last dimension
        w0 = (_decode_e2m3(word & 0x3F) * ws).to(w_dtype)
        w1 = (_decode_e2m3((word >> 6) & 0x3F) * ws).to(w_dtype)
        w2 = (_decode_e2m3((word >> 12) & 0x3F) * ws).to(w_dtype)
        w3 = (_decode_e2m3((word >> 18) & 0x3F) * ws).to(w_dtype)
        w = tl.interleave(tl.interleave(w0, w2), tl.interleave(w1, w3))

        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        x_t = tl.trans(x)
        if INPUT_FP8:
            xs = tl.load(xs_ptrs, mask=m_mask[:, None] & s_mask[None, :], other=127)
            acc = tl.dot_scaled(w, None, "bf16", x_t, xs, "e4m3", acc=acc)
            xs_ptrs += SCALE_K * stride_xsk
        else:
            acc = tl.dot(w, x_t, acc=acc)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += 3 * PACKED_K * stride_wp
        ws_ptrs += SCALE_K * stride_wsk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias.to(tl.float32)[:, None]

    out_ptrs = out_ptr + offs_n[:, None] * stride_on + offs_m[None, :] * stride_om
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=n_mask[:, None] & m_mask[None, :],
    )
