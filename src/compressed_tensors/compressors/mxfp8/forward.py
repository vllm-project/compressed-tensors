# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Compressed forward pass for linear modules with MXFP8 weights.

All supported schemes are served by a single Triton kernel, `_mxfp8_linear_kernel`,
which consumes the FP8 (E4M3) weights and E8M0 scales directly via `tl.dot_scaled`.
On GPUs with FP8 tensor cores, this is a native microscaled matmul. On other GPUs,
Triton upcasts the operands to bf16.

For weight-only schemes, activations are multiplied in bf16 or fp16. For MXFP8
activations, activations are quantized to E4M3 with E8M0 group scales before the
matmul, using the same qparams as the fake-quantized forward.
"""

from functools import partial
from typing import Callable

import torch
from compressed_tensors.compressors.mx_utils import (
    compress_mx_scale,
    decompress_mx_scale,
)
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


__all__ = ["get_mxfp8_linear_forward"]


MX_GROUP_SIZE = 32
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def get_mxfp8_linear_forward(
    module: torch.nn.Module,
) -> Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None:
    """
    Resolve the forward function of a compressed linear module with MXFP8 weights.
    The Triton kernel is used when possible, otherwise the weight is dequantized
    on the fly.

    :param module: compressed module
    :return: forward function, or None if the module is not an MXFP8 linear module
    """
    scheme: QuantizationScheme = module.quantization_scheme
    weights = scheme.weights
    if not (
        isinstance(module, torch.nn.Linear)
        and weights is not None
        and weights.num_bits == 8
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
        return _mxfp8_linear_eager

    input_fp8 = _is_mxfp8(scheme.input_activations)
    return partial(_mxfp8_linear_triton, input_fp8=input_fp8)


def _is_mxfp8(args: QuantizationArgs | None) -> bool:
    """
    :return: True if activations are dynamically quantized to MXFP8, in which case
        they are quantized before the matmul. Otherwise activations are fake
        quantized in torch
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
    weight = module.weight.to(dtype)
    scale = decompress_mx_scale(module.weight_scale).to(dtype)

    weight = weight.unflatten(-1, (-1, MX_GROUP_SIZE)) * scale.unsqueeze(-1)
    return weight.flatten(-2)


def _mxfp8_linear_eager(module: torch.nn.Linear, input: torch.Tensor) -> torch.Tensor:
    """Dequantize the weight and perform a dense linear"""
    scheme: QuantizationScheme = module.quantization_scheme

    if scheme.input_activations is not None:
        input = forward_quantize(module, input, "input", scheme.input_activations)

    weight = _dequantize_weight(module, input.dtype)
    output = torch.nn.functional.linear(input, weight, module.bias)

    if scheme.output_activations is not None:
        output = forward_quantize(module, output, "output", scheme.output_activations)

    return output


def _mxfp8_linear_triton(
    module: torch.nn.Linear, input: torch.Tensor, input_fp8: bool
) -> torch.Tensor:
    scheme: QuantizationScheme = module.quantization_scheme
    weight = module.weight
    weight_scale = module.weight_scale
    bias = module.bias
    n, k = weight.shape
    output_dtype = input.dtype

    if input_fp8:
        input_2d, input_scale = _quantize_input(
            module, input.reshape(-1, k), scheme.input_activations
        )
        input_format = "e4m3"
    else:
        if scheme.input_activations is not None:
            input = forward_quantize(module, input, "input", scheme.input_activations)
        input_2d = input.reshape(-1, k)
        input_scale = weight_scale  # unused
        input_format = "fp16" if input.dtype == torch.float16 else "bf16"

    m = input_2d.shape[0]
    output = torch.empty((m, n), dtype=output_dtype, device=input.device)

    block_m = min(max(triton.next_power_of_2(m), 16), 128)
    if m <= 16:
        block_n, block_k, num_warps, num_stages = 64, 256, 4, 3
    elif input_fp8:
        block_n, block_k, num_warps, num_stages = 128, 256, 8, 2
    else:
        block_n, block_k, num_warps, num_stages = 128, 64, 8, 3
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    with torch.get_device_module(input.device).device(input.device):
        _mxfp8_linear_kernel[grid](
            input_2d,
            input_scale,
            weight,
            weight_scale,
            bias if bias is not None else output,  # unused if no bias
            output,
            m,
            n,
            k,
            *input_2d.stride(),
            *input_scale.stride(),
            *weight.stride(),
            *weight_scale.stride(),
            *output.stride(),
            INPUT_FORMAT=input_format,
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

    :return: quantized activations of shape ``[M, K]`` and E8M0 scales of shape
        ``[M, K // 32]``
    """
    scale, _ = compute_dynamic_scales_and_zp(value=input, args=args, module=module)
    # quantize manually, so that the rounding matches `fake_quantize` exactly:
    # clamp to the fp8 range, then cast
    input_q = input.unflatten(-1, (-1, args.group_size))
    input_q = (input_q / scale.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return input_q.to(torch.float8_e4m3fn).flatten(-2), compress_mx_scale(
        scale, torch.uint8
    )


@triton.jit
def _mxfp8_linear_kernel(
    x_ptr,  # [M, K] activations, fp8 if INPUT_FORMAT is e4m3
    xs_ptr,  # [M, K // 32] e8m0 activation scales, if INPUT_FORMAT is e4m3
    w_ptr,  # [N, K] fp8 weights
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
    stride_wk,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    INPUT_FORMAT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """out = x @ dequant(w).T + bias, computed by `tl.dot_scaled`"""
    INPUT_FP8: tl.constexpr = INPUT_FORMAT == "e4m3"
    SCALE_K: tl.constexpr = BLOCK_K // 32

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, SCALE_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # weights are loaded transposed, as [K, N]
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + offs_sk[None, :] * stride_xsk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    ws_ptrs = ws_ptr + offs_n[:, None] * stride_wsn + offs_sk[None, :] * stride_wsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (kt * BLOCK_K + offs_k) < K
        s_mask = (kt * SCALE_K + offs_sk) < K // 32

        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        ws = tl.load(ws_ptrs, mask=n_mask[:, None] & s_mask[None, :], other=127)

        if INPUT_FP8:
            xs = tl.load(xs_ptrs, mask=m_mask[:, None] & s_mask[None, :], other=127)
            acc = tl.dot_scaled(x, xs, "e4m3", w, ws, "e4m3", acc=acc)
            xs_ptrs += SCALE_K * stride_xsk
        else:
            x = x.to(tl.float16 if INPUT_FORMAT == "fp16" else tl.bfloat16)
            acc = tl.dot_scaled(x, None, INPUT_FORMAT, w, ws, "e4m3", acc=acc)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
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
