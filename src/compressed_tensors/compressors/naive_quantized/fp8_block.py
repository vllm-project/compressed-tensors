# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Backend implementations for the block-quantized FP8 linear forward pass.

Three implementations back ``FloatQuantizationCompressor.compressed_forward`` for
block-quantized FP8 (deepseekv3-style) weights, dispatched by ``ImplBackend`` in
priority order:

1. ``fp8`` (GPUs with fp8 tensor cores, i.e. compute capability >= 8.9):
   activations are quantized to ``float8_e4m3fn`` on the fly with per-token,
   per-K-group scales, and a real fp8 ``tl.dot`` runs on the tensor cores. The
   combined per-block dequant scale (``activation_scale * weight_scale``) is
   applied per K-block during fp32 accumulation (the DeepSeek-V3 W8A8 recipe).
2. ``emulation`` (any Triton GPU): a fused dequantize-and-matmul Triton kernel.
   Weights are descaled inline (each 2D block shares one scale) and cast to bf16,
   then multiplied -- the matmul happens in higher precision (emulation).
3. eager fallback (the ``compressed_forward`` entrypoint on the compressor):
   ``forward_quantize`` for activations, ``dequantize_fp8_block_weight`` for the
   weight, then a dense ``F.linear``.

Weights are stored as ``float8_e4m3fn`` with a single float scale per
``block_structure`` (e.g. ``128 x 128``) block, a layout consumable directly by
vLLM's block-scaled FP8 kernels.
"""

import torch
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.quantization.utils import maybe_pad_tensor_for_block_quant
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import HAS_TRITON, tl, triton


__all__ = [
    "dequantize_fp8_block_weight",
    "fp8_block_forward_emulation",
    "fp8_block_forward_fp8",
]

FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


# ---------------------------------------------------------------------------
# Backend selection requirements
# ---------------------------------------------------------------------------


def _emulation_req(module: torch.nn.Module, input: torch.Tensor) -> bool:
    return HAS_TRITON and (input.is_cuda or input.is_xpu)


def _fp8_tensorcore_req(module: torch.nn.Module, input: torch.Tensor) -> bool:
    """
    Usable only on CUDA GPUs that expose fp8 tensor cores (Ada ``sm_89`` and
    Hopper ``sm_90`` onwards) for an activation-quantized (w8a8) block scheme
    whose block dimensions are ``tl.dot``-compatible (multiples of 16).
    """
    if not (HAS_TRITON and input.is_cuda):
        return False

    scheme = getattr(module, "quantization_scheme", None)
    if scheme is None or scheme.input_activations is None or scheme.weights is None:
        return False

    block_structure = scheme.weights.block_structure
    if block_structure is None:
        return False
    block_height, block_width = block_structure
    if block_height % 16 != 0 or block_width % 16 != 0:
        return False

    return torch.get_device_module().get_device_capability(input.device) >= (8, 9)


# ---------------------------------------------------------------------------
# Shared weight dequantization (eager)
# ---------------------------------------------------------------------------


def dequantize_fp8_block_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_structure: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequantize block-quantized FP8 weights into a dense float tensor.

    Each ``block_height x block_width`` block of the weight shares a single scale.
    The weight is padded up to block-divisible dimensions, descaled block-wise,
    and truncated back to its original shape.

    :param weight: fp8 weights ``[N, K]`` (float8_e4m3fn)
    :param weight_scale: per-block scales ``[ceil(N / bh), ceil(K / bw)]``
    :param block_structure: ``(block_height, block_width)`` block shape
    :param dtype: output dtype
    :return: dense weight ``[N, K]``
    """
    n, k = weight.shape
    block_height, block_width = block_structure

    w = maybe_pad_tensor_for_block_quant(weight.to(dtype), (block_height, block_width))
    padded_n, padded_k = w.shape
    num_row_blocks = padded_n // block_height
    num_col_blocks = padded_k // block_width

    # reshape into blocks, broadcast the per-block scale, then restore
    w = w.reshape(num_row_blocks, block_height, num_col_blocks, block_width)
    scale = weight_scale.to(dtype).reshape(num_row_blocks, 1, num_col_blocks, 1)
    w = (w * scale).reshape(padded_n, padded_k)

    return w[:n, :k].contiguous().to(dtype)


# ---------------------------------------------------------------------------
# Emulation backend: fused dequant + bf16 matmul
# ---------------------------------------------------------------------------


@triton.jit
def _fp8_block_emulation_kernel(
    x_ptr,  # [M, K] activations (already fake-quantized)
    w_ptr,  # [N, K] fp8 weights
    s_ptr,  # [NRB, NCB] per-block float scales
    out_ptr,  # [M, N] output
    M,
    N,
    K,
    NRB,
    NCB,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sc,
    stride_om,
    stride_on,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused dequantize-and-matmul: out = x @ dequant(w).T

    The weight tile is loaded transposed to ``[BLOCK_K, BLOCK_N]``, the matching
    per-block scale is gathered per element (block row = ``n // BLOCK_H``, block
    col = ``k // BLOCK_W``), and the descaled weight is cast to bf16 and multiplied
    in bf16 -- an emulation of the fp8 matmul, not a real fp8 tensor-core op.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k0 = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = kt * BLOCK_K + offs_k0
        k_mask = offs_k < K

        # load weight transposed to [BLOCK_K, BLOCK_N]
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = k_mask[:, None] & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # gather the per-block scale for each (k, n) element
        s_row = offs_n // BLOCK_H  # scale row indexed by output col n
        s_col = offs_k // BLOCK_W  # scale col indexed by contraction dim k
        s_ptrs = s_ptr + s_col[:, None] * stride_sc + s_row[None, :] * stride_sn
        s_mask = (s_col[:, None] < NCB) & (s_row[None, :] < NRB)
        s = tl.load(s_ptrs, mask=s_mask, other=0.0)

        w = (w * s).to(tl.bfloat16)  # [BLOCK_K, BLOCK_N]

        # load activations [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & k_mask[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.bfloat16)

        acc += tl.dot(x, w)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def _fp8_block_emulation_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_structure: tuple[int, int],
    bias: torch.Tensor | None,
) -> torch.Tensor:
    n, k = weight.shape
    m = x.shape[0]
    block_height, block_width = block_structure
    num_row_blocks, num_col_blocks = weight_scale.shape

    scale = weight_scale.to(torch.float32).contiguous()
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 64
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    with torch.get_device_module().device(x.device):
        _fp8_block_emulation_kernel[grid](
            x,
            weight,
            scale,
            out,
            m,
            n,
            k,
            num_row_blocks,
            num_col_blocks,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            scale.stride(0),
            scale.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_H=block_height,
            BLOCK_W=block_width,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

    out = out.to(x.dtype)
    if bias is not None:
        out += bias
    return out


@ImplBackend.register("fp8_block_forward", _emulation_req, 1)
def fp8_block_forward_emulation(
    module: torch.nn.Module, input: torch.Tensor
) -> torch.Tensor:
    """Triton emulation backend: fake-quantize activations, fused dequant matmul."""
    scheme = module.quantization_scheme
    if scheme.input_activations is not None:
        input = forward_quantize(module, input, "input", scheme.input_activations)

    original_shape = input.shape
    input_2d = input.reshape(-1, original_shape[-1])

    output = _fp8_block_emulation_matmul(
        input_2d,
        module.weight,
        module.weight_scale,
        tuple(scheme.weights.block_structure),
        getattr(module, "bias", None),
    )
    return output.reshape(*original_shape[:-1], output.shape[-1])


# ---------------------------------------------------------------------------
# FP8 tensor-core backend: dynamic-quant activations + real fp8 matmul
# ---------------------------------------------------------------------------


def _quantize_activations_fp8_block(
    x: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamically quantize activations to ``float8_e4m3fn`` with per-token,
    per-K-group scales.

    The K axis is split into contiguous groups of ``group_size`` (matching the
    weight's ``block_width``), so the combined activation/weight dequant scale is
    constant across each K-block of the matmul.

    :param x: activations ``[M, K]``
    :param group_size: number of K elements sharing one scale (``block_width``)
    :return: ``(x_fp8 [M, K], scale [M, ceil(K / group_size)])`` (scale is fp32)
    """
    m, k = x.shape
    num_groups = triton.cdiv(k, group_size)
    padded_k = num_groups * group_size

    xf = x.to(torch.float32)
    if padded_k != k:
        xf = torch.nn.functional.pad(xf, (0, padded_k - k))

    xg = xf.reshape(m, num_groups, group_size)
    amax = xg.abs().amax(dim=-1, keepdim=True)
    scale = (amax / FP8_E4M3_MAX).clamp(min=1e-12)

    x_fp8 = (xg / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.reshape(m, padded_k)[:, :k].contiguous()
    scale = scale.reshape(m, num_groups).contiguous()
    return x_fp8, scale


@triton.jit
def _fp8_block_fp8_kernel(
    x_ptr,  # [M, K] fp8 activations
    w_ptr,  # [N, K] fp8 weights
    sx_ptr,  # [M, NCB] per-token per-group activation scales
    sw_ptr,  # [NRB, NCB] per-block weight scales
    out_ptr,  # [M, N] output
    M,
    N,
    K,
    NRB,
    NCB,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_sxm,
    stride_sxc,
    stride_swn,
    stride_swc,
    stride_om,
    stride_on,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Block-scaled fp8 matmul: out = x @ dequant(w).T

    ``BLOCK_K`` equals the weight ``block_width`` and the activation group size,
    so each K-tile maps to exactly one weight block column and one activation
    group. The fp8 ``tl.dot`` runs on the tensor cores and its fp32 result is
    scaled by ``activation_scale[m] * weight_scale[n // BLOCK_H]`` (both constant
    over the tile) before being accumulated.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k0 = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N
    sw_row = offs_n // BLOCK_H  # weight block row indexed by output col n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = kt * BLOCK_K + offs_k0
        k_mask = offs_k < K

        # load fp8 activations [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # load fp8 weights transposed to [BLOCK_K, BLOCK_N]
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # per-tile scales (one activation group / weight block column == kt)
        sx = tl.load(sx_ptr + offs_m * stride_sxm + kt * stride_sxc, mask=m_mask)
        sw = tl.load(sw_ptr + sw_row * stride_swn + kt * stride_swc, mask=n_mask)

        acc += tl.dot(x, w) * sx[:, None] * sw[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _fp8_block_fp8_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_structure: tuple[int, int],
    bias: torch.Tensor | None,
) -> torch.Tensor:
    n, k = weight.shape
    m = x.shape[0]
    block_height, block_width = block_structure
    num_row_blocks, num_col_blocks = weight_scale.shape

    x_fp8, x_scale = _quantize_activations_fp8_block(x, block_width)
    w_scale = weight_scale.to(torch.float32).contiguous()
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)

    BLOCK_M, BLOCK_N = 64, 128
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    with torch.get_device_module().device(x.device):
        _fp8_block_fp8_kernel[grid](
            x_fp8,
            weight,
            x_scale,
            w_scale,
            out,
            m,
            n,
            k,
            num_row_blocks,
            num_col_blocks,
            x_fp8.stride(0),
            x_fp8.stride(1),
            weight.stride(0),
            weight.stride(1),
            x_scale.stride(0),
            x_scale.stride(1),
            w_scale.stride(0),
            w_scale.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_H=block_height,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=block_width,
        )

    out = out.to(x.dtype)
    if bias is not None:
        out += bias
    return out


@ImplBackend.register("fp8_block_forward", _fp8_tensorcore_req, 0)
def fp8_block_forward_fp8(module: torch.nn.Module, input: torch.Tensor) -> torch.Tensor:
    """FP8 tensor-core backend: dynamic-quant activations to fp8, real fp8 matmul."""
    scheme = module.quantization_scheme

    original_shape = input.shape
    input_2d = input.reshape(-1, original_shape[-1])

    output = _fp8_block_fp8_matmul(
        input_2d,
        module.weight,
        module.weight_scale,
        tuple(scheme.weights.block_structure),
        getattr(module, "bias", None),
    )
    return output.reshape(*original_shape[:-1], output.shape[-1])
