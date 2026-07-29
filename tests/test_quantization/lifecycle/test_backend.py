# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization.lifecycle.backend import (
    EagerQuantizationBackend,
    QuantizationBackend,
    get_quantization_backend,
    set_quantization_backend,
)
from compressed_tensors.quantization.lifecycle.forward_helpers import (
    _dequantize,
    _quantize,
    _quantize_dequantize,
)
from compressed_tensors.quantization.quant_args import QuantizationArgs


@pytest.fixture(autouse=True)
def _reset_backend():
    # ensure each test starts and ends on the default backend
    set_quantization_backend("eager")
    yield
    set_quantization_backend("eager")


def _args():
    return QuantizationArgs(num_bits=8, symmetric=True)


def _reference_qdq(x, scale, zero_point, q_min, q_max, args):
    from compressed_tensors.quantization.quant_args import round_to_quantized_type_args

    scaled = x / scale
    if zero_point is not None:
        scaled = scaled + zero_point.to(x.dtype)
    quantized = round_to_quantized_type_args(
        tensor=scaled, args=args, min=q_min, max=q_max
    )
    dequant = quantized.to(scale.dtype)
    if zero_point is not None:
        dequant = dequant - zero_point.to(scale.dtype)
    return dequant * scale


def test_eager_is_default_backend():
    assert get_quantization_backend() is EagerQuantizationBackend
    assert "eager" in QuantizationBackend.registered_names()


def test_eager_qdq_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(4, 128)
    scale = torch.tensor(0.05)
    zp = torch.tensor(0.0)
    q_min, q_max = torch.tensor(-128.0), torch.tensor(127.0)
    args = _args()

    got = _quantize_dequantize(x, scale, zp, q_min, q_max, args)
    ref = _reference_qdq(x, scale, zp, q_min, q_max, args)
    assert torch.equal(got, ref)


def test_quantize_then_dequantize_roundtrip():
    torch.manual_seed(1)
    x = torch.randn(8, 64)
    scale = torch.tensor(0.1)
    zp = torch.tensor(0.0)
    q_min, q_max = torch.tensor(-128.0), torch.tensor(127.0)
    args = _args()

    q = _quantize(x, scale, zp, q_min, q_max, args)
    dq = _dequantize(q, scale, zp)
    # dequantized value equals fused qdq
    assert torch.allclose(dq, _quantize_dequantize(x, scale, zp, q_min, q_max, args))


def test_set_unknown_backend_raises():
    with pytest.raises(KeyError):
        set_quantization_backend("does-not-exist")


def test_custom_backend_dispatch_and_fallback():
    sentinel = torch.tensor([1234.0])

    @QuantizationBackend.register(name="_test_sentinel")
    class SentinelBackend(QuantizationBackend):
        available = True

        @staticmethod
        def is_available(x, args):
            return SentinelBackend.available

        @staticmethod
        def quantize_dequantize(
            x, scale, zero_point, q_min, q_max, args, global_scale=None
        ):
            return sentinel

    args = _args()
    x = torch.randn(2, 8)
    scale, zp = torch.tensor(0.05), torch.tensor(0.0)
    q_min, q_max = torch.tensor(-128.0), torch.tensor(127.0)

    set_quantization_backend("_test_sentinel")
    # dispatched to the custom backend
    assert torch.equal(_quantize_dequantize(x, scale, zp, q_min, q_max, args), sentinel)

    # when the backend reports unavailable, transparently fall back to eager
    SentinelBackend.available = False
    out = _quantize_dequantize(x, scale, zp, q_min, q_max, args)
    assert not torch.equal(out, sentinel)
    assert torch.equal(out, _reference_qdq(x, scale, zp, q_min, q_max, args))
