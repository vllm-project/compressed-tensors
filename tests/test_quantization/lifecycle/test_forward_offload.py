# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.offload import disable_offloading, offload_module
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
)
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)


class ScaledLinear(torch.nn.Linear):
    def forward(self, inputs):
        return super().forward(inputs) * 2


class ScaledEmbedding(torch.nn.Embedding):
    def forward(self, inputs):
        return super().forward(inputs) * 2


def make_layer(layer_cls, device):
    is_linear = issubclass(layer_cls, torch.nn.Linear)
    layer = layer_cls(4, 2, bias=False) if is_linear else layer_cls(2, 4)
    layer.weight.data.copy_(
        torch.tensor([[0.14, 0.26, -0.34, -0.46], [0.26, -0.14, 0.46, -0.34]])
    )
    scheme = QuantizationScheme(
        targets=["Linear" if is_linear else "Embedding"],
        weights=QuantizationArgs(num_bits=8, type="int", strategy="channel"),
    )
    initialize_module_for_quantization(layer, scheme)
    layer.weight_scale.data.fill_(0.1)
    layer.weight_zero_point.data.zero_()
    layer.quantization_status = QuantizationStatus.FROZEN
    layer = layer.to(device)
    inputs = (
        torch.eye(4, device=device)
        if is_linear
        else torch.tensor([0, 1, 0], device=device)
    )
    raw_weight = layer.weight.detach().clone()
    quantized_weight = forward_quantize(layer, raw_weight, "weight", scheme.weights)
    op = torch.nn.functional.linear if is_linear else torch.nn.functional.embedding
    multiplier = 2 if layer_cls in (ScaledLinear, ScaledEmbedding) else 1
    raw_output = op(inputs, raw_weight) * multiplier
    expected = op(inputs, quantized_weight) * multiplier
    assert not torch.equal(raw_output, expected)
    return layer, inputs, raw_weight, raw_output, expected


@pytest.mark.parametrize(
    "layer_cls", [torch.nn.Linear, torch.nn.Embedding, ScaledLinear, ScaledEmbedding]
)
@pytest.mark.parametrize("offload_device", [None, "cpu", "disk"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("device", ["cpu", "accelerator"])
def test_quantized_forward_offload(
    layer_cls, offload_device, enabled, device, tmp_path
):
    if device == "accelerator":
        if not torch.accelerator.is_available():
            pytest.skip("Accelerator device required")
        device = torch.accelerator.current_accelerator()
    layer, inputs, raw_weight, raw_output, expected = make_layer(layer_cls, device)
    layer.quantization_enabled = enabled
    if offload_device is not None:
        offload_module(
            layer,
            onload_device=device,
            offload_device=offload_device,
            **({"offload_dir": str(tmp_path)} if offload_device == "disk" else {}),
        )
    for _ in range(2):
        with torch.no_grad():
            actual = layer(inputs)
        torch.testing.assert_close(
            actual, expected if enabled else raw_output, rtol=0, atol=0
        )
        torch.testing.assert_close(layer.weight, raw_weight, rtol=0, atol=0)
        assert not OffloadCache.offloading_disabled
        assert not OffloadCache.keep_onloaded_values


@pytest.mark.parametrize("layer_cls", [ScaledLinear, ScaledEmbedding])
def test_quantized_forward_preserves_outer_offload_context(layer_cls, tmp_path):
    layer, inputs, raw_weight, _, expected = make_layer(layer_cls, "cpu")
    offload_module(layer, "cpu", "disk", offload_dir=str(tmp_path))
    with disable_offloading():
        cached_weight = layer.weight
        with torch.no_grad():
            actual = layer(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert layer.weight is cached_weight
        torch.testing.assert_close(cached_weight, raw_weight, rtol=0, atol=0)
        assert OffloadCache.offloading_disabled
        assert OffloadCache.keep_onloaded_values
    assert not OffloadCache.offloading_disabled
    assert not OffloadCache.keep_onloaded_values


@pytest.mark.parametrize("outer_context", [False, True])
def test_quantized_forward_offload_restores_weight_on_error(outer_context, tmp_path):
    from contextlib import nullcontext

    class FailingLinear(torch.nn.Linear):
        fail = True

        def forward(self, inputs):
            if self.fail:
                self.observed_weight = self.weight.detach().clone()
                raise RuntimeError("forward failed")
            return super().forward(inputs)

    layer, inputs, raw_weight, _, expected = make_layer(FailingLinear, "cpu")
    quantized_weight = forward_quantize(
        layer, raw_weight, "weight", layer.quantization_scheme.weights
    )
    offload_module(layer, "cpu", "disk", offload_dir=str(tmp_path))
    with disable_offloading() if outer_context else nullcontext():
        with pytest.raises(RuntimeError, match="forward failed"):
            layer(inputs)
        torch.testing.assert_close(
            layer.observed_weight, quantized_weight, rtol=0, atol=0
        )
        torch.testing.assert_close(layer.weight, raw_weight, rtol=0, atol=0)
        assert OffloadCache.offloading_disabled == outer_context
        assert bool(OffloadCache.keep_onloaded_values) == outer_context
        layer.fail = False
        with torch.no_grad():
            actual = layer(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not OffloadCache.offloading_disabled
    assert not OffloadCache.keep_onloaded_values
