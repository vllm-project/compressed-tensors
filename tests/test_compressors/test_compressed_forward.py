# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.compressors.naive_quantized import NaiveQuantizationCompressor
from compressed_tensors.offload import offload_module
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationMetadata,
    QuantizationScheme,
    QuantizationStatus,
    initialize_module_for_quantization,
)
from tests.testing_utils import requires_gpu


class CountingCompressor(NaiveQuantizationCompressor):
    """Naive compressor whose compressed forward returns a sentinel output"""

    num_resolutions = 0

    @classmethod
    def get_compressed_forward(cls, module):
        cls.num_resolutions += 1
        return _sentinel_forward


class NoForwardCompressor(NaiveQuantizationCompressor):
    @classmethod
    def get_compressed_forward(cls, module):
        return None


def _sentinel_forward(module: torch.nn.Linear, input: torch.Tensor) -> torch.Tensor:
    assert module.weight.dtype == torch.int8  # runs on compressed weight
    return torch.full((*input.shape[:-1], module.out_features), 7.0)


def _make_module() -> torch.nn.Linear:
    module = torch.nn.Linear(64, 32, bias=False)
    scheme = QuantizationScheme(
        targets=["Linear"], weights=QuantizationArgs(num_bits=8, symmetric=True)
    )
    initialize_module_for_quantization(module, scheme)
    module.weight_scale.data.fill_(0.01)
    return module


@pytest.fixture(autouse=True)
def reset_counts():
    CountingCompressor.num_resolutions = 0


def test_compressed_forward_resolved_once():
    module = _make_module()
    quantized_forward_func = module.forward.__func__

    CountingCompressor.compress_module(module)
    assert module.quantization_status == QuantizationStatus.COMPRESSED
    assert CountingCompressor.num_resolutions == 1
    assert module.forward.__func__ is not quantized_forward_func

    for _ in range(3):
        output = module(torch.randn(4, 64))
        assert torch.all(output == 7.0)
    assert CountingCompressor.num_resolutions == 1


def test_decompress_restores_forward():
    module = _make_module()
    quantized_forward_func = module.forward.__func__

    CountingCompressor.compress_module(module)
    CountingCompressor.decompress_module(module)
    assert module.quantization_status == QuantizationStatus.DECOMPRESSED
    assert module.forward.__func__ is quantized_forward_func

    # decompressed forward runs on the dense weight
    output = module(torch.randn(4, 64))
    assert output.shape == (4, 32)
    assert not torch.all(output == 7.0)

    # decompressing again does not unwrap the quantized forward
    NoForwardCompressor.decompress_module(module)
    assert module.forward.__func__ is quantized_forward_func


def test_no_compressed_forward():
    module = _make_module()
    quantized_forward_func = module.forward.__func__

    NoForwardCompressor.compress_module(module)
    assert module.forward.__func__ is quantized_forward_func
    NoForwardCompressor.decompress_module(module)
    assert module.forward.__func__ is quantized_forward_func


def test_clear_quantization_after_compress():
    module = _make_module()
    CountingCompressor.compress_module(module)
    CountingCompressor.decompress_module(module)
    QuantizationMetadata.clear_quantization(module)
    assert module.forward.__func__ is torch.nn.Linear.forward


@requires_gpu
def test_compressed_forward_offloaded():
    module = _make_module()
    offload_module(module, "cuda", "cpu")
    offload_forward = module.forward

    CountingCompressor.compress_module(module)
    # offload wrapper is kept as the outermost forward
    assert module.forward == offload_forward
    assert module._original_forward_func._is_compressed_forward
    assert torch.all(module(torch.randn(4, 64)) == 7.0)

    CountingCompressor.decompress_module(module)
    assert module.forward == offload_forward
    assert not getattr(module._original_forward_func, "_is_compressed_forward", False)
