# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors import IntQuantizationCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize


def make_quant_scheme(strategy, group_size=None, symmetric=True):
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            strategy=strategy, group_size=group_size, symmetric=symmetric
        ),
    )


@pytest.mark.parametrize(
    "strategy,symmetric,group_size,sc,zp",
    [
        [QuantizationStrategy.TENSOR, True, None, torch.tensor(0.01), torch.tensor(0)],
        [
            QuantizationStrategy.GROUP,
            True,
            128,
            torch.rand((512, 8)) * 0.01,
            torch.zeros((512, 8), dtype=torch.int8),
        ],
        [
            QuantizationStrategy.CHANNEL,
            False,
            None,
            torch.rand((512, 1)) * 0.01,
            ((torch.rand((512, 1)) - 0.5) * 127).to(torch.int8),
        ],
    ],
)
def test_quant_format(strategy, symmetric, group_size, sc, zp):
    module_sd = {
        "weight": torch.rand((512, 1024)),
        "weight_scale": sc.to(torch.float32),
        "weight_zero_point": zp.to(torch.int32),
    }
    scheme = make_quant_scheme(
        strategy=strategy, group_size=group_size, symmetric=symmetric
    )

    compressed = IntQuantizationCompressor.compress(module_sd, scheme=scheme)

    # zero_point is dropped for symmetric quantization
    if symmetric:
        assert "weight_zero_point" not in compressed
    else:
        assert compressed["weight_zero_point"].dtype == torch.int32

    # weight should be compressed to int8
    assert compressed["weight"].dtype == torch.int8
    assert compressed["weight_scale"].dtype == torch.float32


@pytest.mark.parametrize(
    "strategy,group_size,sc,zp",
    [
        [QuantizationStrategy.TENSOR, None, torch.tensor(0.01), torch.tensor(0)],
        [
            QuantizationStrategy.GROUP,
            128,
            torch.rand((300, 8)) * 0.01,
            torch.zeros((300, 8), dtype=torch.int8),
        ],
        [
            QuantizationStrategy.CHANNEL,
            None,
            torch.rand((300, 1)) * 0.01,
            torch.zeros((300, 1), dtype=torch.int8),
        ],
    ],
)
def test_compress_decompress_match(strategy, group_size, sc, zp):
    module_sd = {
        "weight": torch.rand((300, 1024)),
        "weight_scale": sc.to(torch.float32),
        "weight_zero_point": zp.to(torch.int32),
    }
    scheme = make_quant_scheme(strategy=strategy, group_size=group_size)

    compressed = IntQuantizationCompressor.compress(module_sd, scheme=scheme)
    decompressed = IntQuantizationCompressor.decompress(compressed, scheme=scheme)

    fake_quant = fake_quantize(
        module_sd["weight"],
        scale=module_sd["weight_scale"],
        zero_point=module_sd["weight_zero_point"],
        args=scheme.weights,
    )
    assert torch.equal(fake_quant, decompressed["weight"].to(torch.float32))
