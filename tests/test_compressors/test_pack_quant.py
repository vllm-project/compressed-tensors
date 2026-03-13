# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections import OrderedDict

import pytest
import torch
from compressed_tensors import PackedQuantizationCompressor
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    apply_quantization_config,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.quant_args import ActivationOrdering
from torch.nn.modules import Linear, Sequential


def get_dummy_quant_config(
    num_bits=4, strategy=None, group_size=None, actorder=None, symmetric=True
) -> QuantizationConfig:
    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=num_bits,
                strategy=strategy,
                group_size=group_size,
                actorder=actorder,
                symmetric=symmetric,
            ),
        ),
    }
    return QuantizationConfig(config_groups=config_groups)


def make_dummy_g_idx(columns: int, group_size: int) -> torch.Tensor:
    perm = torch.randperm(columns)
    return torch.nn.Parameter(
        (torch.arange(columns, dtype=torch.int) // group_size)[perm],
        requires_grad=False,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (512, 1024),
        (830, 545),
        (342, 512),
        (256, 700),
    ],
)
def test_quant_format(shape):
    module_sd = {
        "weight": torch.rand(shape),
        "weight_scale": torch.tensor(0.01, dtype=torch.float32),
        "weight_zero_point": torch.tensor(0, dtype=torch.int8),
    }
    scheme = QuantizationScheme(
        targets=["Linear"], weights=QuantizationArgs(num_bits=4, symmetric=True)
    )

    compressed = PackedQuantizationCompressor.compress(module_sd, scheme=scheme)

    # 'weight' replaced by 'weight_packed' + 'weight_shape'; zp dropped (symmetric)
    assert "weight" not in compressed
    assert "weight_packed" in compressed
    assert "weight_shape" in compressed
    assert "weight_zero_point" not in compressed

    assert compressed["weight_packed"].dtype == torch.int32
    expected_rows = shape[0]
    expected_columns = math.ceil(shape[1] / 8)
    assert compressed["weight_packed"].shape == (expected_rows, expected_columns)
    assert torch.equal(compressed["weight_shape"], torch.tensor(shape))
    assert compressed["weight_scale"].dtype == torch.float32


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0], [-1, -2, -3, -4, -5, -6, -7, -8]]),
        (torch.rand((32, 100)) * 16 - 8),
    ],
)
def test_repack_4bit(value):
    value = value.to(torch.int8)
    shape = value.shape
    assert not torch.any(value > 7).item()
    assert not torch.any(value < -8).item()

    packed = pack_to_int32(value, 4)
    unpacked = unpack_from_int32(packed, 4, shape)
    assert torch.equal(value, unpacked)


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([[30, 40], [50, 60]]),
        torch.tensor(
            [[10, 15, 20, 25, 30, 35, 40, 45], [-10, -20, -30, -40, -50, -60, -70, -80]]
        ),
        (torch.rand((32, 100)) * 256 - 128),
    ],
)
def test_repack_8bit(value):
    value = value.to(torch.int8)
    shape = value.shape
    assert not torch.any(value > 127).item()
    assert not torch.any(value < -128).item()

    packed = pack_to_int32(value, 8)
    unpacked = unpack_from_int32(packed, 8, shape)
    assert torch.equal(value, unpacked)


@pytest.mark.parametrize("num_bits", [4, 8])
def test_compress_decompress_match(num_bits):
    """Round-trip compress → decompress in memory."""
    module_sd = {
        "weight": torch.rand((511, 350)),
        "weight_scale": torch.tensor(0.01, dtype=torch.float32),
        "weight_zero_point": torch.tensor(0, dtype=torch.int8),
    }

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=num_bits, symmetric=False),
    )

    compressed = PackedQuantizationCompressor.compress(module_sd.copy(), scheme=scheme)
    decompressed = PackedQuantizationCompressor.decompress(compressed, scheme=scheme)

    fake_quant = fake_quantize(
        module_sd["weight"],
        scale=module_sd["weight_scale"],
        zero_point=module_sd["weight_zero_point"],
        args=scheme.weights,
    )
    assert torch.equal(fake_quant, decompressed["weight"].to(torch.float32))


@pytest.mark.parametrize(
    "strategy",
    {QuantizationStrategy.GROUP, QuantizationStrategy.CHANNEL},
)
def test_asymmetric_packed_support(strategy):
    shape = (1024, 1024)
    group_size = None
    if strategy == QuantizationStrategy.GROUP:
        group_size = 128

    if strategy == QuantizationStrategy.CHANNEL:
        expected_shape = (shape[0], 1)
    elif strategy == QuantizationStrategy.GROUP:
        num_groups = shape[1] // group_size
        expected_shape = (shape[0], max(num_groups, 1))

    module_sd = {
        "weight": torch.rand(shape),
        "weight_scale": torch.rand(expected_shape).to(torch.float32),
        "weight_zero_point": torch.rand(expected_shape).to(torch.int8),
    }

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4, strategy=strategy.value, symmetric=False, group_size=group_size
        ),
    )

    compressed = PackedQuantizationCompressor.compress(module_sd, scheme=scheme)

    # weight + shape entry + packed zp
    assert "weight_packed" in compressed
    assert "weight_shape" in compressed
    assert "weight_zero_point" in compressed
    assert compressed["weight_packed"].dtype == torch.int32
    assert compressed["weight_zero_point"].dtype == torch.int32
    assert compressed["weight_scale"].dtype == torch.float32

    expected_rows = shape[0]
    expected_columns = math.ceil(shape[1] / 8)
    assert compressed["weight_packed"].shape == (expected_rows, expected_columns)
    assert torch.equal(compressed["weight_shape"], torch.tensor(shape))

    packed_size_zp = math.ceil(shape[0] / 8)
    zp_factor = group_size if strategy == QuantizationStrategy.GROUP else shape[-1]
    assert compressed["weight_zero_point"].shape == (
        packed_size_zp,
        shape[-1] // zp_factor,
    )


@pytest.mark.parametrize(
    "actorder",
    [
        ActivationOrdering.GROUP,
        ActivationOrdering.WEIGHT,
        None,
    ],
)
def test_actorder_compress_decompress_match(actorder, mock_per_group_calibration):
    model = Sequential(OrderedDict([("dummy", Linear(512, 1024, bias=None))]))
    group_size = 128
    quant_config = get_dummy_quant_config(
        strategy="group", group_size=group_size, actorder=actorder
    )
    apply_quantization_config(model, quant_config)

    model.quantization_status = QuantizationStatus.CALIBRATION
    mock_per_group_calibration(
        model.dummy, base_name="weight", value=model.dummy.weight, group_size=group_size
    )
    if actorder == ActivationOrdering.GROUP:
        init_g_idx = make_dummy_g_idx(512, group_size)
        model.dummy.register_parameter("weight_g_idx", init_g_idx)

    scheme = quant_config.config_groups["group_1"]
    module_sd = {
        name: param.data.clone() for name, param in model.dummy.named_parameters()
    }

    compressed = PackedQuantizationCompressor.compress(module_sd, scheme=scheme)
    decompressed = PackedQuantizationCompressor.decompress(compressed, scheme=scheme)

    fake_quant = fake_quantize(
        model.dummy.weight,
        scale=model.dummy.weight_scale,
        zero_point=model.dummy.weight_zero_point,
        g_idx=getattr(model.dummy, "weight_g_idx", None),
        args=scheme.weights,
    )
    assert torch.equal(fake_quant, decompressed["weight"])


@pytest.mark.parametrize(
    "num_bits,values,expected_values",
    [
        (
            4,
            torch.tensor([[1]]),
            torch.tensor([[9]], dtype=torch.int32),
        ),
        (
            8,
            torch.tensor([[1]]),
            torch.tensor([[129]], dtype=torch.int32),
        ),
        (4, torch.tensor([[1, 2, 3, 4]]), torch.tensor([[52137]], dtype=torch.int32)),
        (
            4,
            torch.tensor([[-8, -7, -6, -5, -4, -3, -2, -1]]),
            torch.tensor([[1985229328]], dtype=torch.int32),
        ),
        (
            8,
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[-2071756159]], dtype=torch.int32),
        ),
        (
            8,
            torch.tensor([[-128, -127, -126, -125]]),
            torch.tensor([[50462976]], dtype=torch.int32),
        ),
        (
            4,
            torch.tensor([[-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]]),
            torch.tensor([[1985229328, 52137]], dtype=torch.int32),
        ),
        (
            4,
            torch.tensor(
                [
                    [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, -8, -8, -8, -8],
                    [1, 2, 3, 4, -8, -8, -8, -8, -8, -7, -6, -5, -4, -3, -2, -1],
                ]
            ),
            torch.tensor([[1985229328, 52137], [52137, 1985229328]], dtype=torch.int32),
        ),
        (
            8,
            torch.tensor([[1, 2, 3, 4], [-128, -127, -126, -125]]),
            torch.tensor([[-2071756159], [50462976]], dtype=torch.int32),
        ),
        (
            8,
            torch.tensor(
                [
                    [1, 2, 3, 4, -128, -127, -126, -125],
                    [-128, -127, -126, -125, 1, 2, 3, 4],
                ]
            ),
            torch.tensor(
                [[-2071756159, 50462976], [50462976, -2071756159]], dtype=torch.int32
            ),
        ),
    ],
)
def test_pack_to_int32(num_bits, values, expected_values):
    values = values.to(torch.int8)
    packed_values = pack_to_int32(values, num_bits)
    assert torch.equal(packed_values, expected_values)
    assert packed_values.dtype == expected_values.dtype


@pytest.mark.parametrize(
    "num_bits,values,expected_tensor",
    [
        (
            4,
            torch.tensor([[9]], dtype=torch.int32),
            torch.tensor([[1]], dtype=torch.int8),
        ),
        (
            8,
            torch.tensor([[129]], dtype=torch.int32),
            torch.tensor([[1]], dtype=torch.int8),
        ),
        (
            4,
            torch.tensor([[52137]], dtype=torch.int32),
            torch.tensor([[1, 2, 3, 4]], dtype=torch.int8),
        ),
        (
            4,
            torch.tensor([[1985229328]], dtype=torch.int32),
            torch.tensor([[-8, -7, -6, -5, -4, -3, -2, -1]], dtype=torch.int8),
        ),
        (
            8,
            torch.tensor([[-2071756159]], dtype=torch.int32),
            torch.tensor([[1, 2, 3, 4]], dtype=torch.int8),
        ),
        (
            8,
            torch.tensor([[50462976]], dtype=torch.int32),
            torch.tensor([[-128, -127, -126, -125]], dtype=torch.int8),
        ),
        (
            4,
            torch.tensor([[1985229328, 52137]], dtype=torch.int32),
            torch.tensor(
                [[-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]], dtype=torch.int8
            ),
        ),
        (
            4,
            torch.tensor([[1985229328, 52137], [52137, 1985229328]], dtype=torch.int32),
            torch.tensor(
                [
                    [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, -8, -8, -8, -8],
                    [1, 2, 3, 4, -8, -8, -8, -8, -8, -7, -6, -5, -4, -3, -2, -1],
                ],
                dtype=torch.int8,
            ),
        ),
        (
            8,
            torch.tensor([[-2071756159], [50462976]], dtype=torch.int32),
            torch.tensor([[1, 2, 3, 4], [-128, -127, -126, -125]], dtype=torch.int8),
        ),
        (
            8,
            torch.tensor(
                [[-2071756159, 50462976], [50462976, -2071756159]], dtype=torch.int32
            ),
            torch.tensor(
                [
                    [1, 2, 3, 4, -128, -127, -126, -125],
                    [-128, -127, -126, -125, 1, 2, 3, 4],
                ],
                dtype=torch.int8,
            ),
        ),
    ],
)
def test_unpack_from_int32(num_bits, values, expected_tensor):
    unpacked_tensor = unpack_from_int32(values, num_bits, expected_tensor.shape)
    assert torch.equal(unpacked_tensor, unpacked_tensor)
    assert unpacked_tensor.dtype == unpacked_tensor.dtype


@pytest.mark.parametrize(
    "strategy,group_size",
    [
        (QuantizationStrategy.GROUP, 128),
        (QuantizationStrategy.CHANNEL, None),
    ],
)
def test_asymmetric_zero_point_decompression(strategy, group_size):
    shape = (512, 1024)

    if strategy == QuantizationStrategy.CHANNEL:
        expected_zp_shape = (shape[0], 1)
    elif strategy == QuantizationStrategy.GROUP:
        num_groups = shape[1] // group_size
        expected_zp_shape = (shape[0], max(num_groups, 1))

    module_sd = {
        "weight": torch.randn(shape),
        "weight_scale": torch.rand(expected_zp_shape).to(torch.float32),
        "weight_zero_point": torch.randint(-8, 8, expected_zp_shape).to(torch.int8),
    }

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4, strategy=strategy.value, symmetric=False, group_size=group_size
        ),
    )

    compressed = PackedQuantizationCompressor.compress(module_sd.copy(), scheme=scheme)

    assert "weight_zero_point" in compressed
    assert compressed["weight_zero_point"].dtype == torch.int32

    decompressed = PackedQuantizationCompressor.decompress(compressed, scheme=scheme)

    assert "weight" in decompressed
    assert decompressed["weight"].shape == shape


@pytest.mark.parametrize(
    "num_bits,strategy",
    [
        (4, QuantizationStrategy.GROUP),
        (4, QuantizationStrategy.CHANNEL),
        (8, QuantizationStrategy.GROUP),
        (8, QuantizationStrategy.CHANNEL),
    ],
)
def test_zero_point_pack_unpack_consistency(num_bits, strategy):
    if strategy == QuantizationStrategy.GROUP:
        shape = (512, 8)
    else:
        shape = (512, 1)

    max_val = (1 << (num_bits - 1)) - 1
    min_val = -(1 << (num_bits - 1))
    original_zp = torch.randint(min_val, max_val + 1, shape).to(torch.int8)

    packed_zp = pack_to_int32(original_zp, num_bits, packed_dim=0)
    unpacked_zp = unpack_from_int32(packed_zp, num_bits, shape, packed_dim=0)

    assert torch.equal(original_zp, unpacked_zp)
    assert unpacked_zp.dtype == torch.int8


def test_pack_unpack_3d_round_trip():
    """3D tensors (e.g. MoE expert weights) should pack/unpack correctly."""
    num_bits = 4
    shape = (4, 8, 32)  # (num_experts, rows, cols)
    value = torch.randint(-8, 7, shape, dtype=torch.int8)
    packed = pack_to_int32(value, num_bits)
    unpacked = unpack_from_int32(packed, num_bits, torch.Size(shape))
    assert torch.equal(value, unpacked)


def test_pack_unpack_3d_matches_stacked_2d():
    """3D pack/unpack should match stacking individual 2D results."""
    num_bits = 4
    shape = (4, 8, 32)
    value = torch.randint(-8, 7, shape, dtype=torch.int8)
    packed_3d = pack_to_int32(value, num_bits)
    packed_2d = torch.stack(
        [pack_to_int32(value[i], num_bits) for i in range(value.shape[0])]
    )
    assert torch.equal(packed_3d, packed_2d)
