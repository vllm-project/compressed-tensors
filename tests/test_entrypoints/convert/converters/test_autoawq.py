# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert import (
    AutoAWQConverter,
    reverse_awq_order,
    unpack_awq,
)
from compressed_tensors.quantization import QuantizationStatus


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.int32)
    packed = torch.zeros(values.shape[0], values.shape[1] // 8, dtype=torch.int32)
    for offset in range(8):
        packed |= values[:, offset::8] << (offset * 4)
    return packed


@pytest.mark.unit
def test_unpack_awq_and_reverse_order():
    packed_values = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int8)
    qweight = _pack_int4(packed_values)

    unpacked, _ = unpack_awq(qweight, None, bits=4)
    reordered, _ = reverse_awq_order(unpacked, None, bits=4)

    assert torch.equal(torch.bitwise_and(unpacked, 15), packed_values)
    assert torch.equal(
        torch.bitwise_and(reordered, 15),
        torch.tensor([[0, 4, 1, 5, 2, 6, 3, 7]], dtype=torch.int8),
    )


@pytest.mark.unit
def test_autoawq_converter_processes_gemm_tensors():
    converter = AutoAWQConverter(
        group_size=2,
        targets=[r"re:.*proj$"],
    )
    qweight_values = torch.tensor(
        [
            [8, 9, 10, 11, 12, 13, 14, 15],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ],
        dtype=torch.int8,
    )
    qzeros_values = torch.tensor([[8, 8, 8, 8, 8, 8, 8, 8]], dtype=torch.int8)
    scales = torch.ones(1, 8, dtype=torch.float16)
    tensors = {
        "model.layers.0.mlp.up_proj.qweight": _pack_int4(qweight_values),
        "model.layers.0.mlp.up_proj.qzeros": _pack_int4(qzeros_values),
        "model.layers.0.mlp.up_proj.scales": scales,
        "model.embed_tokens.weight": torch.ones(4, 4),
    }

    converter.validate(tensors)
    converter.process(tensors)

    assert "model.layers.0.mlp.up_proj.qweight" not in tensors
    assert "model.layers.0.mlp.up_proj.qzeros" not in tensors
    assert "model.layers.0.mlp.up_proj.scales" not in tensors
    assert tensors["model.layers.0.mlp.up_proj.weight"].shape == (8, 2)
    assert tensors["model.layers.0.mlp.up_proj.weight"].dtype == torch.int8
    assert tensors["model.layers.0.mlp.up_proj.weight_scale"].shape == (8, 1)
    assert tensors["model.layers.0.mlp.up_proj.weight_scale"].is_contiguous()
    assert tensors["model.layers.0.mlp.up_proj.weight_zero_point"].shape == (8, 1)

    assert torch.equal(
        tensors["model.layers.0.mlp.up_proj.weight_zero_point"],
        torch.zeros(8, 1, dtype=torch.int8),
    )
    assert torch.equal(
        tensors["model.layers.0.mlp.up_proj.weight"][:, 0],
        torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int8),
    )
    assert torch.equal(
        tensors["model.layers.0.mlp.up_proj.weight"][:, 1],
        torch.tensor([-8, -4, -7, -3, -6, -2, -5, -1], dtype=torch.int8),
    )


@pytest.mark.unit
def test_autoawq_converter_processes_packed_gemm_tensors():
    converter = AutoAWQConverter(
        group_size=2,
        targets=[r"re:.*proj$"],
        quantization_format=CompressionFormat.pack_quantized.value,
    )
    tensors = {
        "model.layers.0.mlp.up_proj.qweight": _pack_int4(
            torch.tensor(
                [
                    [8, 9, 10, 11, 12, 13, 14, 15],
                    [0, 1, 2, 3, 4, 5, 6, 7],
                ],
                dtype=torch.int8,
            )
        ),
        "model.layers.0.mlp.up_proj.qzeros": _pack_int4(
            torch.full((1, 8), 8, dtype=torch.int8)
        ),
        "model.layers.0.mlp.up_proj.scales": torch.ones(1, 8, dtype=torch.float16),
    }

    converter.process(tensors)

    assert "model.layers.0.mlp.up_proj.weight" not in tensors
    assert tensors["model.layers.0.mlp.up_proj.weight_packed"].shape == (8, 1)
    assert torch.equal(
        tensors["model.layers.0.mlp.up_proj.weight_shape"], torch.tensor([8, 2])
    )
    assert tensors["model.layers.0.mlp.up_proj.weight_zero_point"].shape == (1, 1)


@pytest.mark.unit
def test_autoawq_converter_config_from_autoawq_config():
    converter = AutoAWQConverter.from_autoawq_config(
        {
            "bits": 4,
            "group_size": 64,
            "zero_point": True,
            "version": "gemm",
            "modules_to_not_convert": ["vision_tower"],
        },
        quantization_format=CompressionFormat.pack_quantized.value,
    )

    config = converter.create_config()
    scheme = config.config_groups["config_group_0"]

    assert config.format == CompressionFormat.pack_quantized.value
    assert config.quantization_status == QuantizationStatus.COMPRESSED
    assert config.ignore == ["lm_head", "re:.*vision_tower.*"]
    assert scheme.format == CompressionFormat.pack_quantized.value
    assert scheme.weights.num_bits == 4
    assert scheme.weights.group_size == 64
    assert scheme.weights.symmetric is False


@pytest.mark.unit
def test_autoawq_converter_dependencies():
    converter = AutoAWQConverter(targets=[r"re:.*down_proj$"])

    assert converter.get_dependencies("model.layers.0.mlp.down_proj.qweight") == {
        "model.layers.0.mlp.down_proj.qzeros",
        "model.layers.0.mlp.down_proj.scales",
    }
    assert converter.get_dependencies("model.layers.0.mlp.up_proj.qweight") == set()


@pytest.mark.unit
def test_autoawq_converter_validate_requires_dependencies():
    converter = AutoAWQConverter()

    with pytest.raises(ValueError, match="without corresponding"):
        converter.validate({"model.layers.0.mlp.down_proj.qweight": torch.zeros(1, 1)})
