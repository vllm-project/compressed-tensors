# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert import AutoAWQConverter
from compressed_tensors.quantization import QuantizationStatus
from transformers import AutoConfig


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

    unpacked, _ = AutoAWQConverter.unpack_awq(qweight, None, bits=4)
    reordered, _ = AutoAWQConverter.reverse_awq_order(unpacked, None, bits=4)

    assert torch.equal(torch.bitwise_and(unpacked, 15), packed_values)
    assert torch.equal(
        torch.bitwise_and(reordered, 15),
        torch.tensor([[0, 4, 1, 5, 2, 6, 3, 7]], dtype=torch.int8),
    )


@pytest.mark.unit
@pytest.mark.parametrize("zero_point", [True, False])
def test_autoawq_converter_processes_gemm_tensors(zero_point):
    converter = AutoAWQConverter(
        group_size=2,
        targets=[r"re:.*proj$"],
        zero_point=zero_point,
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
    if not zero_point:
        del tensors["model.layers.0.mlp.up_proj.qzeros"]

    converter.validate(tensors)
    converter.process(tensors)

    assert "model.layers.0.mlp.up_proj.qweight" not in tensors
    assert "model.layers.0.mlp.up_proj.qzeros" not in tensors
    assert "model.layers.0.mlp.up_proj.scales" not in tensors
    assert "model.layers.0.mlp.up_proj.weight" not in tensors
    assert tensors["model.layers.0.mlp.up_proj.weight_packed"].shape == (8, 1)
    assert torch.equal(
        tensors["model.layers.0.mlp.up_proj.weight_shape"], torch.tensor([8, 2])
    )
    assert tensors["model.layers.0.mlp.up_proj.weight_scale"].shape == (8, 1)
    assert tensors["model.layers.0.mlp.up_proj.weight_scale"].is_contiguous()

    if zero_point:
        assert tensors["model.layers.0.mlp.up_proj.weight_zero_point"].shape == (1, 1)
    else:
        assert "model.layers.0.mlp.up_proj.weight_zero_point" not in tensors


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
def test_autoawq_converter_from_pretrained(monkeypatch):
    class MockConfig:
        quantization_config = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 32,
            "zero_point": True,
            "version": "gemm",
            "modules_to_not_convert": ["visual"],
        }

    def mock_from_pretrained(model_name_or_path, trust_remote_code=False):
        assert model_name_or_path == "test/model"
        assert trust_remote_code is True
        return MockConfig()

    monkeypatch.setattr(AutoConfig, "from_pretrained", mock_from_pretrained)

    converter = AutoAWQConverter.from_pretrained(
        "test/model",
        trust_remote_code=True,
    )

    assert converter.bits == 4
    assert converter.group_size == 32
    assert converter.zero_point is True
    assert converter.version == "gemm"
    assert converter.ignore == ["lm_head", "re:.*visual.*"]


@pytest.mark.unit
def test_autoawq_converter_dependencies():
    converter = AutoAWQConverter(targets=[r"re:.*down_proj$"])

    assert converter.get_dependencies("model.layers.0.mlp.down_proj.qweight") == {
        "model.layers.0.mlp.down_proj.qzeros",
        "model.layers.0.mlp.down_proj.scales",
    }
    assert converter.get_dependencies("model.layers.0.mlp.up_proj.qweight") == set()

    symmetric_converter = AutoAWQConverter(
        targets=[r"re:.*down_proj$"],
        zero_point=False,
    )
    assert symmetric_converter.get_dependencies(
        "model.layers.0.mlp.down_proj.qweight"
    ) == {"model.layers.0.mlp.down_proj.scales"}


@pytest.mark.unit
def test_autoawq_converter_validate_requires_dependencies():
    converter = AutoAWQConverter()

    with pytest.raises(ValueError, match="without corresponding"):
        converter.validate(
            {"model.layers.0.mlp.down_proj.qweight": torch.zeros(1, 1, device="meta")}
        )
