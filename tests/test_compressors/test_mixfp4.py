# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.compressors.mixfp4 import (
    MixFP4PackedCompressor,
    pack_mixfp4_to_uint8,
    unpack_mixfp4_from_uint8,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    preset_name_to_scheme,
)


def _flagged_scale(is_int4: torch.Tensor) -> torch.Tensor:
    return _flagged_scale_with_magnitude(torch.ones_like(is_int4), is_int4)


def _flagged_scale_with_magnitude(
    magnitude: torch.Tensor, is_int4: torch.Tensor
) -> torch.Tensor:
    scale = magnitude.to(torch.float8_e4m3fn)
    raw = scale.view(torch.uint8).clone()
    raw = (raw & 0x7F) | (is_int4.to(torch.uint8) << 7)
    return raw.view(torch.float8_e4m3fn)


def test_mixfp4_pack_unpack_uses_scale_sign_bit_only():
    weight = torch.tensor(
        [
            [
                0.0,
                0.5,
                1.0,
                2.0,
                3.0,
                4.0,
                6.0,
                -6.0,
                -7.0,
                -3.0,
                -1.0,
                0.0,
                1.0,
                3.0,
                6.0,
                7.0,
            ],
            [1.0] * 16,
        ],
        dtype=torch.bfloat16,
    )
    scale = _flagged_scale(torch.tensor([[True], [False]]))
    global_scale = torch.tensor([1.0], dtype=torch.float32)

    packed = pack_mixfp4_to_uint8(weight, scale, global_scale)
    dequant = unpack_mixfp4_from_uint8(packed, scale, global_scale, dtype=torch.float32)

    assert packed.shape == (2, 8)
    assert not hasattr(MixFP4PackedCompressor, "format_flags")
    assert dequant[0, 8].item() == -7.0
    assert dequant[0, 15].item() == 7.0
    assert torch.all(dequant[1] == 1.0)


def test_mixfp4_pack_unpack_supports_per_row_global_scale():
    weight = torch.tensor(
        [
            [-7, -6, -5, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7],
            [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
        ],
        dtype=torch.bfloat16,
    )
    scale = _flagged_scale_with_magnitude(
        torch.tensor([[2.0], [4.0]]), torch.tensor([[True], [False]])
    )
    global_scale = torch.tensor([2.0, 4.0], dtype=torch.float32)

    packed = pack_mixfp4_to_uint8(weight, scale, global_scale)
    dequant = unpack_mixfp4_from_uint8(packed, scale, global_scale, dtype=torch.float32)

    assert torch.equal(dequant, weight.float())


def test_mixfp4_pack_unpack_supports_weight_scale_shaped_global_scale():
    int4_group = torch.tensor(
        [-7, -6, -5, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.bfloat16,
    )
    fp4_group = torch.tensor(
        [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
        dtype=torch.bfloat16,
    )
    weight = torch.stack(
        [torch.cat([int4_group, fp4_group]), torch.cat([fp4_group, int4_group])]
    )
    scale = _flagged_scale_with_magnitude(
        torch.tensor([[2.0, 4.0], [8.0, 16.0]]),
        torch.tensor([[True, False], [False, True]]),
    )
    global_scale = torch.tensor([[2.0, 4.0], [8.0, 16.0]], dtype=torch.float32)

    packed = pack_mixfp4_to_uint8(weight, scale, global_scale)
    dequant = unpack_mixfp4_from_uint8(packed, scale, global_scale, dtype=torch.float32)

    assert torch.equal(dequant, weight.float())


def test_mixfp4_legacy_format_alias_is_canonicalized():
    assert (
        CompressionFormat("mixed-fp4-int4-pack-quantized")
        is CompressionFormat.mixfp4_pack_quantized
    )
    assert (
        BaseCompressor.get_value_from_registry("mixed-fp4-int4-pack-quantized")
        is MixFP4PackedCompressor
    )
    scheme = preset_name_to_scheme("NVFP4_INT4_MIXED", ["Linear"])
    assert scheme.format == CompressionFormat.mixfp4_pack_quantized

    config = QuantizationConfig(
        config_groups={"group_1": QuantizationScheme(targets=["Linear"])},
        format="mixed-fp4-int4-pack-quantized",
    )
    assert config.format == CompressionFormat.mixfp4_pack_quantized.value
    assert config.to_dict()["format"] == CompressionFormat.mixfp4_pack_quantized.value


def test_mixfp4_does_not_steal_nvfp4_inference():
    nvfp4 = preset_name_to_scheme("NVFP4A16", ["Linear"])
    mixfp4 = preset_name_to_scheme("MIXFP4A16", ["Linear"])

    assert (
        infer_module_format(torch.nn.Linear, nvfp4)
        == CompressionFormat.nvfp4_pack_quantized
    )
    assert (
        infer_module_format(torch.nn.Linear, mixfp4)
        == CompressionFormat.mixfp4_pack_quantized
    )


def test_mixfp4_decompress_scale_strips_selector_bit():
    scheme = preset_name_to_scheme("MIXFP4A16", ["Linear"])
    state_dict = {
        "weight": torch.ones((2, 16), dtype=torch.bfloat16),
        "weight_scale": _flagged_scale(torch.tensor([[False], [True]])),
        "weight_global_scale": torch.tensor([1.0], dtype=torch.float32),
    }

    compressed = MixFP4PackedCompressor.compress(state_dict, scheme)
    compressed_raw = compressed["weight_scale"].view(torch.uint8)
    assert (compressed_raw[1, 0] & 0x80) != 0

    decompressed = MixFP4PackedCompressor.decompress(compressed, scheme)

    assert torch.equal(
        decompressed["weight_scale"], torch.ones((2, 1), dtype=torch.bfloat16)
    )
    assert torch.all(decompressed["weight_scale"] >= 0)


def test_mixfp4_decompress_rejects_non_fp8_scale():
    scheme = preset_name_to_scheme("MIXFP4A16", ["Linear"])
    state_dict = {
        "weight_packed": torch.zeros((1, 8), dtype=torch.uint8),
        "weight_scale": torch.ones((1, 1), dtype=torch.bfloat16),
        "weight_global_scale": torch.tensor([1.0], dtype=torch.float32),
    }

    with pytest.raises(ValueError, match="float8_e4m3fn weight_scale"):
        MixFP4PackedCompressor.decompress(state_dict, scheme)


def test_mixfp4_compress_casts_live_signed_scale_to_fp8():
    scheme = preset_name_to_scheme("MIXFP4A16", ["Linear"])
    flagged = _flagged_scale(torch.tensor([[False], [True]]))
    state_dict = {
        "weight": torch.ones((2, 16), dtype=torch.bfloat16),
        "weight_scale": flagged.to(torch.bfloat16),
        "weight_global_scale": torch.tensor([1.0], dtype=torch.float32),
    }

    compressed = MixFP4PackedCompressor.compress(state_dict, scheme)

    assert compressed["weight_scale"].dtype == torch.float8_e4m3fn
    assert torch.equal(
        compressed["weight_scale"].view(torch.uint8), flagged.view(torch.uint8)
    )


def test_mixfp4_zero_scale_group_packs_to_zero_nibbles():
    weight = torch.full((1, 16), 7.0, dtype=torch.bfloat16)
    scale = torch.zeros((1, 1), dtype=torch.float8_e4m3fn)
    global_scale = torch.tensor([1.0], dtype=torch.float32)

    packed = pack_mixfp4_to_uint8(weight, scale, global_scale)
    dequant = unpack_mixfp4_from_uint8(packed, scale, global_scale, dtype=torch.float32)

    assert torch.equal(packed, torch.zeros_like(packed))
    assert torch.equal(dequant, torch.zeros_like(dequant))


def test_mixfp4_compress_drops_zero_point_side_metadata():
    scheme = preset_name_to_scheme("MIXFP4A16", ["Linear"])
    state_dict = {
        "weight": torch.ones((2, 16), dtype=torch.bfloat16),
        "weight_scale": _flagged_scale(torch.tensor([[False], [True]])),
        "weight_global_scale": torch.tensor([1.0], dtype=torch.float32),
        "weight_zero_point": torch.zeros((2, 1), dtype=torch.float8_e4m3fn),
    }

    compressed = MixFP4PackedCompressor.compress(state_dict, scheme)

    assert set(compressed) == {
        "weight_packed",
        "weight_scale",
        "weight_global_scale",
    }
    assert compressed["weight_scale"].dtype == torch.float8_e4m3fn


def test_mixfp4_rejects_non_canonical_storage_contracts():
    scheme = preset_name_to_scheme("MIXFP4A16", ["Linear"])
    assert MixFP4PackedCompressor.can_compress(torch.nn.Linear, scheme)

    scheme.weights.symmetric = False
    assert not MixFP4PackedCompressor.can_compress(torch.nn.Linear, scheme)

    scheme = preset_name_to_scheme("MIXFP4A16", ["Linear"])
    scheme.weights.scale_dtype = torch.bfloat16
    assert not MixFP4PackedCompressor.can_compress(torch.nn.Linear, scheme)

    scheme = preset_name_to_scheme("MIXFP4A16", ["Linear"])
    scheme.weights = QuantizationArgs(num_bits=4, type="float", group_size=16)
    assert not MixFP4PackedCompressor.can_compress(torch.nn.Linear, scheme)


def test_mixfp4_rejects_invalid_global_scale():
    weight = torch.ones((1, 16), dtype=torch.bfloat16)
    scale = _flagged_scale(torch.tensor([[False]]))

    for global_scale in (torch.tensor([0.0]), torch.tensor([float("nan")])):
        with pytest.raises(ValueError, match="weight_global_scale"):
            pack_mixfp4_to_uint8(weight, scale, global_scale)


def test_mixfp4_fake_quantize_preserves_input_dtype():
    from compressed_tensors.quantization import fake_quantize

    weight = torch.ones((1, 16), dtype=torch.bfloat16)
    scale = _flagged_scale(torch.tensor([[False]]))
    zero_point = torch.zeros_like(scale)
    global_scale = torch.tensor([1.0], dtype=torch.float32)
    args = preset_name_to_scheme("MIXFP4A16", ["Linear"]).weights

    quantized = fake_quantize(
        weight, scale, zero_point, args, global_scale=global_scale
    )

    assert quantized.dtype == torch.bfloat16


def test_mixfp4_fake_quantize_supports_per_row_global_scale():
    from compressed_tensors.quantization import fake_quantize

    weight = torch.tensor(
        [
            [-7, -6, -5, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7],
            [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
        ],
        dtype=torch.bfloat16,
    )
    scale = _flagged_scale_with_magnitude(
        torch.tensor([[2.0], [4.0]]), torch.tensor([[True], [False]])
    )
    zero_point = torch.zeros_like(scale)
    global_scale = torch.tensor([2.0, 4.0], dtype=torch.float32)
    args = preset_name_to_scheme("MIXFP4A16", ["Linear"]).weights

    quantized = fake_quantize(
        weight, scale, zero_point, args, global_scale=global_scale
    )

    assert torch.equal(quantized, weight)


def test_mixfp4_fake_quantize_supports_weight_scale_shaped_global_scale():
    from compressed_tensors.quantization import fake_quantize

    int4_group = torch.tensor(
        [-7, -6, -5, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.bfloat16,
    )
    fp4_group = torch.tensor(
        [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
        dtype=torch.bfloat16,
    )
    weight = torch.stack(
        [torch.cat([int4_group, fp4_group]), torch.cat([fp4_group, int4_group])]
    )
    scale = _flagged_scale_with_magnitude(
        torch.tensor([[2.0, 4.0], [8.0, 16.0]]),
        torch.tensor([[True, False], [False, True]]),
    )
    zero_point = torch.zeros_like(scale)
    global_scale = torch.tensor([[2.0, 4.0], [8.0, 16.0]], dtype=torch.float32)
    args = preset_name_to_scheme("MIXFP4A16", ["Linear"]).weights

    quantized = fake_quantize(
        weight, scale, zero_point, args, global_scale=global_scale
    )

    assert torch.equal(quantized, weight)


def test_mixfp4_fake_quantize_interprets_scale_sign_bit():
    from compressed_tensors.quantization import fake_quantize

    int4_group = torch.tensor(
        [-7, -6, -5, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.float32,
    )
    fp4_group = torch.tensor(
        [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
        dtype=torch.float32,
    )
    weight = torch.stack([int4_group, fp4_group])
    scale = _flagged_scale(torch.tensor([[True], [False]]))
    zero_point = torch.zeros_like(scale)
    global_scale = torch.tensor([1.0], dtype=torch.float32)
    args = preset_name_to_scheme("MIXFP4A16", ["Linear"]).weights

    quantized = fake_quantize(
        weight, scale, zero_point, args, global_scale=global_scale
    )

    assert torch.equal(quantized, weight)
