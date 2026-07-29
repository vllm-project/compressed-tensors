# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.quant_scheme import preset_name_to_scheme
from compressed_tensors.quantization.utils import (
    LUT_B_PACKED_TILE_BYTES,
    dequantize_lut_b,
    fake_quantize_lut_b,
    pack_lut_b_indices,
    quantize_lut_b,
    unpack_lut_b_indices,
)


def test_lut_b_index_pack_round_trip():
    indices = torch.randint(0, 8, (5, 512), dtype=torch.uint8)

    packed = pack_lut_b_indices(indices)

    assert packed.shape == (5, LUT_B_PACKED_TILE_BYTES)
    torch.testing.assert_close(unpack_lut_b_indices(packed), indices)


def test_lut_b_index_pack_rejects_out_of_range_indices():
    with pytest.raises(ValueError, match=r"\[0, 7\]"):
        pack_lut_b_indices(torch.tensor([[-1] * 8], dtype=torch.int8))


def test_lut_b_canonical_layout_is_3_125_bits_per_weight():
    codebooks = torch.tensor(
        [[[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]]],
        dtype=torch.float8_e4m3fn,
    )
    indices = torch.arange(512, dtype=torch.uint8).remainder(8).reshape(1, 1, -1)
    packed = pack_lut_b_indices(indices)
    weight = dequantize_lut_b(packed, codebooks, dtype=torch.float32)

    repacked, fitted_codebooks = quantize_lut_b(weight, codebooks)
    reconstructed = dequantize_lut_b(
        repacked,
        fitted_codebooks,
        dtype=torch.float32,
    )

    assert repacked.shape == (1, 1, 192)
    assert fitted_codebooks.shape == (1, 1, 8)
    assert fitted_codebooks.dtype == torch.float8_e4m3fn
    assert repacked.numel() + fitted_codebooks.numel() == 200
    torch.testing.assert_close(reconstructed, weight)


def test_lut_b_qdq_fits_codebook_without_calibration():
    torch.manual_seed(0)
    weight = torch.randn(8, 64, dtype=torch.float32)

    reconstructed = fake_quantize_lut_b(weight)

    assert reconstructed.shape == weight.shape
    assert reconstructed.dtype == weight.dtype
    assert torch.isfinite(reconstructed).all()
    assert (reconstructed - weight).square().mean() < weight.square().mean()


def test_lut_b_fused_expert_tensors_match_stacked_2d_results():
    torch.manual_seed(1)
    weight = torch.randn(3, 16, 128)

    packed, codebooks = quantize_lut_b(weight)
    reconstructed = dequantize_lut_b(packed, codebooks, dtype=weight.dtype)
    stacked = [quantize_lut_b(expert) for expert in weight]

    assert packed.shape == (3, 2, 2, 192)
    assert codebooks.shape == (3, 2, 2, 8)
    assert reconstructed.shape == weight.shape
    torch.testing.assert_close(packed, torch.stack([item[0] for item in stacked]))
    torch.testing.assert_close(codebooks, torch.stack([item[1] for item in stacked]))


def test_lut_b_public_fake_quantize_dispatches_to_codebook_qdq():
    torch.manual_seed(1)
    weight = torch.randn(8, 64)
    args = preset_name_to_scheme("LUTB", ["Linear"]).weights

    reconstructed = fake_quantize(weight, None, None, args)

    torch.testing.assert_close(reconstructed, fake_quantize_lut_b(weight))


def test_lut_b_uses_supplied_calibrated_codebook():
    weight = torch.linspace(-1.0, 1.0, 512).reshape(8, 64)
    codebooks = torch.tensor(
        [[[1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 2.0]]],
        dtype=torch.float32,
    )

    packed, stored_codebooks = quantize_lut_b(weight, codebooks)
    reconstructed = dequantize_lut_b(
        packed,
        stored_codebooks,
        dtype=torch.float32,
    )

    expected_codebooks = (
        codebooks.to(torch.float8_e4m3fn)
        .to(torch.float32)
        .sort(dim=-1)
        .values.to(torch.float8_e4m3fn)
    )
    torch.testing.assert_close(stored_codebooks, expected_codebooks)
    assert torch.isin(
        reconstructed.flatten(),
        expected_codebooks.to(torch.float32).flatten(),
    ).all()


def test_lut_b_dequantization_preserves_codebook_order():
    codebooks = torch.tensor(
        [[[4.0, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0]]],
        dtype=torch.float8_e4m3fn,
    )
    indices = torch.zeros((1, 1, 512), dtype=torch.uint8)
    packed = pack_lut_b_indices(indices)

    reconstructed = dequantize_lut_b(packed, codebooks, dtype=torch.float32)

    torch.testing.assert_close(reconstructed, torch.full((8, 64), 4.0))


@pytest.mark.parametrize("shape", [(7, 64), (8, 63), (8, 64, 1)])
def test_lut_b_rejects_noncanonical_weight_shapes(shape):
    with pytest.raises(ValueError, match="LUT-B"):
        quantize_lut_b(torch.empty(shape))
