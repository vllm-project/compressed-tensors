# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.entrypoints.convert import FP8BlockDequantizer


@pytest.mark.unit
def test_fp8_block_to_bfloat16_conversion():
    """
    Test that _create_bfloat16_weight correctly converts FP8 block-quantized
    weights to bfloat16 by multiplying by the scale_inv per block.
    """
    converter = FP8BlockDequantizer(weight_block_size=(128, 128))

    # Create a weight tensor divisible by block size (256x256 = 2x2 blocks of 128x128)
    original_weight = torch.randn(256, 256, dtype=torch.bfloat16)

    # Simulate block quantization: divide into blocks and create per-block scales
    num_row_blocks = 2
    num_col_blocks = 2

    # Create per-block scale_inv (2x2 for 2x2 blocks)
    weight_scale_inv = torch.randn(num_row_blocks, num_col_blocks, dtype=torch.float32)

    # Convert original to fp8 (simulate quantization by just converting dtype)
    weight_fp8 = original_weight.to(torch.float32).to(torch.float8_e4m3fn)

    # Test conversion
    result = converter._create_bfloat16_weight(weight_fp8, weight_scale_inv)

    # Verify using helper
    _verify_block_conversion(result, weight_fp8, weight_scale_inv, (128, 128))


@pytest.mark.unit
def test_fp8_block_to_bfloat16_conversion_with_padding():
    """
    Test that _create_bfloat16_weight correctly handles tensors that need padding
    (dimensions not evenly divisible by block size).
    """
    converter = FP8BlockDequantizer(weight_block_size=(128, 128))

    # Create a weight tensor NOT divisible by block size (200x300)
    # Should be padded to 256x384 (2x3 blocks)
    weight_fp8 = torch.randn(200, 300, dtype=torch.float32).to(torch.float8_e4m3fn)

    # Scale_inv for padded size: 2 row blocks x 3 col blocks
    num_row_blocks = 2  # ceil(200/128) = 2
    num_col_blocks = 3  # ceil(300/128) = 3
    weight_scale_inv = torch.ones(num_row_blocks, num_col_blocks, dtype=torch.float32)

    # Test conversion
    result = converter._create_bfloat16_weight(weight_fp8, weight_scale_inv)

    # Verify output shape matches original (not padded)
    assert result.shape == (200, 300), "Output shape should match original, not padded"
    assert result.dtype == torch.bfloat16, "Output dtype should be bfloat16"


@pytest.mark.unit
def test_fp8_block_converter_process():
    """
    Test that the converter's process method correctly converts FP8 block-quantized
    tensors in a dict to bfloat16, removing weight_scale_inv tensors.
    """
    converter = FP8BlockDequantizer(
        targets=[r"re:.*layer\d+\.mlp\..*proj$"], weight_block_size=(128, 128)
    )

    # Create mock tensors dict with FP8 weights and scale_inv tensors
    num_row_blocks = 2
    num_col_blocks = 2

    # Non-targeted tensor (should not be modified)
    non_targeted_weight = torch.randn(128, 128, dtype=torch.bfloat16)

    tensors = {
        "model.layer0.mlp.up_proj.weight": torch.randn(
            256, 256, dtype=torch.float32
        ).to(torch.float8_e4m3fn),
        "model.layer0.mlp.up_proj.weight_scale_inv": torch.randn(
            num_row_blocks, num_col_blocks, dtype=torch.float32
        ),
        "model.layer1.mlp.down_proj.weight": torch.randn(
            256, 256, dtype=torch.float32
        ).to(torch.float8_e4m3fn),
        "model.layer1.mlp.down_proj.weight_scale_inv": torch.randn(
            num_row_blocks, num_col_blocks, dtype=torch.float32
        ),
        "model.embed_tokens.weight": non_targeted_weight,
    }

    # Save references to original tensors before processing
    weight_fp8_layer0 = tensors["model.layer0.mlp.up_proj.weight"].clone()
    scale_inv_layer0 = tensors["model.layer0.mlp.up_proj.weight_scale_inv"].clone()
    weight_fp8_layer1 = tensors["model.layer1.mlp.down_proj.weight"].clone()
    scale_inv_layer1 = tensors["model.layer1.mlp.down_proj.weight_scale_inv"].clone()

    # Process the tensors
    converter.process(tensors)

    # Verify that weight_scale_inv tensors were removed
    assert (
        "model.layer0.mlp.up_proj.weight_scale_inv" not in tensors
    ), "weight_scale_inv should be removed"
    assert (
        "model.layer1.mlp.down_proj.weight_scale_inv" not in tensors
    ), "weight_scale_inv should be removed"

    # Verify that weights were converted to bfloat16
    assert "model.layer0.mlp.up_proj.weight" in tensors, "weight should still exist"
    assert "model.layer1.mlp.down_proj.weight" in tensors, "weight should still exist"

    # Verify the conversion is correct using helper
    _verify_block_conversion(
        tensors["model.layer0.mlp.up_proj.weight"],
        weight_fp8_layer0,
        scale_inv_layer0,
        (128, 128),
    )
    _verify_block_conversion(
        tensors["model.layer1.mlp.down_proj.weight"],
        weight_fp8_layer1,
        scale_inv_layer1,
        (128, 128),
    )

    # Verify non-targeted tensor was not modified
    assert torch.equal(
        tensors["model.embed_tokens.weight"], non_targeted_weight
    ), "Non-targeted tensor should not be modified"


def _verify_block_conversion(
    result: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: tuple[int, int],
):
    """
    Helper method to verify that FP8 block conversion to bfloat16 is correct.
    Checks that each block is correctly scaled by its corresponding scale_inv value.
    """
    block_height, block_width = block_size
    num_row_blocks = weight_scale_inv.shape[0]
    num_col_blocks = weight_scale_inv.shape[1]

    # Verify output properties
    assert result.shape == weight_fp8.shape, "Output shape should match input shape"
    assert result.dtype == torch.bfloat16, "Output dtype should be bfloat16"

    # Verify the conversion logic: each block should be multiplied by its scale_inv
    for row_block in range(num_row_blocks):
        for col_block in range(num_col_blocks):
            row_start = row_block * block_height
            row_end = min((row_block + 1) * block_height, result.shape[0])
            col_start = col_block * block_width
            col_end = min((col_block + 1) * block_width, result.shape[1])

            # Get the block from result
            result_block = result[row_start:row_end, col_start:col_end]

            # Get expected: weight_fp8 block * scale_inv
            expected_block = (
                weight_fp8[row_start:row_end, col_start:col_end].to(torch.float32)
                * weight_scale_inv[row_block, col_block].to(torch.float32)
            ).to(torch.bfloat16)

            # They should be equal (within floating point precision)
            assert torch.allclose(
                result_block.to(torch.float32),
                expected_block.to(torch.float32),
                rtol=1e-2,
                atol=1e-3,
            ), f"Block ({row_block}, {col_block}) conversion mismatch"
