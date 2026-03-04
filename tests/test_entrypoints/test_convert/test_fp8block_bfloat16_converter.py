# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.entrypoints.convert.converters.fp8block_bfloat16 import (
    FP8BlockToBfloat16Converter,
)


class TestFP8BlockToBfloat16Converter:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "block_size,weight_shape,expected_behavior,verify_correctness",
        [
            # Evenly divisible by block size
            ([128, 128], (256, 256), "basic", False),
            # Not evenly divisible by block size
            ([128, 128], (200, 300), "basic", False),
            # Smaller than block size
            ([128, 128], (64, 96), "basic", False),
            # No block size (non-block quantization)
            (None, (200, 300), "no_block", False),
            # Dequantization correctness test
            ([128, 128], (128, 128), "correctness", True),
            # Multiple blocks with different scales
            ([128, 128], (256, 256), "multiple_blocks", True),
            # Various odd dimensions
            ([128, 128], (127, 127), "basic", False),
            ([128, 128], (129, 127), "basic", False),
            ([128, 128], (255, 257), "basic", False),
            ([128, 128], (127, 256), "basic", False),
        ],
    )
    def test_create_bfloat16_weight(
        self, block_size, weight_shape, expected_behavior, verify_correctness
    ):
        """
        Parameterized test for create_bfloat16_weight covering:
        - Evenly divisible dimensions
        - Non-evenly divisible dimensions (requires padding)
        - Dimensions smaller than block size
        - No block size (non-block quantization)
        - Mathematical correctness of dequantization
        - Multiple blocks with different scales
        """
        converter = FP8BlockToBfloat16Converter(weight_block_size=block_size)

        # Calculate number of blocks based on block_size
        if block_size is None:
            # Non-block quantization: scale_inv has same shape as weight
            num_blocks = weight_shape
            weight_scale_inv = torch.randn(num_blocks, dtype=torch.float32).abs() + 0.1
        else:
            # Block quantization: calculate ceiling division
            num_blocks = (
                (weight_shape[0] + block_size[0] - 1) // block_size[0],
                (weight_shape[1] + block_size[1] - 1) // block_size[1],
            )
            weight_scale_inv = (
                torch.randn(num_blocks, dtype=torch.float32).abs() + 0.1
            )

        # Create test data based on expected_behavior
        if expected_behavior == "correctness":
            # Simple test with known values for verification
            # weight_fp8 * weight_scale_inv = 2.0 * 0.5 = 1.0
            weight_fp8 = (torch.ones(weight_shape, dtype=torch.float32) * 2.0).to(
                torch.float8_e4m3fn
            )
            weight_scale_inv = torch.ones(num_blocks, dtype=torch.float32) * 0.5
        elif expected_behavior == "multiple_blocks":
            # Different values in each block - create as float32 first
            weight_f32 = torch.ones(weight_shape, dtype=torch.float32)
            weight_f32[: weight_shape[0] // 2, : weight_shape[1] // 2] *= 1.0
            weight_f32[: weight_shape[0] // 2, weight_shape[1] // 2 :] *= 2.0
            weight_f32[weight_shape[0] // 2 :, : weight_shape[1] // 2] *= 3.0
            weight_f32[weight_shape[0] // 2 :, weight_shape[1] // 2 :] *= 4.0
            weight_fp8 = weight_f32.to(torch.float8_e4m3fn)
            # Use reciprocal scales so multiplication gives 1.0
            # Block [0,0]: 1.0 * 1.0 = 1.0
            # Block [0,1]: 2.0 * 0.5 = 1.0
            # Block [1,0]: 3.0 * (1/3) = 1.0
            # Block [1,1]: 4.0 * 0.25 = 1.0
            weight_scale_inv = torch.tensor(
                [[1.0, 0.5], [1.0 / 3.0, 0.25]], dtype=torch.float32
            )
        else:
            # Random values - create as float32 then convert to fp8
            weight_fp8 = torch.randn(weight_shape, dtype=torch.float32).to(
                torch.float8_e4m3fn
            )

        # Convert to bfloat16
        result = converter.create_bfloat16_weight(weight_fp8, weight_scale_inv)

        # Basic assertions for all test cases
        assert (
            result.shape == weight_shape
        ), f"Shape mismatch: expected {weight_shape}, got {result.shape}"
        assert result.dtype == torch.bfloat16, f"Dtype mismatch: got {result.dtype}"

        # Additional verification for correctness tests
        if verify_correctness and expected_behavior == "correctness":
            # weight * scale_inv = 2.0 * 0.5 = 1.0
            expected = torch.ones(weight_shape, dtype=torch.bfloat16)
            assert torch.allclose(
                result, expected, rtol=1e-2, atol=1e-2
            ), "Dequantization correctness check failed"

        if verify_correctness and expected_behavior == "multiple_blocks":
            # Verify each block was scaled correctly (all should equal 1.0)
            block_h, block_w = block_size
            assert torch.allclose(
                result[:block_h, :block_w],
                torch.ones(block_h, block_w, dtype=torch.bfloat16),
                rtol=1e-2,
                atol=1e-2,
            ), "Block [0,0] scaling failed"
            assert torch.allclose(
                result[:block_h, block_w:],
                torch.ones(block_h, block_w, dtype=torch.bfloat16),
                rtol=1e-2,
                atol=1e-2,
            ), "Block [0,1] scaling failed"
            assert torch.allclose(
                result[block_h:, :block_w],
                torch.ones(block_h, block_w, dtype=torch.bfloat16),
                rtol=1e-2,
                atol=1e-2,
            ), "Block [1,0] scaling failed"
            assert torch.allclose(
                result[block_h:, block_w:],
                torch.ones(block_h, block_w, dtype=torch.bfloat16),
                rtol=1e-2,
                atol=1e-2,
            ), "Block [1,1] scaling failed"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "targets,should_raise,match_text,case_name",
        [
            # Valid case
            (["layer"], False, None, "valid"),
            # Missing scale_inv
            (["layer"], True, "Found weight without corresponding weight_scale_inv", "missing_scale"),
            # Unexpected tensor name
            (["layer"], True, "Found unexpected targeted tensor", "unexpected"),
            # Non-targeted layer with scale_inv
            (["layer1"], True, "Found unexpected non-targeted tensor", "non_targeted"),
        ],
    )
    def test_validate(self, targets, should_raise, match_text, case_name):
        """
        Parameterized test for validation covering:
        - Valid tensors
        - Missing weight_scale_inv
        - Unexpected tensor names
        - Non-targeted layers with disallowed tensors
        """
        converter = FP8BlockToBfloat16Converter(
            weight_block_size=[128, 128], targets=targets
        )

        # Create tensors based on case_name
        match case_name:
            case "valid":
                tensors = {
                    "layer.weight": torch.randn((128, 128), dtype=torch.float32).to(
                        torch.float8_e4m3fn
                    ),
                    "layer.weight_scale_inv": torch.randn((1, 1), dtype=torch.float32),
                }
            case "missing_scale":
                tensors = {
                    "layer.weight": torch.randn((128, 128), dtype=torch.float32).to(
                        torch.float8_e4m3fn
                    ),
                }
            case "unexpected":
                tensors = {
                    "layer.weight": torch.randn((128, 128), dtype=torch.float32).to(
                        torch.float8_e4m3fn
                    ),
                    "layer.weight_scale_inv": torch.randn((1, 1), dtype=torch.float32),
                    "layer.unexpected_param": torch.randn((128, 128)),
                }
            case "non_targeted":
                tensors = {
                    "layer1.weight": torch.randn((128, 128), dtype=torch.float32).to(
                        torch.float8_e4m3fn
                    ),
                    "layer1.weight_scale_inv": torch.randn((1, 1), dtype=torch.float32),
                    "layer2.weight": torch.randn((128, 128)),
                    "layer2.weight_scale_inv": torch.randn((1, 1), dtype=torch.float32),
                }

        if should_raise:
            with pytest.raises(ValueError, match=match_text):
                converter.validate(tensors)
        else:
            # Should not raise
            converter.validate(tensors)

    @pytest.mark.unit
    def test_process_full_workflow(self):
        """Test the full process method to ensure it integrates correctly"""
        block_size = [128, 128]
        converter = FP8BlockToBfloat16Converter(
            weight_block_size=block_size, targets=["layer"]
        )

        # Create mock tensors dictionary
        weight_shape = (256, 256)
        num_blocks = (2, 2)

        tensors = {
            "layer.weight": torch.randn(weight_shape, dtype=torch.float32).to(
                torch.float8_e4m3fn
            ),
            "layer.weight_scale_inv": torch.randn(num_blocks, dtype=torch.float32).abs()
            + 0.1,
        }

        # Process the tensors
        converter.process(tensors)

        # Check that weight was converted and scale_inv was removed
        assert "layer.weight" in tensors
        assert "layer.weight_scale_inv" not in tensors
        assert tensors["layer.weight"].dtype == torch.bfloat16
        assert tensors["layer.weight"].shape == weight_shape
