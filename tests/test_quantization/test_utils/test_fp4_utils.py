# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization.utils.fp4_utils import (
    cast_to_fp4_torch,
    cast_to_fp4_triton,
)
from tests.testing_utils import requires_gpu


@requires_gpu
@pytest.mark.parametrize("size", [1, 10, 100, 1000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cast_to_fp4_cpu_gpu_match(size, dtype):
    # Create random tensor
    x_cpu = torch.randn(size, dtype=dtype)
    x_gpu = x_cpu.cuda()

    # Quantize on CPU and GPU
    result_cpu = cast_to_fp4_torch(x_cpu)
    result_gpu = cast_to_fp4_triton(x_gpu)

    # Compare outputs (convert to same dtype for comparison)
    assert torch.allclose(result_cpu.cuda(), result_gpu, atol=1e-6)


@requires_gpu
def test_cast_to_fp4_boundary_values():
    input_values = torch.tensor(
        [
            # Exact FP4 values
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
            # Values at boundaries
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            -0.25,
            -0.75,
            -1.25,
            -1.75,
            -2.5,
            -3.5,
            -5.0,
            # Values between boundaries
            0.3,
            0.6,
            0.9,
            1.3,
            1.8,
            2.7,
            4.5,
            7.0,
            -0.3,
            -0.6,
            -0.9,
            -1.3,
            -1.8,
            -2.7,
            -4.5,
            -7.0,
        ],
        device="cuda",
    )

    expected_output = torch.tensor(
        [
            # Exact FP4 values
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
            # Values at boundaries
            0.0,
            1.0,
            1.0,
            2.0,
            2.0,
            4.0,
            4.0,
            -0.0,
            -1.0,
            -1.0,
            -2.0,
            -2.0,
            -4.0,
            -4.0,
            # Values between boundaries
            0.5,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.5,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device="cuda",
    )

    result = cast_to_fp4_triton(input_values)
    assert torch.allclose(result, expected_output, atol=1e-6)


@requires_gpu
@pytest.mark.parametrize("size", [1024, 10240, 102400, 1024000])
def test_cast_to_fp4_memory_usage(size):
    """Test that peak memory usage is reasonable for large tensors.

    The implementation should not create excessive intermediate tensors.
    Expected memory usage should be roughly: input + output + small overhead.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create input tensor
    x = torch.randn(size, dtype=torch.float32, device="cuda")
    input_memory = x.element_size() * x.numel()

    # Record baseline memory after input creation
    baseline_memory = torch.cuda.memory_allocated()

    # Perform quantization
    result = cast_to_fp4_triton(x)
    output_memory = result.element_size() * result.numel()

    # Check peak memory usage
    peak_memory = torch.cuda.max_memory_allocated()
    actual_overhead = peak_memory - baseline_memory - output_memory

    # Expected overhead: allow up to 20% extra for intermediate computations
    # This is generous to account for Triton kernel overhead
    max_allowed_overhead = 0.2 * (input_memory + output_memory)

    assert actual_overhead <= max_allowed_overhead, (
        f"Memory overhead too high for size {size}. "
        f"Input: {input_memory / 1024**2:.2f} MB, "
        f"Output: {output_memory / 1024**2:.2f} MB, "
        f"Actual overhead: {actual_overhead / 1024**2:.2f} MB, "
        f"Max allowed overhead: {max_allowed_overhead / 1024**2:.2f} MB"
    )

    # Clean up
    del x, result
    torch.cuda.empty_cache()
