# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.utils.impl_backend import ImplBackend
from tests.testing_utils import requires_gpu


@requires_gpu
@pytest.mark.parametrize("dtype, log_elements", [
    (torch.float32, 22), # these were the fastest settings
    (torch.float16, 16),
    (torch.bfloat16, 16),
])
def test_cast_to_fp4_cpu_gpu_match(dtype, log_elements):
    # check every possible value in 2**log_elements chunks, (about 15 seconds total)
    bits = 16 if dtype in [torch.float16, torch.bfloat16] else 32
    num_loops = 2**(bits - log_elements)
    elements = torch.arange(2**log_elements, dtype=torch.int32)
    for i in range(num_loops):
        x_cpu = (i << log_elements | elements).view(dtype)
        x_cpu[x_cpu.isnan()] = 0.0

        x_gpu = x_cpu.cuda()

        # Quantize on CPU and GPU
        result_cpu = ImplBackend.call("cast_to_fp4", x_cpu).cuda()
        result_gpu = ImplBackend.call("cast_to_fp4_triton", x_gpu)

        # Compare outputs (convert to same dtype for comparison)
        assert torch.equal(result_cpu, result_gpu)


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
            -0.0,
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
            # Regression: fp32 values near boundaries that a bf16-casting
            # kernel would snap onto the boundary and round the wrong way
            0.2501,
            0.7499,
            1.2501,
            1.7499,
            2.501,
            3.499,
            5.001,
            -0.2501,
            -0.7499,
            -1.2501,
            -1.7499,
            -2.501,
            -3.499,
            -5.001,
        ],
        dtype=torch.float32,
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
            -0.0,
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
            # Regression: expected fp32 near-boundary values
            # These are slightly past the boundary in fp32 but snap onto
            # it in bf16, so a bf16-casting kernel gets them wrong
            0.5,
            0.5,
            1.5,
            1.5,
            3.0,
            3.0,
            6.0,
            -0.5,
            -0.5,
            -1.5,
            -1.5,
            -3.0,
            -3.0,
            -6.0,
        ],
        dtype=torch.float32,
        device="cuda",
    )

    result = ImplBackend.call("cast_to_fp4_triton", input_values)
    assert torch.equal(result, expected_output), (
        f"Mismatch at indices: "
        f"{(result != expected_output).nonzero(as_tuple=True)[0].tolist()}\n"
        f"Got:      {result[result != expected_output].tolist()}\n"
        f"Expected: {expected_output[result != expected_output].tolist()}"
    )
    # Note: Triton kernel does not preserve -0.0 sign bit (becomes +0.0).
    # This is acceptable since -0.0 == +0.0 mathematically.


@requires_gpu
@pytest.mark.parametrize("size", [1024, 10240, 102400, 1024000])
def test_cast_to_fp4_memory_usage(size):
    """Test that peak memory usage is reasonable for large tensors.

    The implementation should not create excessive intermediate tensors.
    Expected memory usage should be roughly: input + output + small overhead.
    """
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats()

    # Create input tensor
    x = torch.randn(size, dtype=torch.float32, device="cuda")
    input_memory = x.element_size() * x.numel()

    # Record baseline memory after input creation
    baseline_memory = torch.accelerator.memory_allocated()

    # Perform quantization
    result = ImplBackend.call("cast_to_fp4_triton", x)
    output_memory = result.element_size() * result.numel()

    # Check peak memory usage
    peak_memory = torch.accelerator.max_memory_allocated()
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
    torch.accelerator.empty_cache()


@requires_gpu
@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),
        torch.tensor([-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]),
        # midpoints between FP4 values to exercise rounding
        torch.tensor([0.3, 0.7, 1.2, 1.8, 2.6, 3.8, 5.5]),
        # 2-D tensor
        torch.arange(-6.0, 6.5, 0.5).reshape(5, -1).float(),
        # larger random tensor
        torch.randn(128, 128),
    ],
)
def test_cast_to_fp4_backends_match(x):
    x = x.to(torch.accelerator.current_accelerator())
    torch_out = ImplBackend.call("cast_to_fp4", x)
    triton_out = ImplBackend.call("cast_to_fp4_triton", x)

    assert torch_out.shape == triton_out.shape
    assert torch.allclose(
        torch_out, triton_out
    ), f"Max diff: {(torch_out - triton_out).abs().max().item()}"
