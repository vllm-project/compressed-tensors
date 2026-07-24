# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8
from compressed_tensors.quantization.lifecycle.forward_helpers import QuantBufferPool


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="CUDA not available")
class TestQuantBufferPoolCUDA:
    """CUDA-specific tests for QuantBufferPool."""

    def setup_method(self):
        QuantBufferPool.clear()

    def teardown_method(self):
        QuantBufferPool.clear()

    def test_cuda_basic(self):
        """Buffer works on CUDA device."""
        shape = (100, 100)
        dtype = torch.float32
        device = torch.device("cuda:0")

        buf = QuantBufferPool.get_buffer(shape, dtype, device)

        assert buf.shape == shape
        assert buf.dtype == dtype
        assert buf.is_cuda

    def test_cuda_buffer_reuse(self):
        """CUDA buffer is reused for same size requests."""
        shape = (100, 100)
        dtype = torch.float32
        device = torch.device("cuda:0")

        buf1 = QuantBufferPool.get_buffer(shape, dtype, device)
        ptr1 = buf1.data_ptr()

        buf2 = QuantBufferPool.get_buffer(shape, dtype, device)
        ptr2 = buf2.data_ptr()

        assert ptr1 == ptr2, "CUDA buffer should be reused"

    def test_separate_pools_per_device(self):
        """CPU and CUDA have separate pools."""
        shape = (100, 100)
        dtype = torch.float32

        cpu_buf = QuantBufferPool.get_buffer(shape, dtype, torch.device("cpu"))
        cuda_buf = QuantBufferPool.get_buffer(shape, dtype, torch.device("cuda:0"))

        assert not cpu_buf.is_cuda
        assert cuda_buf.is_cuda
        # Different devices should have different pools
        assert cpu_buf.data_ptr() != cuda_buf.data_ptr()

    def test_pack_fp4_with_vs_without_buffer_pool(self):
        """pack_fp4_to_uint8 produces identical results with and without buffer pool."""
        # Create random FP4 indices (0-15 representing the 16 FP4 values)
        shape = (4096, 4096)
        x = torch.randint(0, 16, shape, dtype=torch.uint8, device="cuda")

        # Pack without buffer pool
        packed_without = pack_fp4_to_uint8(x, use_buffer_pool=False)

        # Pack with buffer pool
        QuantBufferPool.clear()
        packed_with = pack_fp4_to_uint8(x, use_buffer_pool=True)

        assert torch.equal(packed_without, packed_with), (
            "pack_fp4_to_uint8 should produce identical results "
            "with and without buffer pool"
        )
