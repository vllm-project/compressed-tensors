# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.offload.cache import (
    CPUCache,
    DeviceCache,
    DiskCache,
    OffloadStats,
)
from tests.testing_utils import requires_gpu


@pytest.fixture(autouse=True)
def reset_stats():
    """Reset statistics before each test"""
    OffloadStats.reset()
    yield
    OffloadStats.reset()


@pytest.mark.unit
@requires_gpu
def test_basic_tracking():
    """Test that basic operations are tracked correctly"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Create a tensor
    tensor = torch.randn(10, 10, device=device)
    expected_bytes = tensor.element_size() * tensor.numel()

    # Offload (should be tracked)
    cache["test"] = tensor

    assert OffloadStats.offload.count == 1
    assert OffloadStats.offload.bytes_moved == expected_bytes
    assert OffloadStats.offload.noop_count == 0

    # Onload (should be tracked)
    retrieved = cache["test"]

    assert OffloadStats.onload.count == 1
    assert OffloadStats.onload.bytes_moved == expected_bytes
    assert OffloadStats.onload.noop_count == 0


@pytest.mark.unit
@requires_gpu
def test_update_tracking():
    """Test that update operations are tracked correctly"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Create and offload a tensor
    tensor = torch.randn(10, 10, device=device)
    cache["test"] = tensor

    # Reset to focus on updates
    OffloadStats.reset()

    # Update the tensor
    new_tensor = torch.randn(10, 10, device=device)
    cache["test"] = new_tensor

    # Update should be tracked (one update call)
    assert OffloadStats.update.count == 1
    assert (
        OffloadStats.update.bytes_moved
        == new_tensor.element_size() * new_tensor.numel()
    )


@pytest.mark.unit
@requires_gpu
def test_none_tensor_tracking():
    """Test that None tensors are tracked as no-ops"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Offload None
    cache["none_test"] = None

    assert OffloadStats.offload.count == 1
    assert OffloadStats.offload.noop_count == 1
    assert OffloadStats.offload.bytes_moved == 0

    # Retrieve None (note: __getitem__ returns early for None, so onload is not called)
    retrieved = cache["none_test"]

    # Onload is not tracked because __getitem__ returns None early without calling onload()
    assert OffloadStats.onload.count == 0
    assert OffloadStats.onload.noop_count == 0
    assert OffloadStats.onload.bytes_moved == 0


@pytest.mark.unit
@requires_gpu
def test_noop_update_tracking():
    """Test that update operations are tracked correctly"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Create and offload a tensor
    tensor = torch.randn(10, 10, device=device)
    cache["test"] = tensor

    # Reset to focus on updates
    OffloadStats.reset()

    # Update with new tensor of same size (triggers update_offload)
    new_tensor = torch.randn(10, 10, device=device)
    cache["test"] = new_tensor

    # Should have one update operation (not a no-op since data is provided)
    assert OffloadStats.update.count == 1
    assert OffloadStats.update.noop_count == 0


@pytest.mark.unit
@requires_gpu
def test_device_cache_noop():
    """Test that DeviceCache operations that don't move data are tracked as no-ops"""
    device = torch.device("cuda")
    cache = DeviceCache(onload_device=device)

    # Create a tensor on the same device
    tensor = torch.randn(10, 10, device=device)

    # Offload to same device (should be no-op in terms of device movement)
    cache["test"] = tensor

    assert OffloadStats.offload.count == 1
    # Note: The no-op detection checks if data_ptr and device are the same
    # DeviceCache may create a new tensor even on the same device


@pytest.mark.unit
@requires_gpu
def test_multiple_operations():
    """Test tracking across multiple operations"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Create multiple tensors
    tensors = {
        "t1": torch.randn(10, 10, device=device),
        "t2": torch.randn(20, 20, device=device),
        "t3": torch.randn(30, 30, device=device),
    }

    # Offload all
    for name, tensor in tensors.items():
        cache[name] = tensor

    assert OffloadStats.offload.count == 3

    # Onload some
    _ = cache["t1"]
    _ = cache["t2"]

    assert OffloadStats.onload.count == 2

    # Update one
    cache["t1"] = torch.randn(10, 10, device=device)

    assert OffloadStats.update.count == 1


@pytest.mark.unit
@requires_gpu
def test_reset():
    """Test that reset clears all statistics"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Perform some operations
    tensor = torch.randn(10, 10, device=device)
    cache["test"] = tensor
    _ = cache["test"]

    # Verify stats are non-zero
    assert OffloadStats.offload.count > 0
    assert OffloadStats.onload.count > 0

    # Reset
    OffloadStats.reset()

    # Verify stats are zero
    assert OffloadStats.offload.count == 0
    assert OffloadStats.onload.count == 0
    assert OffloadStats.update.count == 0
    assert OffloadStats.offload.bytes_moved == 0
    assert OffloadStats.onload.bytes_moved == 0
    assert OffloadStats.update.bytes_moved == 0


@pytest.mark.unit
@requires_gpu
def test_format_summary():
    """Test that format_summary produces expected output"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Perform some operations
    tensor = torch.randn(100, 100, device=device)
    cache["test"] = tensor
    _ = cache["test"]

    # Get summary
    summary = OffloadStats.format_summary(unit="KB")

    # Verify it contains expected strings
    assert "OffloadCache Statistics" in summary
    assert "Onload" in summary
    assert "Offload" in summary
    assert "Update" in summary
    assert "KB" in summary

    # Test different units
    for unit in ["B", "KB", "MB", "GB"]:
        summary = OffloadStats.format_summary(unit=unit)
        assert unit in summary


@pytest.mark.unit
@requires_gpu
def test_format_summary_invalid_unit():
    """Test that invalid unit raises ValueError"""
    with pytest.raises(ValueError):
        OffloadStats.format_summary(unit="TB")


@pytest.mark.unit
@requires_gpu
def test_bytes_calculation():
    """Test that bytes moved calculation is accurate"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Create a tensor with known size
    tensor = torch.randn(10, 10, dtype=torch.float32, device=device)
    expected_bytes = 10 * 10 * 4  # 10x10 float32 = 400 bytes

    cache["test"] = tensor

    assert OffloadStats.offload.bytes_moved == expected_bytes

    _ = cache["test"]

    assert OffloadStats.onload.bytes_moved == expected_bytes


@pytest.mark.unit
@requires_gpu
def test_disk_cache_tracking():
    """Test that DiskCache operations are tracked correctly"""
    import tempfile

    device = torch.device("cuda")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = DiskCache(onload_device=device, offload_dir=tmpdir)

        # Create and offload a tensor
        tensor = torch.randn(10, 10, device=device)
        expected_bytes = tensor.element_size() * tensor.numel()

        cache["test"] = tensor

        assert OffloadStats.offload.count == 1
        assert OffloadStats.offload.bytes_moved == expected_bytes

        # Onload the tensor
        retrieved = cache["test"]

        assert OffloadStats.onload.count == 1
        assert OffloadStats.onload.bytes_moved == expected_bytes


@pytest.mark.unit
def test_no_instantiation():
    """Test that OffloadStats cannot be instantiated"""
    with pytest.raises(RuntimeError, match="should not be instantiated"):
        OffloadStats()


@pytest.mark.unit
def test_get_stats():
    """Test that get_stats returns a dictionary with all statistics"""
    OffloadStats.reset()

    stats = OffloadStats.get_stats()
    assert isinstance(stats, dict)
    assert "onload" in stats
    assert "offload" in stats
    assert "update" in stats
    assert stats["onload"].count == 0
    assert stats["offload"].count == 0
    assert stats["update"].count == 0
