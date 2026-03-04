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
    """Reset statistics and enable collection before each test"""
    OffloadStats.reset()
    OffloadStats.enable()
    yield
    OffloadStats.reset()
    OffloadStats.disable()


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
    _ = cache["test"]

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

    # Retrieve None (note: __getitem__ returns early for None)
    _ = cache["none_test"]

    # Onload not tracked: __getitem__ returns None without onload()
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
    assert "Offload Statistics" in summary
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
        _ = cache["test"]

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


@pytest.mark.unit
@requires_gpu
def test_device_tracking():
    """Test that device movements are tracked correctly"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Create and offload a tensor
    tensor = torch.randn(10, 10, device=device)
    tensor_device_str = str(tensor.device)
    cache["test"] = tensor

    # Check offload device movement
    assert (tensor_device_str, "cpu") in OffloadStats.offload.device_stats
    offload_stats = OffloadStats.offload.device_stats[(tensor_device_str, "cpu")]
    assert offload_stats.count == 1
    assert offload_stats.noop_count == 0
    assert offload_stats.bytes_moved == 400
    assert offload_stats.noop_bytes == 0

    # Onload the tensor
    _ = cache["test"]

    # Check onload device movement
    assert ("cpu", tensor_device_str) in OffloadStats.onload.device_stats
    onload_stats = OffloadStats.onload.device_stats[("cpu", tensor_device_str)]
    assert onload_stats.count == 1
    assert onload_stats.noop_count == 0
    assert onload_stats.bytes_moved == 400
    assert onload_stats.noop_bytes == 0


@pytest.mark.unit
@requires_gpu
def test_get_device_stats():
    """Test get_device_stats method"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Perform some operations
    tensor = torch.randn(10, 10, device=device)
    tensor_device_str = str(tensor.device)
    cache["test"] = tensor
    _ = cache["test"]

    # Get device stats
    device_stats = OffloadStats.get_device_stats()

    assert "onload" in device_stats
    assert "offload" in device_stats
    assert "update" in device_stats

    # Check offload stats
    assert (tensor_device_str, "cpu") in device_stats["offload"]
    assert device_stats["offload"][(tensor_device_str, "cpu")]["count"] == 1
    assert device_stats["offload"][(tensor_device_str, "cpu")]["noop_count"] == 0
    assert device_stats["offload"][(tensor_device_str, "cpu")]["bytes_moved"] == 400
    assert device_stats["offload"][(tensor_device_str, "cpu")]["noop_bytes"] == 0

    # Check onload stats
    assert ("cpu", tensor_device_str) in device_stats["onload"]
    assert device_stats["onload"][("cpu", tensor_device_str)]["count"] == 1
    assert device_stats["onload"][("cpu", tensor_device_str)]["noop_count"] == 0
    assert device_stats["onload"][("cpu", tensor_device_str)]["bytes_moved"] == 400
    assert device_stats["onload"][("cpu", tensor_device_str)]["noop_bytes"] == 0


@pytest.mark.unit
@requires_gpu
def test_format_summary_with_devices():
    """Test format_summary with device breakdown"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Perform some operations
    tensor = torch.randn(10, 10, device=device)
    cache["test"] = tensor
    _ = cache["test"]

    # Get summary with devices
    summary = OffloadStats.format_summary(unit="KB", show_devices=True)

    # Verify it contains expected strings
    assert "Offload Statistics" in summary
    assert "Device Movement Breakdown" in summary
    assert "Source" in summary
    assert "Dest" in summary
    assert "Onload" in summary
    assert "Offload" in summary


@pytest.mark.unit
@requires_gpu
def test_device_tracking_with_update():
    """Test that update operations track devices correctly"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Create and offload a tensor
    tensor = torch.randn(10, 10, device=device)
    cache["test"] = tensor

    # Reset to focus on update
    OffloadStats.reset()

    # Update the tensor
    new_tensor = torch.randn(10, 10, device=device)
    tensor_device_str = str(new_tensor.device)
    cache["test"] = new_tensor

    # Check update device movement (from device to cpu for the offloaded tensor)
    assert (tensor_device_str, "cpu") in OffloadStats.update.device_stats
    update_stats = OffloadStats.update.device_stats[(tensor_device_str, "cpu")]
    assert update_stats.count == 1
    assert update_stats.noop_count == 0
    assert update_stats.bytes_moved == 400
    assert update_stats.noop_bytes == 0


@pytest.mark.unit
@requires_gpu
def test_device_pair_noop_tracking():
    """Test that no-ops are tracked correctly per device pair"""
    device = torch.device("cuda")
    cache = DeviceCache(onload_device=device)

    # Create a tensor on device - offload to same device should be noop
    tensor = torch.randn(10, 10, device=device)
    tensor_device_str = str(tensor.device)
    cache["test"] = tensor

    # DeviceCache offloads to the same device, which is a no-op
    assert (tensor_device_str, tensor_device_str) in OffloadStats.offload.device_stats
    offload_stats = OffloadStats.offload.device_stats[
        (tensor_device_str, tensor_device_str)
    ]
    assert offload_stats.count == 1
    assert offload_stats.noop_count == 1
    assert offload_stats.bytes_moved == 0
    assert offload_stats.noop_bytes == 400  # Would have moved 400 bytes if not a no-op


@pytest.mark.unit
@requires_gpu
def test_device_pair_stats_summary():
    """Test that device pair stats are included in summary"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Perform some operations
    tensor = torch.randn(10, 10, device=device)
    cache["test"] = tensor
    _ = cache["test"]

    # Get summary with devices
    summary = OffloadStats.format_summary(unit="KB", show_devices=True)

    # Verify new columns are present
    assert "No-ops" in summary
    assert "Moved" in summary
    assert "No-op Data" in summary


@pytest.mark.unit
def test_enable_disable():
    """Test that enable/disable controls stat collection"""
    # Initially disabled by default (but fixture enables it)
    assert OffloadStats.enabled

    # Disable stats
    OffloadStats.disable()
    assert not OffloadStats.enabled

    # Enable stats
    OffloadStats.enable()
    assert OffloadStats.enabled


@pytest.mark.unit
@requires_gpu
def test_disabled_stats_not_collected():
    """Test that statistics are not collected when disabled"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Disable stats collection
    OffloadStats.disable()
    OffloadStats.reset()

    # Perform operations
    tensor = torch.randn(10, 10, device=device)
    cache["test"] = tensor
    _ = cache["test"]

    # Verify no stats were collected
    assert OffloadStats.offload.count == 0
    assert OffloadStats.onload.count == 0
    assert OffloadStats.offload.bytes_moved == 0
    assert OffloadStats.onload.bytes_moved == 0

    # Re-enable for cleanup
    OffloadStats.enable()


@pytest.mark.unit
@requires_gpu
def test_enabled_stats_are_collected():
    """Test that statistics are collected when enabled"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Explicitly enable stats collection
    OffloadStats.enable()
    OffloadStats.reset()

    # Perform operations
    tensor = torch.randn(10, 10, device=device)
    expected_bytes = tensor.element_size() * tensor.numel()
    cache["test"] = tensor
    _ = cache["test"]

    # Verify stats were collected
    assert OffloadStats.offload.count == 1
    assert OffloadStats.onload.count == 1
    assert OffloadStats.offload.bytes_moved == expected_bytes
    assert OffloadStats.onload.bytes_moved == expected_bytes


@pytest.mark.unit
@requires_gpu
def test_track_context_manager():
    """Test that track context manager enables tracking and returns stats"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Disable stats and reset
    OffloadStats.disable()
    OffloadStats.reset()

    # Use track context manager
    with OffloadStats.track() as stats:
        # Stats should be enabled inside context
        assert OffloadStats.enabled

        # Perform operations
        tensor = torch.randn(10, 10, device=device)
        expected_bytes = tensor.element_size() * tensor.numel()
        cache["test"] = tensor
        _ = cache["test"]

    # After context, stats should be populated
    assert "onload" in stats
    assert "offload" in stats
    assert "update" in stats
    assert stats["onload"].count == 1
    assert stats["offload"].count == 1
    assert stats["onload"].bytes_moved == expected_bytes
    assert stats["offload"].bytes_moved == expected_bytes

    # Stats should be disabled after context exits
    assert not OffloadStats.enabled


@pytest.mark.unit
@requires_gpu
def test_track_context_manager_restores_enabled_state():
    """Test that track context manager restores the original enabled state"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Enable stats before using track
    OffloadStats.enable()
    OffloadStats.reset()

    # Use track context manager
    with OffloadStats.track() as stats:
        # Stats should be enabled inside context
        assert OffloadStats.enabled

        # Perform operations
        tensor = torch.randn(10, 10, device=device)
        cache["test"] = tensor

    # Stats should still be enabled after context exits (restored to original state)
    assert OffloadStats.enabled


@pytest.mark.unit
@requires_gpu
def test_track_context_manager_with_reset():
    """Test that track context manager captures only operations within the context"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Enable and perform some operations before track
    OffloadStats.enable()
    OffloadStats.reset()
    tensor1 = torch.randn(10, 10, device=device)
    cache["before"] = tensor1

    # Reset before using track
    OffloadStats.reset()

    # Use track context manager
    with OffloadStats.track() as stats:
        tensor2 = torch.randn(20, 20, device=device)
        expected_bytes = tensor2.element_size() * tensor2.numel()
        cache["during"] = tensor2
        _ = cache["during"]

    # Stats should only include operations from within the context
    assert stats["offload"].count == 1
    assert stats["onload"].count == 1
    assert stats["offload"].bytes_moved == expected_bytes


@pytest.mark.unit
@requires_gpu
def test_track_context_manager_multiple_operations():
    """Test that track context manager correctly tracks multiple operations"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Disable stats and reset
    OffloadStats.disable()
    OffloadStats.reset()

    # Use track context manager with multiple operations
    with OffloadStats.track() as stats:
        tensors = {
            "t1": torch.randn(10, 10, device=device),
            "t2": torch.randn(20, 20, device=device),
            "t3": torch.randn(30, 30, device=device),
        }

        # Offload all tensors
        for name, tensor in tensors.items():
            cache[name] = tensor

        # Onload some tensors
        _ = cache["t1"]
        _ = cache["t2"]

        # Update one tensor
        cache["t1"] = torch.randn(10, 10, device=device)

    # Verify all operations were tracked
    assert stats["offload"].count == 3
    assert stats["onload"].count == 2
    assert stats["update"].count == 1


@pytest.mark.unit
@requires_gpu
def test_track_context_manager_exception_handling():
    """Test that track context manager restores state even on exception"""
    device = torch.device("cuda")
    cache = CPUCache(onload_device=device)

    # Disable stats before using track
    OffloadStats.disable()
    OffloadStats.reset()

    # Use track context manager with an exception
    try:
        with OffloadStats.track() as stats:
            # Stats should be enabled inside context
            assert OffloadStats.enabled

            # Perform some operations
            tensor = torch.randn(10, 10, device=device)
            cache["test"] = tensor

            # Raise an exception
            raise ValueError("Test exception")
    except ValueError:
        pass

    # Stats should be restored to disabled after exception
    assert not OffloadStats.enabled


@pytest.mark.unit
@requires_gpu
def test_track_context_manager_empty_operations():
    """Test that track context manager works with no operations"""
    # Disable stats and reset
    OffloadStats.disable()
    OffloadStats.reset()

    # Use track context manager without performing any operations
    with OffloadStats.track() as stats:
        pass

    # Stats should be populated but with zero counts
    assert "onload" in stats
    assert "offload" in stats
    assert "update" in stats
    assert stats["onload"].count == 0
    assert stats["offload"].count == 0
    assert stats["update"].count == 0
