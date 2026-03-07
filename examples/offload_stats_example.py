#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example demonstrating the usage of OffloadStats to track device movement.

This example shows how to:
1. Enable/disable statistics collection
2. Use OffloadCache with automatic statistics tracking
3. Reset statistics
4. Get statistics summary
5. Track bytes moved and no-ops
"""

import torch
from compressed_tensors.offload.cache import CPUCache, OffloadStats


def main():
    # Create some example tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor1 = torch.randn(100, 100, device=device)
    tensor2 = torch.randn(50, 50, device=device)
    tensor3 = torch.randn(200, 200, device=device)

    print("=" * 60)
    print("Offload Statistics Example")
    print("=" * 60)
    print()

    # Enable statistics collection (disabled by default for performance)
    print("Enabling statistics collection...")
    OffloadStats.enable()
    print(f"Statistics enabled: {OffloadStats.enabled}")
    print()

    # Reset statistics to start fresh
    OffloadStats.reset()

    # Create a CPU cache that offloads to CPU and onloads back to device
    cache = CPUCache(onload_device=device)

    print(f"Device: {device}")
    print(f"Tensor 1 size: {tensor1.numel()} elements")
    print(f"Tensor 2 size: {tensor2.numel()} elements")
    print(f"Tensor 3 size: {tensor3.numel()} elements")
    print()

    # Offload tensors to cache (triggers offload tracking)
    print("Offloading tensors to CPU...")
    cache["tensor1"] = tensor1
    cache["tensor2"] = tensor2
    cache["tensor3"] = tensor3
    print()

    # Onload tensors from cache (triggers onload tracking)
    print("Onloading tensors back to device...")
    retrieved1 = cache["tensor1"]
    retrieved2 = cache["tensor2"]
    print()

    # Update a tensor (triggers update tracking)
    print("Updating tensor1...")
    new_data = torch.randn_like(tensor1)
    cache["tensor1"] = new_data
    print()

    # Onload the updated tensor
    print("Onloading updated tensor...")
    retrieved1_updated = cache["tensor1"]
    print()

    # Test with None tensor (should be tracked as no-op)
    print("Testing with None tensor (no-op)...")
    cache["none_tensor"] = None
    retrieved_none = cache["none_tensor"]
    print()

    # Display statistics
    print(OffloadStats.format_summary(unit="KB"))
    print()

    # Display statistics with device breakdown
    print(OffloadStats.format_summary(unit="KB", show_devices=True))
    print()

    # Get raw statistics for programmatic access
    print("Raw statistics:")
    print(f"  Onload operations: {OffloadStats.onload.count}")
    print(f"  Offload operations: {OffloadStats.offload.count}")
    print(f"  Update operations: {OffloadStats.update.count}")
    print(f"  Total no-ops: {OffloadStats.onload.noop_count + OffloadStats.offload.noop_count + OffloadStats.update.noop_count}")
    print()

    # Get device-specific statistics
    print("Device-specific statistics:")
    device_stats = OffloadStats.get_device_stats()
    for op_name, movements in device_stats.items():
        if movements:
            print(f"  {op_name.capitalize()}:")
            for (src, dst), stats in movements.items():
                print(
                    f"    {src} -> {dst}: {stats['count']} ops "
                    f"({stats['noop_count']} no-ops), "
                    f"{stats['bytes_moved']} bytes moved, "
                    f"{stats['noop_bytes']} bytes in no-ops"
                )
    print()

    # Reset and show that statistics are cleared
    print("Resetting statistics...")
    OffloadStats.reset()
    print()
    print(OffloadStats.format_summary(unit="MB"))
    print()

    # Perform more operations and show updated stats
    print("Performing more operations...")
    cache["new_tensor"] = torch.randn(300, 300, device=device)
    _ = cache["new_tensor"]
    print()
    print(OffloadStats.format_summary(unit="MB"))
    print()

    # Demonstrate disabling statistics collection
    print("Disabling statistics collection...")
    OffloadStats.disable()
    print(f"Statistics enabled: {OffloadStats.enabled}")
    OffloadStats.reset()
    print()

    # Operations performed while disabled should not be tracked
    print("Performing operations while stats are disabled...")
    cache["disabled_tensor"] = torch.randn(100, 100, device=device)
    _ = cache["disabled_tensor"]
    print()
    print(OffloadStats.format_summary(unit="MB"))
    print("(Notice: no new operations tracked)")


if __name__ == "__main__":
    main()
