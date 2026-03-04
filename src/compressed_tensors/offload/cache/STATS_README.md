# Offload Statistics Tracking

The `OffloadStats` class provides automatic tracking of device movement operations for all `OffloadCache` subclasses.

## Features

- **Optional tracking**: Statistics collection is disabled by default to avoid runtime overhead. Use `enable()` to turn it on.
- **Automatic tracking**: All onload, offload, and update operations are tracked automatically via decorators when enabled
- **Bytes moved**: Tracks the total amount of data transferred for each operation type
- **No-op detection**: Identifies operations that don't actually move data (e.g., None tensors, same-device operations)
- **Global statistics**: All cache instances share the same statistics, providing a unified view of device movement

## Usage

### Enabling/Disabling Statistics Collection

**Important**: Statistics collection is **disabled by default** to avoid runtime overhead. You must explicitly enable it:

```python
from compressed_tensors.offload.cache import CPUCache, OffloadStats
import torch

# Enable statistics collection
OffloadStats.enable()

# Check if stats are enabled
print(f"Stats enabled: {OffloadStats.enabled}")  # True

# Disable statistics collection when done
OffloadStats.disable()
print(f"Stats enabled: {OffloadStats.enabled}")  # False
```

When statistics collection is disabled, operations are not tracked, reducing runtime overhead. This is the recommended state for production use unless you specifically need to analyze device movement.

### Basic Usage

After enabling statistics collection, operations are tracked automatically:

```python
from compressed_tensors.offload.cache import CPUCache, OffloadStats
import torch

# Enable statistics collection
OffloadStats.enable()

# Create a cache
device = torch.device("cuda")
cache = CPUCache(onload_device=device)

# Perform operations (automatically tracked when enabled)
tensor = torch.randn(100, 100, device=device)
cache["my_tensor"] = tensor  # Offload tracked
retrieved = cache["my_tensor"]  # Onload tracked

# View statistics
print(OffloadStats.format_summary())

# Disable when done
OffloadStats.disable()
```

### Accessing Raw Statistics

```python
# Access individual statistics directly
print(f"Onload count: {OffloadStats.onload.count}")
print(f"Offload bytes: {OffloadStats.offload.bytes_moved}")
print(f"Update no-ops: {OffloadStats.update.noop_count}")

# Or get all stats as a dictionary
stats = OffloadStats.get_stats()
for op_name, op_stats in stats.items():
    print(f"{op_name}: {op_stats.count} operations")
```

### Device-Specific Statistics

Get detailed information about data movement between devices:

```python
# Get device-specific statistics
device_stats = OffloadStats.get_device_stats()

for op_name, movements in device_stats.items():
    print(f"{op_name}:")
    for (src_device, dst_device), stats in movements.items():
        print(f"  {src_device} -> {dst_device}:")
        print(f"    Operations: {stats['count']} ({stats['noop_count']} no-ops)")
        print(f"    Bytes moved: {stats['bytes_moved']}")
        print(f"    No-op bytes: {stats['noop_bytes']}")
```

### Display Device Breakdown

Include device information in the formatted summary:

```python
# Show summary with device breakdown
print(OffloadStats.format_summary(unit="MB", show_devices=True))
```

This will display both the overall statistics and a detailed breakdown by device pair:

```
Offload Statistics
==================================================
Operation       Count   No-ops   Data Moved
--------------------------------------------------
Onload              10        0      39.06 MB
Offload             12        2      39.06 MB
Update               3        0      11.72 MB
--------------------------------------------------
Total               25        2      89.84 MB
==================================================

Device Movement Breakdown
=========================================================================================================
Operation          Source         Dest    Count   No-ops        Moved   No-op Data
---------------------------------------------------------------------------------------------------------
Onload                cpu       cuda:0       10        0      39.06 MB       0.00 MB
Offload            cuda:0          cpu       10        0      39.06 MB       0.00 MB
Offload            cuda:0       cuda:0        2        2       0.00 MB       7.81 MB
Update             cuda:0          cpu        3        0      11.72 MB       0.00 MB
=========================================================================================================
```

The "No-op Data" column shows the amount of data that would have been transferred if the no-op operations had actually moved data. This is useful for understanding the efficiency of caching and identifying unnecessary operations.

### Resetting Statistics

```python
# Reset all statistics to zero
OffloadStats.reset()
```

### Formatting Options

The `format_summary()` method supports different units:

```python
# Display in different units
print(OffloadStats.format_summary(unit="B"))   # Bytes
print(OffloadStats.format_summary(unit="KB"))  # Kilobytes
print(OffloadStats.format_summary(unit="MB"))  # Megabytes (default)
print(OffloadStats.format_summary(unit="GB"))  # Gigabytes
```

### Example Output

```
Offload Statistics
==================================================
Operation        Count   No-ops   Data Moved
--------------------------------------------------
Onload              10        1      39.06 MB
Offload             10        1      39.06 MB
Update               3        0      11.72 MB
--------------------------------------------------
Total               23        2      89.84 MB
==================================================
```

## Implementation Details

### OperationStats

Each operation type (onload, offload, update) has its own `OperationStats` object tracking:
- `count`: Total number of operations
- `bytes_moved`: Total bytes transferred (excluding no-ops)
- `noop_count`: Number of no-op operations
- `device_stats`: Dictionary mapping `(source_device, dest_device)` to `DevicePairStats`

The `record(input_tensor, result_tensor)` method takes both the input and result tensors, automatically determines if the operation was a no-op, and tracks the source and destination devices.

### DevicePairStats

For each device pair (source → destination), the following is tracked:
- `count`: Total operations for this device pair
- `noop_count`: No-op operations for this device pair
- `bytes_moved`: Actual bytes transferred (excluding no-ops)
- `noop_bytes`: Bytes that would have been transferred in no-op operations

This allows you to see, for example, that 10 operations from `cuda:0` to `cpu` occurred, with 2 being no-ops that would have moved 800 bytes if they weren't no-ops.

### No-op Detection

An operation is considered a no-op when:
- Either the input or result tensor is `None`
- The input and result tensors have the same data pointer and device (no actual data movement occurred)

### Decorators

The tracking is implemented using three class method decorators:
- `@OffloadStats.track_onload`: Applied to `onload()` methods
- `@OffloadStats.track_offload`: Applied to `offload()` methods
- `@OffloadStats.track_update`: Applied to `update_offload()` methods

## Notes

- **Disabled by default**: Statistics collection is disabled by default to avoid runtime overhead. Use `OffloadStats.enable()` to turn it on.
- **Global scope**: Statistics are shared across all `OffloadCache` instances
- **Thread safety**: The current implementation is not thread-safe
- **No instantiation**: `OffloadStats` should never be instantiated; attempting to do so will raise a `RuntimeError`
- **Automatic decoration**: All built-in `OffloadCache` subclasses are automatically decorated
- **Performance**: When disabled, the only overhead is a single boolean check per operation, which is negligible

## Example Script

See `examples/offload_stats_example.py` for a complete working example.

## Testing

Run the statistics tests with:

```bash
pytest tests/test_offload/cache/test_stats.py
```
