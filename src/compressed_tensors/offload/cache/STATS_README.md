# OffloadCache Statistics Tracking

The `OffloadStats` class provides automatic tracking of device movement operations for all `OffloadCache` subclasses.

## Features

- **Automatic tracking**: All onload, offload, and update operations are tracked automatically via decorators
- **Bytes moved**: Tracks the total amount of data transferred for each operation type
- **No-op detection**: Identifies operations that don't actually move data (e.g., None tensors, same-device operations)
- **Global statistics**: All cache instances share the same statistics, providing a unified view of device movement

## Usage

### Basic Usage

Statistics are collected automatically - no setup required:

```python
from compressed_tensors.offload.cache import CPUCache, OffloadStats
import torch

# Create a cache
device = torch.device("cuda")
cache = CPUCache(onload_device=device)

# Perform operations (automatically tracked)
tensor = torch.randn(100, 100, device=device)
cache["my_tensor"] = tensor  # Offload tracked
retrieved = cache["my_tensor"]  # Onload tracked

# View statistics
print(OffloadStats.format_summary())
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
OffloadCache Statistics
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
- `bytes_moved`: Total bytes transferred
- `noop_count`: Number of no-op operations

The `record(input_tensor, result_tensor)` method takes both the input and result tensors and automatically determines if the operation was a no-op.

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

- **Global scope**: Statistics are shared across all `OffloadCache` instances
- **Thread safety**: The current implementation is not thread-safe
- **No instantiation**: `OffloadStats` should never be instantiated; attempting to do so will raise a `RuntimeError`
- **Automatic decoration**: All built-in `OffloadCache` subclasses are automatically decorated

## Example Script

See `examples/offload_stats_example.py` for a complete working example.

## Testing

Run the statistics tests with:

```bash
pytest tests/test_offload/cache/test_stats.py
```
