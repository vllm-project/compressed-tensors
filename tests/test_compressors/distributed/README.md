# Distributed Compression Tests

This directory contains tests for distributed parallel compression functionality.

## Test Files

### `test_module_parallel.py`
Unit tests for the `apply_module_parallel` function and related utilities:
- `to_meta()` function tests
- Basic module parallel processing
- Load balancing across ranks
- Offloaded module support
- State broadcasting
- Edge cases (empty lists, single modules, many modules)
- Parameter/buffer manipulation
- Exception handling

### `test_distributed_compression.py`
Integration tests for distributed model compression:
- End-to-end model compression
- Consistency checks across ranks
- Compression with offloaded modules
- Compress/decompress roundtrip
- Load balancing with many layers
- Mixed quantized/non-quantized models
- Edge cases (empty models, single layer)

## Running Tests

### Run all distributed tests
```bash
pytest tests/test_compressors/distributed/ -v
```

### Run specific test file
```bash
pytest tests/test_compressors/distributed/test_module_parallel.py -v
```

### Run specific test
```bash
pytest tests/test_compressors/distributed/test_module_parallel.py::test_apply_module_parallel_basic -v
```

## Requirements

- At least 2 GPUs for distributed tests (tests use `@requires_gpu(2)`)
- Tests use the `@torchrun(world_size=2)` decorator which spawns multi-process distributed tests
- NCCL backend is used for distributed communication

## Test Infrastructure

Tests use the `torchrun` decorator from `tests/test_offload/conftest.py`:
- Automatically spawns distributed processes using `torch.distributed.run`
- Initializes process groups with NCCL backend
- Each test runs in parallel across all ranks
- Test assertions are checked on all ranks independently

## Writing New Tests

When writing new distributed tests:

1. Import the `torchrun` decorator:
   ```python
   from tests.test_offload.conftest import torchrun
   ```

2. Apply decorators:
   ```python
   @pytest.mark.unit
   @requires_gpu(2)
   @torchrun(world_size=2)
   def test_my_distributed_feature():
       # Test code here
       pass
   ```

3. Use `dist.barrier()` when synchronization is needed
4. Use `dist.get_rank()` and `dist.get_world_size()` for rank-specific logic
5. Remember that all ranks execute the test independently

## Common Patterns

### Check consistency across ranks
```python
# Compute a checksum on each rank
checksum = module.weight.sum().item()

# Gather to rank 0
if dist.get_rank() == 0:
    gathered = [None] * dist.get_world_size()
    dist.gather_object(checksum, gathered, dst=0)
    # Verify all ranks have the same value
    assert all(c == gathered[0] for c in gathered)
else:
    dist.gather_object(checksum, None, dst=0)
```

### Test rank-specific behavior
```python
def apply_fn(module):
    # Each rank does something different
    module.weight.data.fill_(float(dist.get_rank()))

apply_module_parallel(modules, apply_fn, module_size)
dist.barrier()

# Verify results are broadcast correctly
```
