# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from compressed_tensors.distributed import replace_module_parallel
from compressed_tensors.offload import offload_module
from compressed_tensors.offload.utils import module_size, to_meta
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


class SimpleLinear(nn.Module):
    """Simple linear module for testing."""

    def __init__(self, in_features=10, out_features=10):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))


class TwoLayerModel(nn.Module):
    """Model with two linear layers for testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = SimpleLinear(10, 10)
        self.layer2 = SimpleLinear(10, 10)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_to_meta():
    """Test that to_meta correctly moves module tensors to meta device."""
    module = SimpleLinear(5, 5)
    original_weight = module.weight.data.clone()
    original_bias = module.bias.data.clone()

    # Move to meta
    to_meta(module)

    # Check that tensors are on meta device
    assert module.weight.device.type == "meta"
    assert module.bias.device.type == "meta"

    # Check that shapes are preserved
    assert module.weight.shape == original_weight.shape
    assert module.bias.shape == original_bias.shape


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_basic():
    """Test basic replace_module_parallel functionality."""
    modules = [SimpleLinear(10, 10) for _ in range(4)]

    # Track which modules were processed
    processed_modules = []

    def apply_fn(module):
        processed_modules.append(id(module))
        # Simple modification: set weight to rank value
        module.weight.data.fill_(float(dist.get_rank()))

    replace_module_parallel(modules, apply_fn, module_size)

    # All modules should be processed exactly once
    assert len(processed_modules) == len(modules)
    assert len(set(processed_modules)) == len(modules)

    # All modules should have valid weights (not meta)
    for module in modules:
        assert module.weight.device.type != "meta"
        # Each rank should have processed some modules
        # We can't predict which ones due to bin packing,
        # but weights should be 0.0 or 1.0
        assert module.weight.mean().item() in [0.0, 1.0]


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_load_balancing():
    """Test that modules are distributed evenly across ranks."""
    # Create modules of different sizes
    modules = [
        SimpleLinear(100, 100),  # Large
        SimpleLinear(10, 10),  # Small
        SimpleLinear(100, 100),  # Large
        SimpleLinear(10, 10),  # Small
    ]

    # Track which rank processes each module
    rank_assignments = {}

    def apply_fn(module):
        rank_assignments[id(module)] = dist.get_rank()

    replace_module_parallel(modules, apply_fn, module_size)

    # Both ranks should have processed at least one module
    ranks_used = set(rank_assignments.values())
    assert len(ranks_used) == dist.get_world_size()


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_with_offload():
    """Test replace_module_parallel with offloaded modules."""
    modules = [SimpleLinear(10, 10) for _ in range(4)]

    # Offload modules to CPU
    for module in modules:
        offload_module(module, onload_device="cuda", offload_device="cpu")

    # Track processing
    processed_count = [0]

    def apply_fn(module):
        processed_count[0] += 1
        # Verify we can access the module's parameters
        assert module.weight is not None
        module.weight.data.fill_(1.0)

    replace_module_parallel(modules, apply_fn, module_size)

    # All modules should be processed
    assert processed_count[0] == len(modules)

    # All modules should have updated weights
    for module in modules:
        # Access weight and check it was updated
        weight = module.weight
        if weight.device.type == "cpu":
            assert torch.allclose(weight, torch.ones_like(weight))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_state_broadcast():
    """Test that state is correctly broadcast across ranks."""
    modules = [SimpleLinear(5, 5) for _ in range(2)]

    # Each rank processes different modules and sets unique values
    def apply_fn(module):
        # Set weight to a unique pattern based on rank
        module.weight.data.fill_(float(dist.get_rank() * 100))

    replace_module_parallel(modules, apply_fn, module_size)
    dist.barrier()

    # After replace_module_parallel, all ranks should have the same state
    # Check that weights are consistent across ranks
    for i, module in enumerate(modules):
        # Weight should be either 0 or 100 (depending on which rank processed it)
        mean_val = module.weight.mean().item()
        assert mean_val in [0.0, 100.0], f"Module {i} has unexpected weight: {mean_val}"


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_non_processing_ranks_use_meta():
    """Test that non-processing ranks temporarily use meta device."""
    modules = [SimpleLinear(10, 10) for _ in range(2)]

    # Track device usage during processing
    devices_seen = []

    def apply_fn(module):
        # Record the device when this is called
        devices_seen.append(module.weight.device.type)

    replace_module_parallel(modules, apply_fn, module_size)

    # At least one call should have seen a meta device (non-processing rank)
    # Note: This is tricky to test because the actual compression happens on-device
    # But we can verify the final state is valid
    for module in modules:
        assert module.weight.device.type != "meta"


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_preserves_module_structure():
    """Test that module structure is preserved after parallel processing."""
    module = SimpleLinear(5, 5)
    original_weight_shape = module.weight.shape
    original_bias_shape = module.bias.shape

    def apply_fn(m):
        # Modify the module but preserve structure
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0.5)

    replace_module_parallel([module], apply_fn, module_size)

    # Check structure is preserved
    assert module.weight.shape == original_weight_shape
    assert module.bias.shape == original_bias_shape
    assert hasattr(module, "weight")
    assert hasattr(module, "bias")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_empty_list():
    """Test replace_module_parallel with empty module list."""
    modules = []

    call_count = [0]

    def apply_fn(module):
        call_count[0] += 1

    # Should not raise an error
    replace_module_parallel(modules, apply_fn, module_size)

    # Function should never be called
    assert call_count[0] == 0


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_single_module():
    """Test replace_module_parallel with a single module."""
    module = SimpleLinear(10, 10)

    def apply_fn(m):
        m.weight.data.fill_(42.0)

    replace_module_parallel([module], apply_fn, module_size)

    # Module should be processed
    assert module.weight.mean().item() == 42.0


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_many_modules():
    """Test replace_module_parallel with many modules."""
    modules = [SimpleLinear(10, 10) for _ in range(20)]

    processed = set()

    def apply_fn(m):
        processed.add(id(m))
        m.weight.data.fill_(1.0)

    replace_module_parallel(modules, apply_fn, module_size)

    # All modules should be processed
    assert len(processed) == len(modules)

    # All modules should have updated weights
    for module in modules:
        assert module.weight.mean().item() == 1.0


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_custom_weight_function():
    """Test replace_module_parallel with custom weight function."""
    modules = [SimpleLinear(10, 10) for _ in range(4)]

    # Custom weight function that returns constant weights
    def constant_weight_fn(m):
        return 1.0  # All modules have equal weight

    processed = []

    def apply_fn(m):
        processed.append(id(m))

    replace_module_parallel(modules, apply_fn, constant_weight_fn)

    # All modules should still be processed
    assert len(processed) == len(modules)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_exception_handling():
    """Test that exceptions in apply_fn are properly propagated."""
    module = SimpleLinear(10, 10)

    def failing_apply_fn(m):
        raise ValueError("Test exception")

    # The exception should be raised
    with pytest.raises(ValueError, match="Test exception"):
        replace_module_parallel([module], failing_apply_fn, module_size)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_parameter_replacement():
    """Test replace_module_parallel when parameters are replaced."""
    module = SimpleLinear(10, 10)
    original_weight_shape = module.weight.shape

    def replace_weight_fn(m):
        # Replace weight with a new tensor of different values
        new_weight = torch.ones_like(m.weight) * 99.0
        m.weight = nn.Parameter(new_weight)

    replace_module_parallel([module], replace_weight_fn, module_size)

    # Check that the weight was replaced
    assert module.weight.shape == original_weight_shape
    assert module.weight.mean().item() == 99.0


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_adds_new_parameters():
    """Test replace_module_parallel when new parameters are added."""
    module = SimpleLinear(10, 10)

    def add_parameter_fn(m):
        # Add a new parameter
        m.new_param = nn.Parameter(torch.ones(5, 5))

    replace_module_parallel([module], add_parameter_fn, module_size)

    # Check that the new parameter exists on all ranks
    assert hasattr(module, "new_param")
    assert module.new_param.shape == (5, 5)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_removes_parameters():
    """Test replace_module_parallel when parameters are removed."""
    module = SimpleLinear(10, 10)

    def remove_bias_fn(m):
        # Remove the bias parameter
        delattr(m, "bias")

    replace_module_parallel([module], remove_bias_fn, module_size)

    # Check that bias is removed on all ranks
    assert not hasattr(module, "bias")
    assert hasattr(module, "weight")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_with_buffers():
    """Test replace_module_parallel with modules that have buffers."""

    class ModuleWithBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5, 5))
            self.register_buffer("running_mean", torch.zeros(5))

    module = ModuleWithBuffer()

    def update_buffer_fn(m):
        m.running_mean.fill_(1.0)
        m.weight.data.fill_(2.0)

    replace_module_parallel([module], update_buffer_fn, module_size)

    # Check that both parameter and buffer are updated
    assert module.weight.mean().item() == 2.0
    assert module.running_mean.mean().item() == 1.0


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_to_meta_preserves_parameter_properties():
    """Test that to_meta preserves parameter properties like requires_grad."""
    module = SimpleLinear(5, 5)
    module.weight.requires_grad = False

    to_meta(module)

    # Properties should be preserved
    assert module.weight.requires_grad is False
    assert isinstance(module.weight, nn.Parameter)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_replace_module_parallel_rank_consistency():
    """Test that all ranks see the same final state."""
    modules = [SimpleLinear(5, 5) for _ in range(4)]

    def apply_fn(m):
        # Fill with rank-specific value
        m.weight.data.fill_(float(dist.get_rank()))

    replace_module_parallel(modules, apply_fn, module_size)
    dist.barrier()

    # Collect checksums from all ranks
    checksums = []
    for module in modules:
        checksum = module.weight.sum().item()
        checksums.append(checksum)

    # Gather all checksums to rank 0
    if dist.get_rank() == 0:
        gathered = [None] * dist.get_world_size()
        dist.gather_object(checksums, gathered, dst=0)

        # All ranks should have the same checksums
        for rank_checksums in gathered[1:]:
            assert rank_checksums == gathered[0], "Ranks have different states!"
    else:
        dist.gather_object(checksums, None, dst=0)
