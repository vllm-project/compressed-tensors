# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from pathlib import Path

import pytest
import torch
from compressed_tensors.offload import (
    align_module_device,
    align_modules,
    disable_offloading,
    disable_onloading,
    get_cache_init_kwargs,
    get_cache_kwargs,
    get_execution_device,
    get_offloaded_device,
    update_offload_parameter,
)
from compressed_tensors.offload.cache import CPUCache
from compressed_tensors.offload.module import offload_module
from tests.test_offload.conftest import assert_device_equal, assert_tensor_equal
from tests.testing_utils import requires_gpu


ONLOAD_DEVICE = torch.device("cuda")
OFFLOAD_DEVICE = torch.device("cpu")


@pytest.fixture(scope="function")
def cache():
    return CPUCache(ONLOAD_DEVICE)


@pytest.fixture(scope="function")
def linear():
    return torch.nn.Linear(5, 5, bias=True, device=OFFLOAD_DEVICE)


@pytest.fixture(scope="function")
def offloaded_linear(linear, cache):
    offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)
    return linear


@pytest.mark.unit
@requires_gpu
def test_disable_offloading():
    cache1 = CPUCache(ONLOAD_DEVICE)
    cache2 = CPUCache(ONLOAD_DEVICE)

    cache1["weight"] = torch.tensor(0, device=OFFLOAD_DEVICE)
    cache2["weight"] = torch.tensor(1, device=OFFLOAD_DEVICE)

    with disable_offloading():
        assert cache1["weight"] in cache1.keep_onloaded_values.values()
        assert cache2["weight"] in cache2.keep_onloaded_values.values()


@pytest.mark.unit
@requires_gpu
def test_disable_onloading():
    cache1 = CPUCache(ONLOAD_DEVICE)
    cache2 = CPUCache(ONLOAD_DEVICE)

    cache1["weight"] = torch.tensor(0, device=OFFLOAD_DEVICE)
    cache2["weight"] = torch.tensor(1, device=OFFLOAD_DEVICE)

    with disable_onloading():
        assert_device_equal(cache1["weight"].device, OFFLOAD_DEVICE)
        assert_device_equal(cache2["weight"].device, OFFLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
@pytest.mark.parametrize("offload", (True, False))
def test_update_offload_parameter(linear: torch.nn.Linear, cache, offload):
    init_data = torch.tensor(0.0, device=OFFLOAD_DEVICE)
    linear.weight = torch.nn.Parameter(init_data, requires_grad=False)
    if offload:
        offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)

    assert linear.weight == 0

    update_offload_parameter(linear, "weight", torch.tensor(1))
    assert linear.weight == 1

    with disable_offloading():
        update_offload_parameter(linear, "weight", torch.tensor(2))
        assert linear.weight == 2
    assert linear.weight == 2

    with disable_onloading():
        update_offload_parameter(linear, "weight", torch.tensor(3))
        assert linear.weight == 3
    assert linear.weight == 3


@pytest.mark.unit
def test_update_offload_parameter_with_grad(linear: torch.nn.Linear):
    zeros = torch.nn.Parameter(torch.zeros(5, 5), requires_grad=True)
    update_offload_parameter(linear, "weight", zeros)
    assert_tensor_equal(linear.weight, zeros)

    ones = torch.nn.Parameter(torch.ones(5, 5), requires_grad=True)
    offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)
    update_offload_parameter(linear, "weight", ones)
    assert_tensor_equal(linear.weight, ones, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_get_execution_device(linear: torch.nn.Linear, cache):
    assert_device_equal(get_execution_device(linear), OFFLOAD_DEVICE)
    linear.to(ONLOAD_DEVICE)
    assert_device_equal(get_execution_device(linear), ONLOAD_DEVICE)

    linear.to(OFFLOAD_DEVICE)
    offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)
    assert_device_equal(get_execution_device(linear), ONLOAD_DEVICE)

    with disable_onloading():
        assert_device_equal(get_execution_device(linear), ONLOAD_DEVICE)

    with disable_offloading():
        assert_device_equal(get_execution_device(linear), ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_get_offloaded_device(linear: torch.nn.Linear, cache):
    assert_device_equal(get_offloaded_device(linear), OFFLOAD_DEVICE)
    linear.to(ONLOAD_DEVICE)
    assert_device_equal(get_offloaded_device(linear), ONLOAD_DEVICE)

    linear.to(OFFLOAD_DEVICE)
    offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)
    assert_device_equal(get_offloaded_device(linear), OFFLOAD_DEVICE)

    with disable_onloading():
        assert_device_equal(get_offloaded_device(linear), OFFLOAD_DEVICE)

    with disable_offloading():
        assert_device_equal(get_offloaded_device(linear), OFFLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_get_cache_kwargs_cpu():
    """Test get_cache_kwargs for CPUCache."""
    # Non-offloaded module should return empty dict
    linear = torch.nn.Linear(5, 5, device=OFFLOAD_DEVICE)
    kwargs = get_cache_kwargs(linear)
    assert kwargs == {}

    # With default provided
    default = {"custom": "value"}
    kwargs = get_cache_kwargs(linear, default=default)
    assert kwargs == {"custom": "value"}

    # Offloaded module with CPUCache (no extra kwargs)
    offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)
    kwargs = get_cache_kwargs(linear)
    assert kwargs == {}


@pytest.mark.unit
@requires_gpu
def test_get_cache_kwargs_disk():
    """Test get_cache_kwargs for DiskCache extracts offload_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        linear = torch.nn.Linear(5, 5, device="cpu")
        offload_module(linear, ONLOAD_DEVICE, "disk", offload_dir=tmpdir)

        kwargs = get_cache_kwargs(linear)
        assert kwargs == {"offload_dir": Path(tmpdir).resolve()}


@pytest.mark.unit
@requires_gpu
def test_get_cache_init_kwargs_cpu():
    """
    Test using get_cache_init_kwargs to copy offloading from one module to another.
    """
    # Create and offload a reference module
    ref_linear = torch.nn.Linear(3, 3, device=OFFLOAD_DEVICE)
    offload_module(ref_linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)

    # Get the init kwargs from the reference module
    kwargs = get_cache_init_kwargs(ref_linear)

    # Create a new module and apply the same offload settings
    new_linear = torch.nn.Linear(5, 5, device=OFFLOAD_DEVICE)
    offload_module(new_linear, **kwargs)

    # Verify the new module has the same offload settings
    assert isinstance(new_linear._parameters, CPUCache)
    assert_device_equal(new_linear._parameters.onload_device, ONLOAD_DEVICE)
    assert_device_equal(new_linear._parameters.offload_device, OFFLOAD_DEVICE)
    # Verify weights work correctly
    assert_device_equal(new_linear.weight.device, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_get_cache_init_kwargs_disk():
    """Test using get_cache_init_kwargs to copy disk offload settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and offload a reference module with disk offloading
        ref_linear = torch.nn.Linear(3, 3, device="cpu")
        offload_module(ref_linear, ONLOAD_DEVICE, "disk", offload_dir=tmpdir)

        # Get the init kwargs from the reference module
        kwargs = get_cache_init_kwargs(ref_linear)

        # Create a new module and apply the same offload settings
        new_linear = torch.nn.Linear(5, 5, device="cpu")
        offload_module(new_linear, **kwargs)

        # Verify the new module has the same offload settings including offload_dir
        assert_device_equal(new_linear._parameters.onload_device, ONLOAD_DEVICE)
        assert new_linear._parameters.offload_device == "disk"
        assert new_linear._parameters.offload_dir == Path(tmpdir).resolve()
        # Verify weights work correctly
        assert_device_equal(new_linear.weight.device, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_register_offload_module_cpu(linear: torch.nn.Linear):
    from compressed_tensors.offload import register_offload_module

    # Non-offloaded parent - submodule should not be offloaded
    sub1 = torch.nn.Linear(1, 1)
    register_offload_module(linear, "sub1", sub1)
    assert linear.sub1 is sub1
    assert not isinstance(sub1._parameters, CPUCache)

    # Offloaded parent - submodule should inherit offloading
    offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)
    sub2 = torch.nn.Linear(1, 1)
    register_offload_module(linear, "sub2", sub2)
    assert linear.sub2 is sub2
    assert isinstance(sub2._parameters, CPUCache)
    assert_device_equal(sub2._parameters.onload_device, ONLOAD_DEVICE)
    assert_device_equal(sub2._parameters.offload_device, OFFLOAD_DEVICE)
    assert_device_equal(sub2.weight.device, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_register_offload_module_disk():
    """Test register_offload_module inherits offload_dir from parent with DiskCache."""
    from compressed_tensors.offload import register_offload_module

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a parent module with disk offloading
        parent = torch.nn.Linear(5, 5, device="cpu")
        offload_module(parent, ONLOAD_DEVICE, "disk", offload_dir=tmpdir)

        # Register a submodule - it should inherit the disk offloading settings
        sub = torch.nn.Linear(3, 3, device="cpu")
        register_offload_module(parent, "sub", sub)

        # Verify the submodule has the same offload settings including offload_dir
        assert parent.sub is sub
        assert_device_equal(sub._parameters.onload_device, ONLOAD_DEVICE)
        assert sub._parameters.offload_device == "disk"
        assert sub._parameters.offload_dir == Path(tmpdir).resolve()
        # Verify weights work correctly
        assert_device_equal(sub.weight.device, ONLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
def test_align_modules(offloaded_linear: torch.nn.Linear):
    linear = torch.nn.Linear(1, 1, device=ONLOAD_DEVICE)

    with align_modules((linear, offloaded_linear), OFFLOAD_DEVICE):
        assert_device_equal(linear.weight.device, OFFLOAD_DEVICE)
        assert_device_equal(offloaded_linear.weight.device, OFFLOAD_DEVICE)


@pytest.mark.unit
@requires_gpu
@pytest.mark.parametrize("offload", (True, False))
def test_align_module_device(linear: torch.nn.Linear, cache, offload):
    if offload:
        offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)
    else:
        linear.to(ONLOAD_DEVICE)

    with align_module_device(linear, OFFLOAD_DEVICE):
        assert_device_equal(linear.weight.device, OFFLOAD_DEVICE)
