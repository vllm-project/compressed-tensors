# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import compressed_tensors.offload.cache.disk as disk_cache
import pytest
import torch
from compressed_tensors.offload.cache.disk import DiskCache
from loguru import logger as loguru_logger
from safetensors import safe_open
from tests.test_offload.cache.helpers import (
    _test_delete,
    _test_disable_offloading,
    _test_disable_onloading,
    _test_garbage_collect,
    _test_offload,
    _test_onload,
    _test_onloading,
    _test_shared_attributes,
    _test_tensor_subclass,
    _test_update_offload,
)
from tests.test_offload.conftest import assert_tensor_equal
from tests.testing_utils import requires_gpu


@pytest.fixture()
def onload_device():
    return torch.accelerator.current_accelerator()


@pytest.fixture()
def offload_device():
    return "disk"


@pytest.mark.unit
@requires_gpu
def test_delete(offload_device, onload_device, offload_cache):
    _test_delete(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_disable_offloading(offload_device, onload_device, offload_cache):
    _test_disable_offloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_disable_onloading(offload_device, onload_device, offload_cache):
    _test_disable_onloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_garbage_collect(offload_device, onload_device, offload_cache):
    _test_garbage_collect(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_offload(offload_device, onload_device, offload_cache):
    _test_offload(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
@requires_gpu
def test_onload(offload_device, onload_device, offload_cache):
    _test_onload(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_onloading(offload_device, onload_device, offload_cache):
    _test_onloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_shared_attributes(offload_device, onload_device, offload_cache):
    _test_shared_attributes(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_tensor_subclass(offload_device, onload_device, offload_cache):
    _test_tensor_subclass(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu
def test_update_offload(offload_device, onload_device, offload_cache):
    _test_update_offload(offload_device, onload_device, offload_cache)


@pytest.mark.unit
def test_files(tmp_path):
    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    # initial write
    DiskCache.index = {}
    cache = DiskCache("cpu", offload_dir=str(offload_dir))
    tensor = torch.zeros(10)
    cache["weight"] = tensor

    files = os.listdir(offload_dir)
    assert len(DiskCache.index) == 1
    assert len(files) == 1
    with safe_open(offload_dir / files[0], framework="pt", device="cpu") as file:
        read_tensor = file.get_tensor("weight")
        assert_tensor_equal(read_tensor, tensor)

    # modify
    tensor = torch.ones(10)
    cache["weight"] = tensor

    files = os.listdir(offload_dir)
    assert len(DiskCache.index) == 1
    assert len(files) == 1
    with safe_open(offload_dir / files[0], framework="pt", device="cpu") as file:
        read_tensor = file.get_tensor("weight")
        assert_tensor_equal(read_tensor, tensor)

    # delete
    del cache["weight"]
    files = os.listdir(offload_dir)
    assert len(DiskCache.index) == 0
    assert len(files) == 0


@pytest.mark.unit
def test_stage(tmp_path):
    offload_dir = tmp_path / "offload_dir"
    offload_dir.mkdir()
    cache = DiskCache("cpu", offload_dir=str(offload_dir))
    tensor = torch.arange(10)
    offloaded = cache.offload(tensor)

    staged = cache.stage(offloaded)

    assert staged.device.type == "cpu"
    assert torch.equal(staged, tensor)


@pytest.mark.unit
@requires_gpu
def test_stage_pinned_memory(tmp_path):
    offload_dir = tmp_path / "offload_dir"
    offload_dir.mkdir()
    onload_device = torch.accelerator.current_accelerator()
    cache = DiskCache(onload_device, offload_dir=str(offload_dir))
    tensor = torch.arange(10, device=onload_device)
    offloaded = cache.offload(tensor)

    staged = cache.stage(offloaded, pin_memory=True)

    assert staged.device.type == "cpu"
    assert staged.is_pinned()
    assert torch.equal(staged, tensor.cpu())


@pytest.mark.unit
@requires_gpu
def test_stage_pinned_memory_logs_hint(tmp_path):
    offload_dir = tmp_path / "offload_dir"
    offload_dir.mkdir()
    onload_device = torch.accelerator.current_accelerator()
    cache = DiskCache(onload_device, offload_dir=str(offload_dir))
    tensor = torch.arange(10, device=onload_device)
    offloaded = cache.offload(tensor)

    original_pin_memory = disk_cache._pin_memory

    def raise_memory_error(tensor):
        raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")

    disk_cache._pin_memory = raise_memory_error

    warnings = []
    handler_id = loguru_logger.add(
        lambda msg: warnings.append(msg.record["message"]), level="WARNING"
    )

    try:
        with pytest.raises(RuntimeError, match="Tried to allocate"):
            cache.stage(offloaded, pin_memory=True)
    finally:
        disk_cache._pin_memory = original_pin_memory
        loguru_logger.remove(handler_id)

    assert any("Pinned-memory staging ran out of host RAM" in w for w in warnings)
