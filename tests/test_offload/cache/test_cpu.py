# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import compressed_tensors.offload.cache.cpu as cpu_cache
import pytest
import torch
from loguru import logger as loguru_logger
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
)
from tests.testing_utils import requires_gpu


@pytest.fixture()
def onload_device():
    return torch.device("cuda")


@pytest.fixture()
def offload_device():
    return torch.device("cpu")


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
def test_offload_logs_memory_hint(onload_device):
    cache = cpu_cache.CPUCache(onload_device)

    original_send_tensors = cpu_cache.send_tensors

    def raise_memory_error(*args, **kwargs):
        raise RuntimeError("mmap failed: Cannot allocate memory")

    cpu_cache.send_tensors = raise_memory_error

    warnings = []
    handler_id = loguru_logger.add(
        lambda msg: warnings.append(msg.record["message"]), level="WARNING"
    )

    try:
        with pytest.raises(RuntimeError, match="Cannot allocate memory"):
            cache.offload(torch.zeros(1, device=onload_device))
    finally:
        cpu_cache.send_tensors = original_send_tensors
        loguru_logger.remove(handler_id)

    assert any(
        "CPU offloading ran out of host RAM or mmap descriptors." in w for w in warnings
    )


@pytest.mark.unit
@requires_gpu
def test_offload_logs_memory_hint_oserror(onload_device):
    import errno

    cache = cpu_cache.CPUCache(onload_device)

    original_send_tensors = cpu_cache.send_tensors

    def raise_memory_error(*args, **kwargs):
        raise OSError(errno.ENOMEM, "Cannot allocate memory")

    cpu_cache.send_tensors = raise_memory_error

    warnings = []
    handler_id = loguru_logger.add(
        lambda msg: warnings.append(msg.record["message"]), level="WARNING"
    )

    try:
        with pytest.raises(OSError):
            cache.offload(torch.zeros(1, device=onload_device))
    finally:
        cpu_cache.send_tensors = original_send_tensors
        loguru_logger.remove(handler_id)

    assert any(
        "CPU offloading ran out of host RAM or mmap descriptors." in w for w in warnings
    )
