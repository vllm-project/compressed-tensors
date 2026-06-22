# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload.cache.base import OffloadCache, force_local_cache
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


def _cache_name(device):
    return OffloadCache.cls_from_device(device).__name__


def _is_distributed_cache(device):
    return "Distributed" in _cache_name(device)


def _is_local_cache(device):
    return "Distributed" not in _cache_name(device)


@pytest.mark.unit
@torchrun(world_size=2, init_dist=True)
def test_force_local_cache_cpu():
    assert _is_distributed_cache("cpu")
    with force_local_cache():
        assert _is_local_cache("cpu")
    assert _is_distributed_cache("cpu")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_force_local_cache_device():
    device = torch.device("cuda", torch.cuda.current_device())
    assert _is_distributed_cache(device)
    with force_local_cache():
        assert _is_local_cache(device)
    assert _is_distributed_cache(device)


@pytest.mark.unit
@torchrun(world_size=2, init_dist=True)
def test_force_local_cache_not_distributed():
    assert dist.is_initialized()
    assert _is_distributed_cache("cpu")
    with force_local_cache():
        assert _is_local_cache("cpu")


@pytest.mark.unit
@torchrun(world_size=2, init_dist=True)
def test_force_local_cache_nesting():
    assert _is_distributed_cache("cpu")
    with force_local_cache():
        assert _is_local_cache("cpu")
        with force_local_cache():
            assert _is_local_cache("cpu")
        assert _is_local_cache("cpu")
    assert _is_distributed_cache("cpu")


@pytest.mark.unit
@torchrun(world_size=2, init_dist=True)
def test_force_local_cache_restores_after_exception():
    try:
        with force_local_cache():
            assert _is_local_cache("cpu")
            raise RuntimeError("test")
    except RuntimeError:
        pass
    assert _is_distributed_cache("cpu")
