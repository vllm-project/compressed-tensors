# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
import time

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.distributed import get_source_rank
from compressed_tensors.offload import disable_onloading
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.cache.disk_utils import disk_load_context
from compressed_tensors.offload.cache.dist_disk import DistributedDiskCache
from compressed_tensors.offload.cache.writeback import AsyncDiskWriteback
from safetensors import safe_open
from safetensors.torch import save_file
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
from tests.test_offload.conftest import assert_tensor_equal, torchrun
from tests.testing_utils import requires_gpu


@pytest.fixture()
def onload_device():
    return torch.accelerator.current_accelerator()


@pytest.fixture()
def offload_device():
    return "disk"


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_delete(offload_device, onload_device, offload_cache):
    _test_delete(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_disable_offloading(offload_device, onload_device, offload_cache):
    _test_disable_offloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_disable_onloading(offload_device, onload_device, offload_cache):
    _test_disable_onloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_garbage_collect(offload_device, onload_device, offload_cache):
    _test_garbage_collect(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_offload(offload_device, onload_device, offload_cache):
    _test_offload(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_onload(offload_device, onload_device, offload_cache):
    _test_onload(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_onloading(offload_device, onload_device, offload_cache):
    _test_onloading(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_shared_attributes(offload_device, onload_device, offload_cache):
    _test_shared_attributes(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_tensor_subclass(offload_device, onload_device, offload_cache):
    _test_tensor_subclass(offload_device, onload_device, offload_cache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_distributed_offload(onload_device, tmp_path):
    # Broadcast directory path from rank 0 to all ranks
    if dist.get_rank() == 0:
        offload_dir = tmp_path / "offload_dir"
        os.mkdir(offload_dir)
        broadcast_obj = [str(offload_dir)]
    else:
        broadcast_obj = [None]

    dist.broadcast_object_list(broadcast_obj, src=0)
    offload_dir = broadcast_obj[0]

    # Ensure directory creation completes before other ranks proceed
    dist.barrier()

    cache = DistributedDiskCache(onload_device, offload_dir=offload_dir)
    tensor = torch.zeros((5, 2))
    cache["tensor"] = tensor

    # check tensor construction
    assert torch.equal(cache["tensor"], tensor.to(onload_device))
    with disable_onloading():
        assert_tensor_equal(cache["tensor"], tensor.to("meta"))

    # update tensor
    tensor = torch.ones((5, 2))
    cache["tensor"] = tensor

    # check tensor construction
    assert torch.equal(cache["tensor"], tensor.to(onload_device))
    with disable_onloading():
        assert_tensor_equal(cache["tensor"], tensor.to("meta"))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_distributed_files(tmp_path):
    # Broadcast directory path from rank 0 to all ranks
    if dist.get_rank() == 0:
        offload_dir = tmp_path / "offload_dir"
        os.mkdir(offload_dir)
        broadcast_obj = [str(offload_dir)]
    else:
        broadcast_obj = [None]

    dist.broadcast_object_list(broadcast_obj, src=0)
    offload_dir = broadcast_obj[0]

    # Ensure directory creation completes before other ranks proceed
    dist.barrier()

    # initial write, broadcasted to all ranks
    DiskCache.index = {}
    cache = DistributedDiskCache("cpu", offload_dir=offload_dir)
    tensor = torch.zeros(10)
    cache["weight"] = tensor

    assert len(DiskCache.index) == 1
    if dist.get_rank() == 0:  # only rank0 bc `tmp_path` is not shared between ranks
        files = os.listdir(offload_dir)
        assert len(files) == 1
        with safe_open(
            os.path.join(offload_dir, files[0]), framework="pt", device="cpu"
        ) as file:
            read_tensor = file.get_tensor("weight")
            assert_tensor_equal(read_tensor, tensor)

    # modify on one rank
    tensor = torch.ones(10)
    if dist.get_rank() == 0:
        cache["weight"] = tensor

    assert len(DiskCache.index) == 1
    if dist.get_rank() == 0:  # only rank0 bc `tmp_path` is not shared between ranks
        files = os.listdir(offload_dir)
        assert len(files) == 1
        with safe_open(
            os.path.join(offload_dir, files[0]), framework="pt", device="cpu"
        ) as file:
            read_tensor = file.get_tensor("weight")
            assert_tensor_equal(read_tensor, tensor)

    # delete
    del cache["weight"]
    assert len(DiskCache.index) == 0
    if dist.get_rank() == 0:  # only rank0 bc `tmp_path` is not shared between ranks
        files = os.listdir(offload_dir)
        assert len(files) == 0


@pytest.mark.unit
@torchrun(world_size=2)
def test_distributed_async_disk_writeback(tmp_path, monkeypatch):
    """DDP offload returns before disk serialization and flush makes reads safe."""
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    writeback_group = dist.new_group(backend="gloo")

    if dist.get_rank() == 0:
        offload_dir = tmp_path / "offload_dir"
        offload_dir.mkdir()
        broadcast_obj = [str(offload_dir)]
    else:
        broadcast_obj = [None]
    dist.broadcast_object_list(broadcast_obj, src=0)
    offload_dir = broadcast_obj[0]
    dist.barrier()

    writer = AsyncDiskWriteback(defer_errors=True)
    cache = DistributedDiskCache("cpu", offload_dir=offload_dir, async_writer=writer)
    started = threading.Event()
    release = threading.Event()
    rank = dist.get_rank()
    source_rank = get_source_rank()

    if rank == source_rank:
        original_write = AsyncDiskWriteback._write_atomically

        def delayed_write(path, tensors):
            started.set()
            # The timeout keeps a regression from hanging the torchrun test if
            # the cache accidentally waits for disk I/O inside offload again.
            release.wait(timeout=3)
            original_write(path, tensors)

        monkeypatch.setattr(
            AsyncDiskWriteback, "_write_atomically", staticmethod(delayed_write)
        )

    try:
        start_time = time.monotonic()
        with writer.batch():
            cache["weight"] = torch.ones(8)
        offload_elapsed = time.monotonic() - start_time

        if rank == source_rank:
            write_started = started.wait(timeout=3)
            weight = cache.offloaded_values["weight"]
            path = cache.index[weight]["safetensors_file"]
            unpublished_while_blocked = write_started and not os.path.exists(path)
            release.set()
        else:
            write_started = True
            unpublished_while_blocked = True

        writer.flush_distributed(writeback_group)
        assert torch.equal(cache["weight"], torch.ones(8))

        if rank == source_rank:
            assert write_started
            assert unpublished_while_blocked
            assert offload_elapsed < 2
    finally:
        release.set()
        try:
            writer.close()
        except RuntimeError:
            pass
        dist.destroy_process_group(writeback_group)


@pytest.mark.unit
@torchrun(world_size=2)
def test_distributed_async_disk_writeback_error_propagates(tmp_path, monkeypatch):
    """A source-rank worker failure is reported on every rank at flush."""
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    writeback_group = dist.new_group(backend="gloo")

    if dist.get_rank() == 0:
        offload_dir = tmp_path / "offload_dir"
        offload_dir.mkdir()
        broadcast_obj = [str(offload_dir)]
    else:
        broadcast_obj = [None]
    dist.broadcast_object_list(broadcast_obj, src=0)
    offload_dir = broadcast_obj[0]
    dist.barrier()

    writer = AsyncDiskWriteback(defer_errors=True)
    cache = DistributedDiskCache("cpu", offload_dir=offload_dir, async_writer=writer)
    source_rank = get_source_rank()
    if dist.get_rank() == source_rank:

        def fail_write(path, tensors):
            raise OSError("injected disk write failure")

        monkeypatch.setattr(
            AsyncDiskWriteback, "_write_atomically", staticmethod(fail_write)
        )

    try:
        with writer.batch():
            cache["weight"] = torch.ones(8)

        with pytest.raises(RuntimeError, match=f"rank {source_rank}:"):
            writer.flush_distributed(writeback_group)
    finally:
        try:
            writer.close()
        except RuntimeError:
            pass
        dist.destroy_process_group(writeback_group)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_distributed_async_update(tmp_path):
    """
    Test that different ranks can update different tensors asynchronously,
    and that values are correct after a barrier.
    """
    offload_dir = tmp_path / "offload_dir"
    if dist.get_rank() == 0:
        os.mkdir(offload_dir)

    # Ensure directory creation completes before other ranks proceed
    dist.barrier()

    onload_device = torch.accelerator.current_accelerator()
    cache = DistributedDiskCache(onload_device, offload_dir=str(offload_dir))

    # Initialize two tensors in the cache
    cache["tensor_0"] = torch.zeros(10, device=onload_device)
    cache["tensor_1"] = torch.zeros(10, device=onload_device)

    # Each rank updates a different tensor
    rank = dist.get_rank()
    if rank == 0:
        # Rank 0 updates tensor_0
        cache[f"tensor_{rank}"] = torch.ones(10, device=onload_device) * 1.0
    elif rank == 1:
        # Rank 1 updates tensor_1
        cache[f"tensor_{rank}"] = torch.ones(10, device=onload_device) * 2.0

    # Synchronize to ensure all updates are complete
    dist.barrier()

    # Verify that both tensors have the correct values on all ranks
    tensor_0 = cache["tensor_0"]
    tensor_1 = cache["tensor_1"]

    assert torch.allclose(tensor_0.cpu(), torch.ones(10) * 1.0)
    assert torch.allclose(tensor_1.cpu(), torch.ones(10) * 2.0)

    # Verify offloaded values are also correct
    with disable_onloading():
        offloaded_0 = cache["tensor_0"]
        offloaded_1 = cache["tensor_1"]
        assert_tensor_equal(offloaded_0, torch.ones(10) * 1.0, "disk")
        assert_tensor_equal(offloaded_1, torch.ones(10) * 2.0, "disk")


@pytest.mark.unit
@torchrun(world_size=2, init_dist=True)
def test_disk_load_context_is_per_rank(tmp_path_factory):
    """Each rank keeps its own handle cache and does not disturb the other's.

    The cache is thread local, so it is also process local. Two ranks reading
    the same shard must each open it once, and one rank leaving the context
    must not close a handle the other rank is still holding.
    """
    from compressed_tensors.offload.cache import disk_utils
    from compressed_tensors.offload.cache.disk_utils import _opened

    rank = dist.get_rank()
    # every rank writes its own shard so the test does not depend on a shared FS
    directory = tmp_path_factory.mktemp(f"shard{rank}")
    shard = directory / "shard.safetensors"
    save_file({f"w{i}": torch.full((4,), float(i)) for i in range(4)}, shard)

    with disk_load_context():
        for i in range(4):
            with _opened(str(shard), "cpu") as file:
                assert_tensor_equal(
                    file.get_tensor(f"w{i}"), torch.full((4,), float(i))
                )
        held = list(disk_utils._open_files.cache.values())
        assert len(held) == 1, f"rank {rank} expected one cached handle"
        dist.barrier()
        # the other rank has also entered and is holding its own handle
        assert disk_utils._open_files.cache is not None

    dist.barrier()
    assert disk_utils._open_files.cache is None, f"rank {rank} leaked its cache"
    for handle in held:
        with pytest.raises(Exception, match="closed"):
            handle.get_tensor("w0")


@pytest.mark.unit
@torchrun(world_size=2, init_dist=True)
def test_disk_load_context_with_distributed_disk_cache(tmp_path_factory):
    """`DistributedDiskCache` onload still returns correct data inside the context."""
    directory = tmp_path_factory.mktemp(f"dist{dist.get_rank()}")
    offload_dir = directory / "offload"
    os.mkdir(offload_dir)

    DistributedDiskCache.index = {}
    cache = DistributedDiskCache("cpu", offload_dir=str(offload_dir))
    expected = {
        name: torch.full((8,), float(i)) for i, name in enumerate(["a", "b", "c"])
    }
    for name, tensor in expected.items():
        cache[name] = tensor

    with disk_load_context():
        for name, tensor in expected.items():
            assert_tensor_equal(cache[name], tensor)

    # and identically outside the context
    for name, tensor in expected.items():
        assert_tensor_equal(cache[name], tensor)
