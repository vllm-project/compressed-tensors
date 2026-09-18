# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.cache.disk_utils import disk_load_context
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


def _counting_safe_open(monkeypatch):
    """Patch `safe_open` in the disk cache module and count how often it opens."""
    from compressed_tensors.offload.cache import disk_utils as disk_module

    real = disk_module.safe_open
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(disk_module, "safe_open", counted)
    return calls


def _cache_with_weights(offload_dir, names):
    """A DiskCache holding `names`, each in its own backing file."""
    DiskCache.index = {}
    cache = DiskCache("cpu", offload_dir=str(offload_dir))
    for i, name in enumerate(names):
        cache[name] = torch.full((4,), float(i))
    return cache


@pytest.mark.unit
def test_disk_load_context_reuses_one_handle_per_file(tmp_path, monkeypatch):
    """The point of the context: N reads from one file cost one open, not N.

    `safe_open` parses a header covering every tensor in the file, so opening
    per read is O(N) per read and O(N**2) across the group. This asserts the
    group collapses to a single open.
    """
    from compressed_tensors.offload.cache.disk_utils import _opened

    names = [f"w{i}" for i in range(6)]
    shard = tmp_path / "shard.safetensors"
    save_file({n: torch.full((4,), float(i)) for i, n in enumerate(names)}, shard)

    opens = _counting_safe_open(monkeypatch)
    with disk_load_context():
        for i, name in enumerate(names):
            with _opened(str(shard), "cpu") as file:
                assert_tensor_equal(file.get_tensor(name), torch.full((4,), float(i)))
    assert len(opens) == 1, f"expected a single open, got {len(opens)}"

    # same reads outside the context open once each
    opens.clear()
    for name in names:
        with _opened(str(shard), "cpu") as file:
            file.get_tensor(name)
    assert len(opens) == len(names)


@pytest.mark.unit
def test_without_context_opens_per_read(tmp_path, monkeypatch):
    """Outside the context the behaviour is unchanged, one open per read."""
    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)
    names = [f"w{i}" for i in range(4)]
    cache = _cache_with_weights(offload_dir, names)

    opens = _counting_safe_open(monkeypatch)
    outside = [cache[name] for name in names]
    assert len(opens) == len(names)

    opens.clear()
    with disk_load_context():
        inside = [cache[name] for name in names]

    for a, b in zip(outside, inside):
        assert_tensor_equal(a, b)


@pytest.mark.unit
def test_disk_load_context_closes_handles_on_exit(tmp_path):
    """Handles must actually be closed on exit, not merely dropped.

    Asserts against the handle itself rather than the cache dict, because
    clearing the dict while leaking the descriptors would otherwise pass.
    """
    from compressed_tensors.offload.cache import disk_utils as disk_module
    from compressed_tensors.offload.cache.disk_utils import _opened

    shard = tmp_path / "shard.safetensors"
    save_file({"a": torch.zeros(4)}, shard)

    with disk_load_context():
        with _opened(str(shard), "cpu") as file:
            file.get_tensor("a")
        held = list(disk_module._open_files.cache.values())
        assert held, "expected a cached handle"
    assert disk_module._open_files.cache is None
    for handle in held:
        with pytest.raises(Exception, match="closed"):
            handle.get_tensor("a")

    # and the same when the block raises
    with pytest.raises(RuntimeError):
        with disk_load_context():
            with _opened(str(shard), "cpu") as file:
                file.get_tensor("a")
            held = list(disk_module._open_files.cache.values())
            raise RuntimeError("boom")
    for handle in held:
        with pytest.raises(Exception, match="closed"):
            handle.get_tensor("a")


@pytest.mark.unit
def test_disk_load_context_nests(tmp_path, monkeypatch):
    """An inner context must not close handles the outer one is still using."""
    from compressed_tensors.offload.cache import disk_utils as disk_module

    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)
    cache = _cache_with_weights(offload_dir, ["w0"])

    opens = _counting_safe_open(monkeypatch)
    with disk_load_context():
        cache["w0"]
        with disk_load_context():
            cache["w0"]
        assert disk_module._open_files.cache is not None
        cache["w0"]
    assert disk_module._open_files.cache is None
    assert len(opens) == 1


@pytest.mark.unit
def test_disk_load_context_bounds_open_handles(tmp_path, monkeypatch):
    """More distinct files than the cap must not hold unbounded descriptors."""
    from compressed_tensors.offload.cache import disk_utils as disk_module

    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)
    count = disk_module._MAX_OPEN_FILES + 4
    names = [f"w{i}" for i in range(count)]
    cache = _cache_with_weights(offload_dir, names)

    with disk_load_context():
        for name in names:
            cache[name]
        assert len(disk_module._open_files.cache) <= disk_module._MAX_OPEN_FILES


def _symlinked_shard(tmp_path, count):
    """A real multi-tensor shard plus one checkpoint symlink per tensor.

    Mirrors `create_checkpoint_symlink`, which names each symlink after
    `id(offloaded)`, so N tensors from one shard arrive as N distinct paths.
    """
    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file({f"w{i}": torch.full((4,), float(i)) for i in range(count)}, shard)

    DiskCache.index = {}
    offloaded = []
    for i in range(count):
        meta = torch.empty(4, device="meta")
        DiskCache.create_checkpoint_symlink(
            meta,
            {
                "safetensors_file": str(shard),
                "weight_name": f"w{i}",
                "dtype": "float32",
            },
            str(offload_dir),
        )
        offloaded.append(meta)
    return DiskCache("cpu", offload_dir=str(offload_dir)), offloaded


@pytest.mark.unit
def test_disk_load_context_hits_across_checkpoint_symlinks(tmp_path, monkeypatch):
    """N tensors symlinked into one shard must share a single handle.

    This is the path the context exists for: the O(N**2) header parse only
    appears on multi-tensor shards, and those arrive through per-tensor
    symlinks. Keying the cache on the given path would miss every time.
    """
    cache, offloaded = _symlinked_shard(tmp_path, count=8)
    assert len({DiskCache.index[m]["safetensors_file"] for m in offloaded}) == 8

    opens = _counting_safe_open(monkeypatch)
    with disk_load_context():
        values = [cache.onload(meta) for meta in offloaded]

    assert len(opens) == 1, f"expected one open for one shard, got {len(opens)}"
    for i, value in enumerate(values):
        assert_tensor_equal(value, torch.full((4,), float(i)))


@pytest.mark.unit
@pytest.mark.parametrize("symlinked", [False, True])
def test_update_offload_is_not_served_stale_in_context(tmp_path, symlinked):
    """A rewrite through `update_offload` must be visible inside the context.

    Covers both ways `update_offload` rewrites: overwriting a CT-written file in
    place, and replacing a checkpoint symlink with a real file.
    """
    if symlinked:
        cache, offloaded = _symlinked_shard(tmp_path, count=2)
        meta = offloaded[0]
    else:
        offload_dir = tmp_path / "offload_dir"
        os.mkdir(offload_dir)
        DiskCache.index = {}
        cache = DiskCache("cpu", offload_dir=str(offload_dir))
        cache["w"] = torch.full((4,), 1.0)
        meta = cache.offloaded_values["w"]

    with disk_load_context():
        cache.onload(meta)
        cache.update_offload(meta, torch.full((4,), 99.0))
        assert_tensor_equal(cache.onload(meta), torch.full((4,), 99.0))


@pytest.mark.unit
def test_update_offload_keeps_shared_shard_handle(tmp_path, monkeypatch):
    """Updating one symlinked tensor must not close the shard others still read.

    `update_offload` replaces that tensor's symlink with its own file and leaves
    the shard untouched, so the other tensors keep hitting the cached handle.
    """
    cache, offloaded = _symlinked_shard(tmp_path, count=4)
    opens = _counting_safe_open(monkeypatch)
    with disk_load_context():
        for meta in offloaded:
            cache.onload(meta)
        cache.update_offload(offloaded[0], torch.full((4,), 99.0))
        for i, meta in enumerate(offloaded[1:], start=1):
            assert_tensor_equal(cache.onload(meta), torch.full((4,), float(i)))
        assert_tensor_equal(cache.onload(offloaded[0]), torch.full((4,), 99.0))

    # one open for the shard, one for the rewritten tensor's new file
    assert len(opens) == 2, f"expected 2 opens, got {len(opens)}"
