# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import json as json_module
import os
import struct as struct_module

import compressed_tensors.offload.cache.disk as ct_disk
import pytest
import torch
from compressed_tensors.offload import offload_module
from compressed_tensors.offload.cache import utils as cache_utils
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.module import (
    remove_module_offload,
    stage_module_offload,
)
from safetensors.torch import save_file
from tests.testing_utils import requires_gpu


@pytest.fixture
def pinned_staging_enabled():
    """Ensure direct pinned staging is on for the test and restored after."""
    old = os.environ.get("COMPRESSED_TENSORS_PINNED_STAGING")
    os.environ.pop("COMPRESSED_TENSORS_PINNED_STAGING", None)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("COMPRESSED_TENSORS_PINNED_STAGING", None)
        else:
            os.environ["COMPRESSED_TENSORS_PINNED_STAGING"] = old


def _checkpoint_backed_cache(tmp_path, tensors, onload_device="cuda"):
    """
    Build a DiskCache whose offloaded values are backed by a single shared
    safetensors shard, mimicking weights offloaded via `device_map="disk"`.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    cache = DiskCache(onload_device, offload_dir=str(tmp_path))
    shard = tmp_path / "shard.safetensors"
    save_file(tensors, str(shard))
    offloaded = {}
    for name, tensor in tensors.items():
        meta = tensor.to(device="meta")
        DiskCache.index[meta] = {
            "safetensors_file": str(shard),
            "weight_name": name,
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        offloaded[name] = meta
    return cache, offloaded


@requires_gpu
@pytest.mark.unit
@pytest.mark.parametrize("shape", [(), (0, 3), (11,), (4, 7)])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int64, torch.int8]
)
def test_pinned_stage_preserves_bytes(tmp_path, pinned_staging_enabled, shape, dtype):
    if shape == (4, 7):
        tensor = torch.arange(28, dtype=torch.float32).reshape(shape).to(dtype)
    else:
        tensor = torch.empty(shape, dtype=dtype)
    cache, offloaded = _checkpoint_backed_cache(tmp_path, {"weight": tensor})

    staged = cache.stage(offloaded["weight"], pin_memory=True)

    assert staged.device.type == "cpu"
    # zero-element tensors have a null data pointer and never report pinned
    if tensor.numel():
        assert staged.is_pinned()
    assert staged.dtype == tensor.dtype
    assert staged.shape == tensor.shape
    assert torch.equal(staged, tensor)


@requires_gpu
@pytest.mark.unit
def test_pinned_stage_casts_through_safetensors(tmp_path, pinned_staging_enabled):
    # checkpoint holds float32 while the offloaded tensor is bfloat16: the raw
    # reader cannot convert, so staging must fall back to safetensors + pin
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    cache = DiskCache("cuda", offload_dir=str(tmp_path))
    shard = tmp_path / "shard.safetensors"
    save_file({"weight": tensor}, str(shard))
    meta = torch.empty(3, 4, dtype=torch.bfloat16, device="meta")
    DiskCache.index[meta] = {
        "safetensors_file": str(shard),
        "weight_name": "weight",
        "dtype": "bfloat16",
    }

    staged = cache.stage(meta, pin_memory=True)

    assert staged.is_pinned()
    assert staged.dtype == torch.bfloat16
    assert torch.equal(staged, tensor.to(torch.bfloat16))


@requires_gpu
@pytest.mark.unit
def test_pinned_stage_preserves_parameter_and_metadata(
    tmp_path, pinned_staging_enabled
):
    parameter = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    parameter.test_metadata = "preserve"
    cache = DiskCache("cuda", offload_dir=str(tmp_path))
    offloaded = cache.offload(parameter)
    offloaded.requires_grad_(True)

    staged = cache.stage(offloaded, pin_memory=True)

    assert isinstance(staged, torch.nn.Parameter)
    assert staged.test_metadata == parameter.test_metadata
    assert staged.requires_grad
    assert staged.is_pinned()
    assert torch.equal(staged, parameter.data)


@requires_gpu
@pytest.mark.unit
def test_staged_onload_is_correct_with_async_copies(tmp_path, pinned_staging_enabled):
    # non_blocking H2D from pinned memory must be ordered before anything
    # that observes the tensor, without an explicit synchronize
    tensors = {
        "a": torch.arange(64, dtype=torch.float32).reshape(8, 8),
        "b": torch.arange(8, dtype=torch.bfloat16),
    }
    cache, offloaded = _checkpoint_backed_cache(tmp_path, tensors)

    # capture the copy mode onload requests for pinned tensors
    seen = []
    real_send = ct_disk.send_tensors

    def capturing_send(value, *args, **kwargs):
        seen.append(kwargs.get("non_blocking"))
        return real_send(value, *args, **kwargs)

    ct_disk.send_tensors = capturing_send
    try:
        for name, tensor in tensors.items():
            staged = cache.stage(offloaded[name], pin_memory=True)
            assert staged.is_pinned()
            cache.is_staged = True  # route onload through the staged path
            onloaded = cache.onload(staged)
            assert onloaded.device.type == "cuda"
            assert torch.equal(onloaded, tensor.to(onloaded.device))
        assert seen == [True, True]

        # unpinned staging must keep synchronous copies
        cache2, offloaded2 = _checkpoint_backed_cache(
            tmp_path / "unpinned", {"c": torch.ones(4, 4)}
        )
        staged = cache2.stage(offloaded2["c"], pin_memory=False)
        cache2.is_staged = True
        onloaded = cache2.onload(staged)
        assert torch.equal(onloaded.cpu(), torch.ones(4, 4))
        assert seen == [True, True, False]
    finally:
        ct_disk.send_tensors = real_send


@requires_gpu
@pytest.mark.unit
def test_stage_module_round_trip_pinned(tmp_path, pinned_staging_enabled):
    module = torch.nn.Linear(8, 8, bias=True)
    orig_weight, orig_bias = module.weight.clone(), module.bias.clone()
    offload_module(module, "cuda", "disk", offload_dir=str(tmp_path))

    stage_module_offload(module, pin_memory=True)

    weight = module._parameters.offloaded_values["weight"]
    assert weight.is_pinned()

    remove_module_offload(module, onload_tensors=True)
    assert module.weight.device.type == "cuda"
    assert torch.equal(module.weight.cpu(), orig_weight)
    assert torch.equal(module.bias.cpu(), orig_bias)


@requires_gpu
@pytest.mark.unit
def test_pinned_staging_disabled_by_env(tmp_path, monkeypatch, pinned_staging_enabled):
    os.environ["COMPRESSED_TENSORS_PINNED_STAGING"] = "0"
    tensor = torch.arange(10)
    cache, offloaded = _checkpoint_backed_cache(tmp_path, {"weight": tensor})

    calls = []
    real_pin = cache_utils._pin_memory
    monkeypatch.setattr(
        cache_utils,
        "_pin_memory",
        lambda value: (calls.append(value.nbytes), real_pin(value))[1],
    )

    staged = cache.stage(offloaded["weight"], pin_memory=True)

    # the fallback reads through safetensors and pins the result
    assert calls == [tensor.nbytes]
    assert staged.is_pinned()
    assert torch.equal(staged, tensor)


@requires_gpu
@pytest.mark.unit
def test_raw_reader_reused_and_closed(tmp_path, pinned_staging_enabled):
    from compressed_tensors.offload.cache.disk_utils import (
        _open_files,
        disk_load_context,
    )
    from compressed_tensors.offload.cache.utils import _load_pinned_raw

    shard = tmp_path / "shard.safetensors"
    save_file({"a": torch.ones(4), "b": torch.zeros(4)}, str(shard))

    opens = []
    real_open = builtins.open

    def counting_open(*args, **kwargs):
        opens.append(args[0])
        return real_open(*args, **kwargs)

    builtins.open = counting_open
    try:
        with disk_load_context():
            for name in ("a", "b"):
                offloaded = {
                    "safetensors_file": str(shard),
                    "weight_name": name,
                    "dtype": "float32",
                }
                assert _load_pinned_raw(offloaded, torch.float32) is not None
        assert len(opens) == 1, f"expected a single open, got {len(opens)}"
        assert not _open_files.raw_cache
    finally:
        builtins.open = real_open


@requires_gpu
@pytest.mark.unit
def test_raw_reader_handles_truncated_files(tmp_path, pinned_staging_enabled):
    from compressed_tensors.offload.cache.utils import _load_pinned_raw

    # a file whose header claims more bytes than it holds
    truncated = tmp_path / "truncated.safetensors"
    header = json_module.dumps(
        {"weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
    ).encode()
    with open(truncated, "wb") as file:
        file.write(struct_module.pack("<Q", len(header)))
        file.write(header)
        file.write(b"\x00" * 8)  # only half the claimed payload

    offloaded = {
        "safetensors_file": str(truncated),
        "weight_name": "weight",
        "dtype": "float32",
    }
    with pytest.raises(EOFError):
        _load_pinned_raw(offloaded, torch.float32)


@requires_gpu
@pytest.mark.unit
def test_raw_reader_evicted_before_update(tmp_path, pinned_staging_enabled):
    from compressed_tensors.offload.cache.disk_utils import disk_load_context

    cache = DiskCache("cuda", offload_dir=str(tmp_path))
    tensor = torch.arange(6, dtype=torch.float32)
    offloaded = cache.offload(tensor)

    with disk_load_context():
        first = cache.stage(offloaded, pin_memory=True)
        # rewrite the file in place; the raw reader must not serve stale bytes
        cache.update_offload(offloaded, tensor + 100)
        second = cache.stage(offloaded, pin_memory=True)

    assert torch.equal(first, tensor)
    assert torch.equal(second, tensor + 100)
