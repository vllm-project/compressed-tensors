# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import compressed_tensors.offload.cache.disk as ct_disk
import pytest
import torch
from compressed_tensors.offload import offload_module
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.module import (
    remove_module_offload,
    stage_module_offload,
)
from safetensors.torch import save_file
from tests.testing_utils import requires_gpu


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
def test_staged_onload_is_correct_with_async_copies(tmp_path):
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
def test_stage_module_round_trip_pinned(tmp_path):
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
