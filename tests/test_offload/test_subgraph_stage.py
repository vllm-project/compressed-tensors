# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import compressed_tensors.offload.module as module_offload
import torch
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.module import offload_module
from safetensors.torch import save_file


class _DeviceOffloadCache(dict):
    offload_device = torch.device("cuda")


def test_subgraph_stage_handles_accelerator_offload(monkeypatch):
    module = torch.nn.Module()
    module._parameters = _DeviceOffloadCache()
    module._buffers = _DeviceOffloadCache()
    calls = []

    monkeypatch.setattr(module_offload, "OffloadCache", _DeviceOffloadCache)
    monkeypatch.setattr(
        module_offload,
        "stage_module_offload",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    module_offload.subgraph_stage_modules({"module": module})

    assert len(calls) == 1
    assert calls[0][0] == (module,)
    assert calls[0][1] == {"pin_memory": False}


def _checkpoint_subgraph(tmp_path, count):
    """`count` disk-offloaded Linears whose weights all live in one shard.

    Mirrors loading a sharded checkpoint with disk offload, where
    `create_checkpoint_symlink` points each weight at the shared shard through
    its own symlink.
    """
    shard = tmp_path / "model.safetensors"
    save_file(
        {f"m{i}.weight": torch.full((4, 4), float(i)) for i in range(count)}, shard
    )
    offload_dir = tmp_path / "offload"
    os.mkdir(offload_dir)

    DiskCache.index = {}
    modules = {}
    for i in range(count):
        linear = torch.nn.Linear(4, 4, bias=False)
        offload_module(linear, "cpu", "disk", offload_dir=str(offload_dir))
        offloaded = linear._parameters.offloaded_values["weight"]
        os.unlink(DiskCache.index.pop(offloaded)["safetensors_file"])
        DiskCache.create_checkpoint_symlink(
            offloaded,
            {
                "safetensors_file": str(shard),
                "weight_name": f"m{i}.weight",
                "dtype": "float32",
            },
            str(offload_dir),
        )
        modules[f"m{i}"] = linear
    return modules


def _count_opens(monkeypatch):
    from compressed_tensors.offload.cache import disk_utils

    opens = []
    real = disk_utils.safe_open

    def counted(*args, **kwargs):
        opens.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(disk_utils, "safe_open", counted)
    return opens


def test_subgraph_stage_opens_each_shard_once(tmp_path, monkeypatch):
    """Staging a subgraph must not reopen its shard once per tensor."""
    modules = _checkpoint_subgraph(tmp_path, count=6)
    opens = _count_opens(monkeypatch)

    module_offload.subgraph_stage_modules(modules)
    assert len(opens) == 1, f"expected one open for one shard, got {len(opens)}"

    module_offload.subgraph_onload_modules(modules)
    for i, module in enumerate(modules.values()):
        assert torch.equal(module.weight.data, torch.full((4, 4), float(i)))


def test_subgraph_onload_without_staging_opens_each_shard_once(tmp_path, monkeypatch):
    """Onloading tensors that were never staged reads from disk, also grouped."""
    modules = _checkpoint_subgraph(tmp_path, count=6)
    opens = _count_opens(monkeypatch)

    module_offload.subgraph_onload_modules(modules)
    assert len(opens) == 1, f"expected one open for one shard, got {len(opens)}"
    for i, module in enumerate(modules.values()):
        assert torch.equal(module.weight.data, torch.full((4, 4), float(i)))
