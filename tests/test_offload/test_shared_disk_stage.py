# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import errno
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import compressed_tensors.offload.module as module_offload
from compressed_tensors.distributed import set_source_process
from compressed_tensors.offload.module import (
    finish_shared_disk_stage,
    offload_module,
    prepare_shared_disk_stage,
    subgraph_onload_modules,
    subgraph_stage_modules,
)
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


@torchrun(world_size=2)
def test_shared_disk_stage_reads_once_and_shares_storage(offload_folder, monkeypatch):
    dist.init_process_group("gloo")
    module = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        module.weight.fill_(2)
    offload_module(
        module, onload_device="cpu", offload_device="disk", offload_dir=offload_folder
    )

    if dist.get_rank() == 1:

        def unexpected_stage(*args, **kwargs):
            raise AssertionError("non-source rank read the disk weight")

        monkeypatch.setattr(module._parameters, "stage", unexpected_stage)

    modules = {"linear": module}
    # Modifier work can change the global source rank while prefetch runs.
    with set_source_process(1):
        prepared = prepare_shared_disk_stage(modules, pin_memory=False, source_rank=0)
        finish_shared_disk_stage(modules, prepared, pin_memory=False, source_rank=0)
    staged = module._parameters.offloaded_values["weight"]
    assert staged.is_shared()
    if dist.get_rank() == 0:
        assert all(
            not os.path.exists(descriptor[0])
            for descriptor in prepared.descriptors.values()
        )

    if dist.get_rank() == 0:
        with torch.no_grad():
            staged[0, 0] = 7
    dist.barrier()
    assert staged[0, 0].item() == 7

    subgraph_onload_modules(modules)
    assert isinstance(module.weight, torch.nn.Parameter)
    assert module.weight[0, 0].item() == 7
    dist.barrier()
    dist.destroy_process_group()


@requires_gpu(2)
def test_subgraph_stage_automatically_shares_pinned_disk_weights():
    # Nested pytest workers can stall during shutdown after both ranks pass.
    worker = Path(__file__).with_name("shared_disk_stage_worker.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(worker),
        ],
        check=True,
    )


@torchrun(world_size=2)
def test_shared_stage_falls_back_to_local_disk_on_other_host(
    offload_folder, monkeypatch
):
    dist.init_process_group("gloo")
    module = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        module.weight.fill_(6)
    offload_module(
        module, onload_device="cpu", offload_device="disk", offload_dir=offload_folder
    )
    monkeypatch.setattr(
        module_offload.socket, "gethostname", lambda: f"host-{dist.get_rank()}"
    )

    modules = {"linear": module}
    prepared = prepare_shared_disk_stage(modules, pin_memory=False, source_rank=0)
    finish_shared_disk_stage(modules, prepared, pin_memory=False, source_rank=0)
    staged = module._parameters.offloaded_values["weight"]
    assert staged.is_shared() == (dist.get_rank() == 0)

    subgraph_onload_modules(modules)
    assert torch.all(module.weight == 6)
    dist.barrier()
    dist.destroy_process_group()


@torchrun(world_size=2)
def test_shared_stage_falls_back_when_shm_is_full(offload_folder, monkeypatch):
    dist.init_process_group("gloo")
    module = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        module.weight.fill_(9)
    offload_module(
        module, onload_device="cpu", offload_device="disk", offload_dir=offload_folder
    )

    if dist.get_rank() == 0:

        def no_shared_space(*args):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(os, "posix_fallocate", no_shared_space)

    modules = {"linear": module}
    subgraph_stage_modules(modules, pin_memory=True)
    prepared = module._shared_disk_stage_result
    assert not prepared.descriptors

    subgraph_onload_modules(modules)
    assert torch.all(module.weight == 9)
    dist.barrier()
    dist.destroy_process_group()
