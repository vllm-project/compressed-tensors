# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload.module import (
    offload_module,
    subgraph_onload_modules,
    subgraph_stage_modules,
)
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


@requires_gpu(2)
def test_pinned_disk_stage_broadcasts_gpu_weights():
    worker = Path(__file__).with_name("distributed_disk_stage_worker.py")
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
def test_gloo_cpu_stage_keeps_local_disk_path(offload_folder):
    dist.init_process_group("gloo")
    module = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        module.weight.fill_(9)
    offload_module(
        module, onload_device="cpu", offload_device="disk", offload_dir=offload_folder
    )

    modules = {"linear": module}
    subgraph_stage_modules(modules, pin_memory=True)
    assert module._parameters.offloaded_values["weight"].device.type == "cpu"

    subgraph_onload_modules(modules)
    assert torch.all(module.weight == 9)
    dist.barrier()
    dist.destroy_process_group()


@torchrun(world_size=2)
def test_stage_error_reaches_all_ranks(offload_folder, monkeypatch):
    dist.init_process_group("gloo")
    module = torch.nn.Linear(4, 4, bias=False)
    offload_module(
        module, onload_device="cpu", offload_device="disk", offload_dir=offload_folder
    )
    if dist.get_rank() == 0:

        def failed_stage(*args, **kwargs):
            raise ValueError("staging failed")

        monkeypatch.setattr(module._parameters, "stage", failed_stage)

    modules = {"linear": module}
    subgraph_stage_modules(modules, pin_memory=True)
    expected_error = ValueError if dist.get_rank() == 0 else RuntimeError
    with pytest.raises(expected_error, match="staging failed"):
        subgraph_onload_modules(modules)
    dist.barrier()
    dist.destroy_process_group()
