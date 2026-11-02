# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from compressed_tensors.distributed import set_source_process
from compressed_tensors.offload.module import (
    finish_shared_disk_stage,
    offload_module,
    prepare_shared_disk_stage,
    subgraph_onload_modules,
)
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


@torchrun(world_size=2, init_dist=True)
def test_shared_disk_stage_reads_once_and_shares_storage(offload_folder, monkeypatch):
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
        with torch.no_grad():
            staged[0, 0] = 7
    dist.barrier()
    assert staged[0, 0].item() == 7

    subgraph_onload_modules(modules)
    assert isinstance(module.weight, torch.nn.Parameter)
    assert module.weight[0, 0].item() == 7


@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_shared_disk_stage_is_pinned_on_every_rank(offload_folder):
    from compressed_tensors.offload.shared_stage import _registrations

    module = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        module.weight.fill_(3)
    offload_module(
        module, onload_device="cuda", offload_device="disk", offload_dir=offload_folder
    )

    modules = {"linear": module}
    prepared = prepare_shared_disk_stage(modules)
    finish_shared_disk_stage(modules, prepared)
    staged = module._parameters.offloaded_values["weight"]
    assert staged.is_shared()
    assert staged.is_pinned()
    assert len(_registrations) == 1

    subgraph_onload_modules(modules)
    assert module.weight.device.type == "cuda"
    assert torch.all(module.weight == 3)
    assert not _registrations
