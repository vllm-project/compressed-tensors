# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Two-rank CUDA worker for the automatic shared disk staging test."""

import os
import tempfile

import compressed_tensors.offload.module as module_offload
import torch
import torch.distributed as dist
from compressed_tensors.distributed import set_source_process
from compressed_tensors.offload.module import (
    offload_module,
    subgraph_onload_modules,
    subgraph_stage_modules,
)
from compressed_tensors.offload.shared_stage import _registrations


def main() -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    rank = dist.get_rank()

    temporary = (
        tempfile.TemporaryDirectory(prefix="ct-shared-stage-") if rank == 0 else None
    )
    payload = [temporary.name if temporary is not None else None]
    dist.broadcast_object_list(payload, src=0)
    offload_folder = payload[0]

    module = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        module.weight.fill_(5)
    offload_module(
        module, onload_device="cuda", offload_device="disk", offload_dir=offload_folder
    )
    modules = {"linear": module}

    with set_source_process(1):
        assert subgraph_stage_modules(modules, pin_memory=True) is None

    original_finish = module_offload.finish_shared_disk_stage

    def checked_finish(*args, **kwargs):
        original_finish(*args, **kwargs)
        staged = module._parameters.offloaded_values["weight"]
        assert staged.is_shared()
        assert staged.is_pinned()
        assert len(_registrations) == 1

    module_offload.finish_shared_disk_stage = checked_finish
    try:
        subgraph_onload_modules(modules)
    finally:
        module_offload.finish_shared_disk_stage = original_finish

    assert module.weight.device.type == "cuda"
    assert torch.all(module.weight == 5)
    assert not _registrations
    assert not hasattr(module, "_shared_disk_stage_result")
    dist.barrier()
    dist.destroy_process_group()
    if temporary is not None:
        temporary.cleanup()
    # The unpinned distributed disk baseline also intermittently receives
    # SIGTERM during Python/CUDA interpreter teardown in this runner.
    os._exit(0)


if __name__ == "__main__":
    main()
