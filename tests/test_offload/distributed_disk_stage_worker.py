# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Two-rank GPU worker for source-only pinned staging and GPU broadcasts."""

import os
import tempfile

import torch
import torch.distributed as dist
from compressed_tensors.offload.module import (
    offload_module,
    subgraph_onload_modules,
    subgraph_stage_modules,
)


def onload_and_count_broadcasts(modules: dict) -> int:
    original = dist.broadcast
    count = 0

    def counted(tensor, *args, **kwargs):
        nonlocal count
        assert tensor.device.type == "cuda"
        count += 1
        return original(tensor, *args, **kwargs)

    dist.broadcast = counted
    try:
        subgraph_onload_modules(modules)
    finally:
        dist.broadcast = original
    return count


def main() -> None:
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    rank = dist.get_rank()

    temporary = (
        tempfile.TemporaryDirectory(prefix="ct-gpu-stage-") if rank == 0 else None
    )
    payload = [temporary.name if temporary is not None else None]
    dist.broadcast_object_list(payload, src=0)
    offload_folder = payload[0]

    module = torch.nn.Linear(4, 4, bias=True)
    with torch.no_grad():
        module.weight.fill_(5)
        module.bias.fill_(3)
    offload_module(
        module, onload_device="cuda", offload_device="disk", offload_dir=offload_folder
    )
    modules = {"linear": module}

    if rank == 1:

        def unexpected_disk_read(*args, **kwargs):
            raise AssertionError("non-source rank read a matching weight from disk")

        module._parameters.stage = unexpected_disk_read

    subgraph_stage_modules(modules, pin_memory=True)
    staged = module._parameters.offloaded_values["weight"]
    if rank == 0:
        assert staged.device.type == "cpu" and staged.is_pinned()
    else:
        assert staged.device.type == "meta"

    assert onload_and_count_broadcasts(modules) == 2
    assert module.weight.device.type == "cuda"
    assert torch.all(module.weight == 5)
    assert torch.all(module.bias == 3)

    dist.barrier()
    dist.destroy_process_group()
    if temporary is not None:
        temporary.cleanup()
    # This runner can receive SIGTERM during Python/CUDA interpreter teardown.
    os._exit(0)


if __name__ == "__main__":
    main()
