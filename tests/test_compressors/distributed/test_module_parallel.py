# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from compressed_tensors.distributed import greedy_bin_packing, replace_module_parallel
from compressed_tensors.offload import offload_module, update_offload_parameter
from compressed_tensors.offload.utils import module_size
from tests.test_offload.conftest import offload_folder, torchrun  # noqa: F401
from tests.testing_utils import requires_gpu


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_delete = nn.Parameter(torch.empty(3))
        self.to_update = nn.Parameter(torch.empty(5))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_cpu_module_parallel():
    modules = [DummyModule(), DummyModule()]
    offload_module(modules[0], "cuda", "cpu")
    offload_module(modules[1], "cuda", "cpu")

    _test_module_parallel(modules)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_disk_module_parallel(offload_folder):  # noqa: F811
    modules = [DummyModule(), DummyModule()]
    offload_module(modules[0], "cuda", "disk", offload_dir=offload_folder)
    offload_module(modules[1], "cuda", "disk", offload_dir=offload_folder)

    _test_module_parallel(modules)


# Note: distributed gpu offloading is not supported


def _test_module_parallel(modules: list[DummyModule]):
    """Test deletion, update, and construction"""

    def apply_fn(module: DummyModule):
        device = module.to_delete.device
        value = float(dist.get_rank())

        # Delete to_delete
        delattr(module, "to_delete")
        # Update to_update with a tensor filled with the rank value
        new_data = torch.full((5,), value, device=device)
        update_offload_parameter(module, "to_update", new_data)
        # Construct to_construct filled with the rank value
        module.to_construct = torch.nn.Parameter(torch.full((1,), value, device=device))

    world_size = dist.get_world_size()
    replace_module_parallel(modules, apply_fn, module_size)
    _, _, assigned_rank = greedy_bin_packing(modules, world_size, module_size)

    for module in modules:
        # Test that to_delete was deleted
        assert not hasattr(module, "to_delete")

        # Test that to_update was changed
        to_update: torch.nn.Parameter = module.to_update
        assert to_update.device.type == "cuda"
        assert to_update.shape == (5,)
        assert torch.all(to_update == assigned_rank[module])

        # Test that to_construct was created correctly
        to_construct: torch.nn.Parameter = module.to_construct
        assert to_construct.device.type == "cuda"
        assert to_construct.shape == (1,)
        assert to_construct[0] == assigned_rank[module]

    dist.barrier()
