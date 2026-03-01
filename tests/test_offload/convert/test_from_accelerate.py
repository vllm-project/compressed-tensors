# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
from compressed_tensors.offload.cache import CPUCache, DeviceCache, DiskCache
from compressed_tensors.offload.convert import from_accelerate
from compressed_tensors.offload.convert.from_accelerate import (
    remove_accelerate_from_module,
)
from compressed_tensors.offload.dist_utils import is_rank0
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


acclerate = pytest.importorskip("accelerate")


@pytest.mark.unit
@requires_gpu
def test_remove_accelerate_from_module_device(cuda_device):
    # there"s no way to force accelerate to "offload" to cuda. Instead, it just
    # stays on cuda with no hooks
    linear = torch.nn.Linear(5, 5, device="cuda:0")
    assert remove_accelerate_from_module(linear) == (cuda_device, cuda_device, None)
    assert not hasattr(linear, "_hf_hook")

    # test idempotency
    assert remove_accelerate_from_module(linear) == (cuda_device, cuda_device, None)
    assert not hasattr(linear, "_hf_hook")


@pytest.mark.unit
@requires_gpu
def test_remove_accelerate_from_module_cpu(cuda_device):
    from accelerate.big_modeling import dispatch_model

    linear = torch.nn.Linear(5, 5)
    dispatch_model(
        linear,
        {"": "cpu"},
        main_device="cuda",
        state_dict=linear.state_dict(),
        force_hooks=True,
    )
    assert remove_accelerate_from_module(linear) == (
        cuda_device,
        torch.device("cpu"),
        None,
    )
    assert not hasattr(linear, "_hf_hook")


@pytest.mark.unit
@requires_gpu
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_remove_accelerate_from_module_disk(cuda_device, tmp_path):
    # `disk_offload` is a super buggy function, and not reflective of real dispatches
    # `dispatch_model` is also super buggy, and requires at least one cpu device
    from accelerate.big_modeling import dispatch_model

    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    linear = torch.nn.Linear(5, 5)
    model = torch.nn.Sequential(linear)
    dispatch_model(
        model,
        {"0": "disk", "fake_module": "cpu"},
        main_device="cuda",
        force_hooks=True,
        offload_dir=offload_dir,
    )
    assert remove_accelerate_from_module(linear) == (cuda_device, "disk", offload_dir)
    assert not hasattr(linear, "_hf_hook")


@pytest.mark.unit
@requires_gpu
def test_from_accelerate(cuda_device, tmp_path):
    from accelerate.big_modeling import dispatch_model

    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    model = torch.nn.Sequential(
        torch.nn.Linear(5, 5), torch.nn.Linear(5, 5), torch.nn.Linear(5, 5)
    )
    if is_rank0():
        dispatch_model(
            model,
            {"0": 0, "1": "cpu", "2": "disk"},
            main_device=str(cuda_device),
            force_hooks=True,
            offload_dir=offload_dir,
        )
    else:
        model.to("meta")

    device_map, _offload_dir = from_accelerate(model)

    # cuda is index agnostic when distributed
    assert device_map == {
        "": (None, None),
        "0": (cuda_device, cuda_device),
        "1": (cuda_device, torch.device("cpu")),
        "2": (cuda_device, "disk"),
    }
    if is_rank0():
        assert _offload_dir == offload_dir
    assert isinstance(model[0]._parameters, DeviceCache)
    assert isinstance(model[1]._parameters, CPUCache)
    assert isinstance(model[2]._parameters, DiskCache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_from_accelerate_dist(cuda_device, tmp_path):
    test_from_accelerate(cuda_device, tmp_path)
