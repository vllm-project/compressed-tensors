# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
from compressed_tensors.offload import dispatch_with_map, offload_module, to_accelerate
from compressed_tensors.offload.convert.to_accelerate import to_accelerate_module
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


acclerate = pytest.importorskip("accelerate")


@pytest.mark.unit
@requires_gpu
@pytest.mark.parametrize("offload_device", ["cuda", "cuda:0", "cpu", "disk"])
def test_to_accelerate_module(offload_device, tmp_path):
    linear = torch.nn.Linear(5, 5)
    if offload_device == "disk":
        offload_dir = tmp_path / "offload_dir"
        os.mkdir(offload_dir)
        offload_module(linear, "cuda", offload_device, offload_dir=str(offload_dir))
    else:
        offload_module(linear, "cuda", offload_device)

    _offload_device = to_accelerate_module(linear, name="", hf_disk_index={})
    if offload_device == "cuda":
        assert _offload_device == "cuda:0"
    else:
        assert _offload_device == offload_device


@pytest.mark.unit
@requires_gpu
def test_to_accelerate(cuda_device, tmp_path):
    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    model = torch.nn.Sequential(
        torch.nn.Linear(5, 5), torch.nn.Linear(5, 5), torch.nn.Linear(5, 5)
    )
    device_map = {
        "0": (torch.device("cuda"), torch.device("cpu")),
        "1": (torch.device("cuda"), torch.device("cuda")),
        "2": (torch.device("cuda"), "disk"),
    }
    dispatch_with_map(model, device_map, offload_dir)

    hf_device_map = to_accelerate(model)
    assert hf_device_map == {"": "cpu", "0": "cpu", "1": str(cuda_device), "2": "disk"}
    assert hasattr(model[0], "_hf_hook")
    assert hasattr(model[1], "_hf_hook")
    assert hasattr(model[2], "_hf_hook")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_to_accelerate_dist(cuda_device, tmp_path):
    test_to_accelerate(cuda_device, tmp_path)
