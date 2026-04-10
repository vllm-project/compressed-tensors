# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
from compressed_tensors.offload import dispatch_with_map, offload_module, to_accelerate
from compressed_tensors.offload.convert.to_accelerate import to_accelerate_module
from compressed_tensors.offload.convert.helpers import norm_device
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


acclerate = pytest.importorskip("accelerate")

def get_offload_devices() -> list[str]: 
    offload_devices = ["cpu", "disk"]
    accelerator_device = torch.accelerator.current_accelerator()

    if accelerator_device is None:
        return offload_devices

    offload_devices.append(accelerator_device.type)
    if accelerator_device.type == "cuda":
        offload_devices.append("cuda:0")

    return offload_devices

@pytest.mark.unit
@requires_gpu
@pytest.mark.parametrize("offload_device", get_offload_devices())
def test_to_accelerate_module(offload_device, tmp_path):
    accelerator_device = torch.accelerator.current_accelerator()
    linear = torch.nn.Linear(5, 5)

    if offload_device == "disk":
        offload_dir = tmp_path / "offload_dir"
        os.mkdir(offload_dir)
        offload_module(linear, accelerator_device, offload_device, offload_dir=str(offload_dir))
    else:
        offload_module(linear, accelerator_device, offload_device)

    _offload_device = to_accelerate_module(linear, name="", hf_disk_index={})

    assert _offload_device == str(norm_device(offload_device))

@pytest.mark.unit
@requires_gpu
def test_to_accelerate(accel_device, tmp_path):
    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)
    current_accelerator = torch.accelerator.current_accelerator()

    model = torch.nn.Sequential(
        torch.nn.Linear(5, 5), torch.nn.Linear(5, 5), torch.nn.Linear(5, 5)
    )
    device_map = {
        "0": (current_accelerator, torch.device("cpu")),
        "1": (current_accelerator, current_accelerator),
        "2": (current_accelerator, "disk"),
    }
    dispatch_with_map(model, device_map, offload_dir)

    hf_device_map = to_accelerate(model)
    assert hf_device_map == {"": "cpu", "0": "cpu", "1": str(accel_device), "2": "disk"}
    assert hasattr(model[0], "_hf_hook")
    assert hasattr(model[1], "_hf_hook")
    assert hasattr(model[2], "_hf_hook")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_to_accelerate_dist(accel_device, tmp_path):
    test_to_accelerate(accel_device, tmp_path)
