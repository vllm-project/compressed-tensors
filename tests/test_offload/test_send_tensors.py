# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.offload.utils import send_tensors
from tests.testing_utils import requires_gpu


def _empty_tensors(device) -> list[torch.Tensor]:
    param = torch.nn.Parameter(torch.empty(0, device=device))
    param.attr = "value"
    return [
        torch.empty(0, device=device),
        torch.empty(1024, device=device)[:0],  # view of a larger allocation
        param,
    ]


@pytest.mark.unit
def test_send_tensors_keeps_tensors_already_on_device():
    tensors = [torch.zeros(2), torch.nn.Parameter(torch.zeros(2))]
    for tensor in tensors + _empty_tensors("cpu"):
        assert send_tensors(tensor, device="cpu") is tensor


@pytest.mark.unit
def test_send_tensors_moves_empty_tensors_to_meta():
    # tensors without elements have a null data pointer on every device, so they
    # must not be mistaken for tensors which are already on the target device
    for tensor in _empty_tensors("cpu"):
        moved = send_tensors(tensor, device="meta")
        assert moved is not tensor
        assert moved.is_meta and moved.shape == tensor.shape
        assert moved.__class__ is tensor.__class__
        assert moved.__dict__ == tensor.__dict__


@pytest.mark.unit
@requires_gpu
def test_send_tensors_moves_empty_tensors_between_devices():
    accel_device = torch.accelerator.current_accelerator()
    for source, target in ((torch.device("cpu"), accel_device), (accel_device, "cpu")):
        for tensor in _empty_tensors(source):
            moved = send_tensors(tensor, device=target)
            assert moved is not tensor
            assert moved.device.type == torch.device(target).type
            assert moved.__class__ is tensor.__class__
