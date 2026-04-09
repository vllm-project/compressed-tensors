# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
from compressed_tensors.distributed import is_source_process
from compressed_tensors.offload import disable_onloading, from_accelerate, to_accelerate
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


acclerate = pytest.importorskip("accelerate")


def get_hf_dispatched_model(cuda_device, tmp_path):
    from accelerate.big_modeling import dispatch_model

    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    model = torch.nn.Sequential(
        torch.nn.Linear(5, 5), torch.nn.Linear(5, 5), torch.nn.Linear(5, 5)
    )

    if is_source_process():
        dispatch_model(
            model,
            {"0": 0, "1": "cpu", "2": "disk"},
            main_device=str(cuda_device),
            force_hooks=True,
            offload_dir=offload_dir,
        )
    else:
        model.to("meta")

    return model, offload_dir


@pytest.mark.unit
@requires_gpu
def test_conversion_lifecycle(cuda_device, tmp_path):
    model, offload_dir = get_hf_dispatched_model(cuda_device, tmp_path)

    exp_device_map = {
        "": (None, None),
        "0": (cuda_device, cuda_device),
        "1": (cuda_device, torch.device("cpu")),
        "2": (cuda_device, "disk"),
    }
    exp_hf_device_map = {"": "cpu", "0": str(cuda_device), "1": "cpu", "2": "disk"}

    # 1. from_accelerate (oneshot/ load_offloaded_model)
    device_map, _offload_dir = from_accelerate(model)
    assert device_map == exp_device_map
    with disable_onloading():
        state_dict = model.state_dict(keep_vars=True)

    # 2. to_accelerate (transformers save)
    hf_device_map = to_accelerate(model)
    assert hf_device_map == exp_hf_device_map

    # 3. from_accelerate (post-save restore)
    device_map, _offload_dir = from_accelerate(model)
    assert device_map == exp_device_map

    # Note that rank 0 tensor pointers remain unchanged:
    # - no extra gpu/cpu/disk memory is allocated
    # - disk index remains valid
    with disable_onloading():
        assert model.state_dict(keep_vars=True) == state_dict

    # 4. Just for completeness, test final invert (not part of lifecycle)
    hf_device_map = to_accelerate(model)
    assert hf_device_map == exp_hf_device_map


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_conversion_lifecycle_dist(cuda_device, tmp_path):
    test_conversion_lifecycle(cuda_device, tmp_path)
