# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload.cache import CPUCache, DeviceCache, DiskCache
from compressed_tensors.offload.convert import from_accelerate, to_accelerate
from compressed_tensors.offload.convert.from_accelerate import (
    remove_accelerate,
    remove_accelerate_from_module,
)
from compressed_tensors.offload.load import load_offloaded_model
from compressed_tensors.offload.module import offload_module
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu
from transformers import AutoModelForCausalLM


acclerate = pytest.importorskip("accelerate")


@pytest.mark.unit
@requires_gpu
def test_remove_accelerate_from_module_device():
    # there"s no way to force accelerate to "offload" to cuda. Instead, it just
    # stays on cuda with no hooks
    linear = torch.nn.Linear(5, 5, device="cuda:0")
    assert remove_accelerate_from_module(linear) == (
        torch.device("cuda:0"),
        torch.device("cuda:0"),
        None,
    )
    assert not hasattr(linear, "_hf_hook")


@pytest.mark.unit
@requires_gpu
def test_remove_accelerate_from_module_cpu():
    from accelerate.big_modeling import dispatch_model

    linear = torch.nn.Linear(5, 5)
    dispatch_model(
        linear,
        {"": "cpu"},
        main_device="cuda",
        state_dict=linear.state_dict(),
        force_hooks=True,
    )
    assert remove_accelerate_from_module(linear) == ("cuda", "cpu", None)
    assert not hasattr(linear, "_hf_hook")


@pytest.mark.unit
@requires_gpu
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_remove_accelerate_from_module_disk(tmp_path):
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
    assert remove_accelerate_from_module(linear) == ("cuda", "disk", offload_dir)
    assert not hasattr(linear, "_hf_hook")


@pytest.mark.unit
@requires_gpu
def test_remove_accelerate(tmp_path):
    from accelerate.big_modeling import dispatch_model

    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    model = torch.nn.Sequential(
        torch.nn.Linear(5, 5), torch.nn.Linear(5, 5), torch.nn.Linear(5, 5)
    )
    dispatch_model(
        model,
        {"0": 0, "1": "cpu", "2": "disk"},
        main_device="cuda",
        force_hooks=True,
        offload_dir=offload_dir,
    )
    assert hasattr(model, "hf_device_map")

    device_map, _offload_dir = remove_accelerate(model)
    assert device_map == {
        "": (None, None),
        "0": (torch.device("cuda:0"), torch.device("cuda:0")),
        "1": ("cuda", "cpu"),
        "2": ("cuda", "disk"),
    }
    assert _offload_dir == offload_dir
    assert not hasattr(model, "hf_device_map")


@pytest.mark.unit
@requires_gpu
def test_from_accelerate(tmp_path):
    from accelerate.big_modeling import dispatch_model

    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    model = torch.nn.Sequential(
        torch.nn.Linear(5, 5), torch.nn.Linear(5, 5), torch.nn.Linear(5, 5)
    )
    dispatch_model(
        model,
        {"0": 0, "1": "cpu", "2": "disk"},
        main_device="cuda",
        force_hooks=True,
        offload_dir=offload_dir,
    )

    device_map, _offload_dir = from_accelerate(model)
    assert device_map == {
        "": (None, None),
        "0": (torch.device("cuda:0"), torch.device("cuda:0")),
        "1": ("cuda", "cpu"),
        "2": ("cuda", "disk"),
    }
    assert _offload_dir == offload_dir
    assert isinstance(model[0]._parameters, DeviceCache)
    assert isinstance(model[1]._parameters, CPUCache)
    assert isinstance(model[2]._parameters, DiskCache)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_from_accelerate_dist(tmp_path):
    from accelerate.big_modeling import dispatch_model

    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    model = torch.nn.Sequential(
        torch.nn.Linear(5, 5), torch.nn.Linear(5, 5), torch.nn.Linear(5, 5)
    )
    if dist.get_rank() == 0:
        dispatch_model(
            model,
            # {"0": 0, "1": "cpu", "2": "disk"},
            {"0": "cpu", "1": "cpu", "2": "disk"},
            main_device="cuda",
            force_hooks=True,
            offload_dir=offload_dir,
        )
    else:
        model.to("meta")

    device_map, _offload_dir = from_accelerate(model)
    assert device_map == {
        "": (None, None),
        "0": ("cuda", "cpu"),
        "1": ("cuda", "cpu"),
        "2": ("cuda", "disk"),
    }
    if dist.get_rank() == 0:
        assert _offload_dir == offload_dir
    # assert isinstance(model[0]._parameters, DeviceCache)
    assert isinstance(model[1]._parameters, CPUCache)
    assert isinstance(model[2]._parameters, DiskCache)


# @pytest.mark.integration
# def test_convert():
#     with load_offloaded_model():
#         acclerate = pytest.importorskip("accelerate")

#         model = AutoModelForCausalLM.from_pretrained(
#             "Qwen/Qwen3-0.6B",
#             device_map="disk",
#             max_memory={"cpu": 596049920},  # force disk offloading
#             offload_folder="temp",
#             dtype=torch.bfloat16,
#         )

#     assert not hasattr(model, "hf_device_map")
#     to_accelerate(model)
#     model.save_pretrained("temp_save")


# @pytest.mark.integration
# def test_idempotency(disable_convert):
#     with load_offloaded_model():
#         model = AutoModelForCausalLM.from_pretrained(
#             "Qwen/Qwen3-0.6B",
#             device_map="disk",
#             max_memory={"cpu": 596049920},  # force disk offloading
#             offload_folder="temp",
#             dtype=torch.bfloat16,
#         )

#     assert hasattr(model, "hf_device_map")

#     from_accelerate(model)
#     from_accelerate(model)

#     to_accelerate(model)
#     to_accelerate(model)
