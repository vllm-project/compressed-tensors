# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload.convert import from_accelerate, to_accelerate
from compressed_tensors.offload.load import load_offloaded_model
from tests.test_offload.conftest import torchrun
from transformers import AutoModelForCausalLM


acclerate = pytest.importorskip("accelerate")


@pytest.mark.integration
@torchrun(world_size=2)
def test_load_disk_dist(disable_convert, tmp_path):

    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    with load_offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            device_map="disk",
            max_memory={"cpu": 596049920},  # force disk offloading (~half model size)
            offload_folder=offload_dir,
            dtype=torch.bfloat16,
        )

    assert model.num_parameters() == 596049920

    if dist.get_rank() == 0:
        assert model.device.type != "meta"
        assert set(model.hf_device_map.values()) == {"cpu", "disk"}
    else:
        assert model.device.type == "meta"

    device_map, _offload_dir = from_accelerate(model)
    for layer_index in range(0, 8):
        assert device_map[f"model.layers.{layer_index}.self_attn.q_proj"] == (
            torch.device("cpu"),
            torch.device("cpu"),
        )
    for layer_index in range(8, 28):
        assert device_map[f"model.layers.{layer_index}.self_attn.q_proj"] == (
            "cpu",
            "disk",
        )

    if dist.get_rank() == 0:
        assert _offload_dir == offload_dir
