# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload.load import load_offloaded_model

# from compressed_tensors.offload.convert import from_accelerate, to_accelerate
from tests.test_offload.conftest import torchrun
from transformers import AutoModelForCausalLM


@pytest.mark.integration
@torchrun(world_size=2)
def test_accelerate_load_disk(disable_convert):
    with load_offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            device_map="disk",
            max_memory={"cpu": 596049920},  # force disk offloading (~half model size)
            offload_folder="temp",
            dtype=torch.bfloat16,
        )

    assert model.num_parameters() == 596049920

    if dist.get_rank() == 0:
        assert set(model.hf_device_map.values()) == {"cpu", "disk"}
    else:
        assert model.device.type == "meta"


@pytest.mark.integration
@torchrun(world_size=2)
def test_thing():
    with load_offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            device_map="disk",
            max_memory={"cpu": 596049920},  # force disk offloading (~half model size)
            offload_folder="temp",
            dtype=torch.bfloat16,
        )

    assert model.num_parameters() == 596049920

    if dist.get_rank() == 0:
        assert set(model.hf_device_map.values()) == {"cpu", "disk"}
    else:
        print(model.hf_device_map)
        assert model.device.type == "meta"
