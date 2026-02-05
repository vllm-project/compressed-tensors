# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import chain

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload import disable_offloading, disable_onloading
from compressed_tensors.offload.convert import offloaded_model
from compressed_tensors.offload.dispatch import (
    convert_accelerate,
    convert_to_accelerate,
)
from tests.test_offload.conftest import torchrun
from transformers import AutoModelForCausalLM


@pytest.mark.integration
@torchrun(world_size=2)
def test_hybrid_disk_offloaded_model():
    with offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            device_map="disk",
            max_memory={"cpu": 596049920},  # force disk offloading
            offload_folder="temp",
            dtype=torch.bfloat16,
        )

    assert model.num_parameters() == 596049920

    expected_device_map = {
        "model.embed_tokens": "cpu",
        "lm_head": "cpu",
        "model.layers.0": "cpu",
        "model.layers.1": "cpu",
        "model.layers.2": "cpu",
        "model.layers.3": "cpu",
        "model.layers.4": "cpu",
        "model.layers.5": "cpu",
        "model.layers.6": "cpu",
        "model.layers.7": "cpu",
        "model.layers.8": "disk",
        "model.layers.9": "disk",
        "model.layers.10": "disk",
        "model.layers.11": "disk",
        "model.layers.12": "disk",
        "model.layers.13": "disk",
        "model.layers.14": "disk",
        "model.layers.15": "disk",
        "model.layers.16": "disk",
        "model.layers.17": "disk",
        "model.layers.18": "disk",
        "model.layers.19": "disk",
        "model.layers.20": "disk",
        "model.layers.21": "disk",
        "model.layers.22": "disk",
        "model.layers.23": "disk",
        "model.layers.24": "disk",
        "model.layers.25": "disk",
        "model.layers.26": "disk",
        "model.layers.27": "disk",
        "model.norm": "disk",
        "model.rotary_emb": "disk",
    }

    if dist.get_rank() == 0:
        assert model.hf_device_map == expected_device_map
        convert_accelerate(model)
    else:
        assert model.device.type == "meta"


@pytest.mark.integration
def test_convert():
    with offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            device_map="disk",
            max_memory={"cpu": 596049920},  # force disk offloading
            offload_folder="temp",
            dtype=torch.bfloat16,
        )

    convert_accelerate(model)
    convert_to_accelerate(model)
    model.save_pretrained("woah")
