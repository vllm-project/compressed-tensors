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
from compressed_tensors.offload import convert, disable_offloading, disable_onloading
from compressed_tensors.offload.load import load_offloaded_model

# from compressed_tensors.offload.convert import from_accelerate, to_accelerate
from tests.test_offload.conftest import torchrun
from transformers import AutoModelForCausalLM


@pytest.fixture()
def disable_convert(monkeypatch):
    # TODO: for some reason this doesn't work
    import compressed_tensors.offload.convert as convert

    monkeypatch.setattr(convert, "from_accelerate", lambda *args, **kwargs: None)
    monkeypatch.setattr(convert, "to_accelerate", lambda *args, **kwargs: None)


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
    else:
        assert model.device.type == "meta"
