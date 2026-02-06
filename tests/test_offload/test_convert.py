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

import pytest
import torch
from compressed_tensors.offload.convert import from_accelerate, to_accelerate
from compressed_tensors.offload.load import load_offloaded_model
from transformers import AutoModelForCausalLM


@pytest.mark.integration
def test_convert():
    with load_offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            device_map="disk",
            max_memory={"cpu": 596049920},  # force disk offloading
            offload_folder="temp",
            dtype=torch.bfloat16,
        )

    assert not hasattr(model, "hf_device_map")
    to_accelerate(model)
    model.save_pretrained("temp_save")


@pytest.mark.integration
def test_idempotency(disable_convert):
    with load_offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            device_map="disk",
            max_memory={"cpu": 596049920},  # force disk offloading
            offload_folder="temp",
            dtype=torch.bfloat16,
        )

    assert hasattr(model, "hf_device_map")

    from_accelerate(model)
    from_accelerate(model)

    to_accelerate(model)
    to_accelerate(model)
