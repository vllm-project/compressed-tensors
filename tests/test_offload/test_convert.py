# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
