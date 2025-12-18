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
from compressed_tensors.offload.dispatch import dispatch_model, offload_model
from tests.testing_utils import requires_gpu
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.mark.integration
@requires_gpu
@pytest.mark.parametrize("model_id", ["meta-llama/Llama-3.2-1B-Instruct"])
@torch.inference_mode()
def test_offload_and_dispatch_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda").eval()

    sample = tokenizer("Hello my name is", return_tensors="pt")
    sample = {k: v.to("cuda:0") for k, v in sample.items()}
    true_logits = model(**sample).logits

    model = offload_model(model, "cuda:0", "cpu")
    offloaded_logits = model(**sample).logits
    assert torch.allclose(offloaded_logits, true_logits)

    model = dispatch_model(model)
    dispatched_logits = model(**sample).logits
    assert torch.allclose(dispatched_logits, true_logits)
