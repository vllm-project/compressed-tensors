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

from compressed_tensors.modeling import (
    IMPL_ATTR,
    KV_CACHE_ATTR,
    QuantizedAttentionImpl,
    QuantizedKVCache,
    register_query_hook,
)
from compressed_tensors.quantization.lifecycle.apply import apply_quantization_config
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_config import (
    QuantizationConfig,
    QuantizationStatus,
)
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from tests.testing_utils import requires_gpu
from transformers import AutoModelForCausalLM


@requires_gpu
def test_apply_config_detects_deepseekv3_attention_and_hooks():
    model = AutoModelForCausalLM.from_pretrained(
        "trl-internal-testing/tiny-DeepseekV3ForCausalLM", device_map="cuda"
    )
    inputs = {key: value.to("cuda") for key, value in model.dummy_inputs.items()}

    # Build attention quantization scheme targeting attention modules
    qa = QuantizationArgs(
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        num_bits=8,
        dynamic=False,
    )
    scheme = QuantizationScheme(
        targets=["re:.*self_attn$"],
        input_activations=qa,
    )
    config = QuantizationConfig(
        config_groups={"group_0": scheme},
        quantization_status=QuantizationStatus.INITIALIZED,
        kv_cache_scheme=None,
    )
    apply_quantization_config(model, config)

    # Validate q/k/v qparams initialized and hooks attached
    q_called = []
    k_called = []
    v_called = []

    for idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        assert isinstance(getattr(attn, IMPL_ATTR), QuantizedAttentionImpl)
        assert isinstance(getattr(attn, KV_CACHE_ATTR), QuantizedKVCache)

        assert hasattr(attn, "q_scale")
        assert hasattr(attn, "k_scale")
        assert hasattr(attn, "v_scale")
        assert hasattr(attn, "q_zero_point")
        assert hasattr(attn, "k_zero_point")
        assert hasattr(attn, "v_zero_point")

        q_called.append(False)
        k_called.append(False)
        v_called.append(False)

        def q_hook(_module, _states, i=idx):
            q_called[i] = True

        register_query_hook(attn, q_hook)
        impl = getattr(attn, IMPL_ATTR)

        def _k_pre_hook(_impl, args, kwargs, i=idx):
            k_called[i] = True
            return args, kwargs

        def _v_pre_hook(_impl, args, kwargs, i=idx):
            v_called[i] = True
            return args, kwargs

        impl.register_forward_pre_hook(_k_pre_hook, with_kwargs=True)
        impl.register_forward_pre_hook(_v_pre_hook, with_kwargs=True)

    model(**inputs, use_cache=True)
    assert all(q_called) and all(k_called) and all(v_called)
