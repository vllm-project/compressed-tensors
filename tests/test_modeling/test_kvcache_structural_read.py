# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.modeling.kvcache import QuantizedKVCache
from transformers import DynamicCache, PretrainedConfig


def _make_qkv_cache():
    # QuantizedKVCache only needs a config and a (weakly referenced) attn module.
    attn_module = torch.nn.Module()
    return QuantizedKVCache(PretrainedConfig(), attn_module)


def test_structural_read_delegates_to_wrapped_cache():
    # transformers>=5 models (e.g. DiffusionGemma's decoder) read the cache
    # structurally via `past_key_values.layers[i].keys` instead of `update()`.
    # The wrapper must delegate such reads to the real cache instead of raising.
    real = DynamicCache()
    key = torch.ones(1, 1, 2, 4)
    value = torch.zeros(1, 1, 2, 4)
    real.update(key, value, 0)

    qkv = _make_qkv_cache()
    qkv.add_past_key_values(real)

    # structural attribute access resolves through to the wrapped cache
    assert qkv.layers is real.layers
    assert torch.equal(qkv.layers[0].keys, key)
    assert torch.equal(qkv.layers[0].values, value)


def test_genuinely_missing_attribute_still_raises():
    real = DynamicCache()
    real.update(torch.ones(1, 1, 1, 4), torch.ones(1, 1, 1, 4), 0)

    qkv = _make_qkv_cache()
    qkv.add_past_key_values(real)

    with pytest.raises(AttributeError):
        qkv.this_attr_does_not_exist_anywhere


def test_missing_attribute_raises_when_no_cache_attached():
    qkv = _make_qkv_cache()  # past_key_values is None
    with pytest.raises(AttributeError):
        qkv.layers
