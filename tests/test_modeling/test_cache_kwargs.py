# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.modeling import get_kv_cache_kwarg_name


class RenamedCacheAttention(torch.nn.Module):
    def forward(self, hidden_states, cache=None):
        return hidden_states


class PluralCacheAttention(torch.nn.Module):
    def forward(self, hidden_states, past_key_values=None):
        return hidden_states


class SingularCacheAttention(torch.nn.Module):
    def forward(self, hidden_states, past_key_value=None):
        return hidden_states


class NoCacheAttention(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class AllCacheNamesAttention(torch.nn.Module):
    def forward(
        self, hidden_states, cache=None, past_key_values=None, past_key_value=None
    ):
        return hidden_states


class LegacyCacheNamesAttention(torch.nn.Module):
    def forward(self, hidden_states, past_key_values=None, past_key_value=None):
        return hidden_states


@pytest.mark.parametrize(
    ("attention_cls", "expected"),
    [
        (RenamedCacheAttention, "cache"),
        (PluralCacheAttention, "past_key_values"),
        (SingularCacheAttention, "past_key_value"),
        (NoCacheAttention, None),
    ],
)
def test_get_kv_cache_kwarg_name(attention_cls, expected):
    assert get_kv_cache_kwarg_name(attention_cls()) == expected


def test_get_kv_cache_kwarg_name_prefers_newest():
    # when several names are present, the newest should be preferred
    assert get_kv_cache_kwarg_name(AllCacheNamesAttention()) == "cache"
    assert get_kv_cache_kwarg_name(LegacyCacheNamesAttention()) == "past_key_values"
