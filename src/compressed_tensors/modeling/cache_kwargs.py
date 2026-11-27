# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect

from torch.nn import Module


__all__ = [
    "KV_CACHE_KWARG_NAMES",
    "get_kv_cache_kwarg_name",
]


# Single source of truth for the names of the attention forward kwargs which
# carry the KV cache, ordered newest first:
# - `cache`: transformers>=5.24 renamed the attention cache argument from
#   `past_key_values` to `cache` (older names remain as deprecated aliases)
# - `past_key_values`: plural rename, introduced in transformers#39956
# - `past_key_value`: original singular name
KV_CACHE_KWARG_NAMES = ("cache", "past_key_values", "past_key_value")


def get_kv_cache_kwarg_name(module: Module) -> str | None:
    """
    Get the name of the KV cache kwarg accepted by a module's forward
    function, preferring the newest name present in the signature (see
    `KV_CACHE_KWARG_NAMES`).

    :param module: module whose forward function to inspect
    :return: name of the KV cache kwarg, or None if the forward function
        accepts none of the names used by transformers
    :raises TypeError: if the forward signature is not introspectable
    :raises ValueError: if the forward signature is invalid
    """
    parameters = inspect.signature(module.forward).parameters
    return next((name for name in KV_CACHE_KWARG_NAMES if name in parameters), None)
