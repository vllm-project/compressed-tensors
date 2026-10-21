# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from typing import Callable, Optional

import torch
from compressed_tensors.modeling.kvcache import initialize_hooked_kv_cache
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.utils import getattr_chain
from compressed_tensors.utils.internal import InternalModule
from loguru import logger
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from transformers import PretrainedConfig, PreTrainedModel
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


__all__ = [
    "QuantizedAttentionImpl",
    "initialize_hooked_attention",
    "register_query_hook",
    "IMPL_ATTR",
]


IMPL_ATTR = "impl"
HOOKED_ATTENTION_NAME = "ct_hooked_attention"


class QuantizedAttentionImpl(InternalModule):
    """
    QuantizedAttentionImpl module which wraps the functionality of the original
    attention implementation. Unlike the original attention function, this
    implementation is a `torch.nn.Module` which can be hooked to trigger
    transforms and calibration hooks.

    This module works by being registered as a submodule to attention modules via
    `initialize_hooked_attention`, registering a new attention implementation function
    which calls this module, then setting the model attention implementation to the new
    function. After triggering hooks and quantization, this module calls the original
    attention implementation function.
    """

    _original_impl = "eager"

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        module: Module,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *args,
        **kwargs,
    ):
        # quantization
        quant_args_attr = "quantization_scheme.input_activations"
        quant_args = getattr_chain(module, quant_args_attr, None)
        quant_enabled = getattr(module, "quantization_enabled", True)
        if quant_args is not None and quant_enabled:
            query = forward_quantize(module, query, "q", quant_args)

        # original attention
        return _get_attention_function(module)(
            module,
            query,
            key,
            value,
            *args,
            **kwargs,
        )


def _get_attention_function(module: Module) -> Callable:
    impl = QuantizedAttentionImpl._original_impl
    if impl == "eager":
        # transformers models call the `eager_attention_forward` defined in their own
        # modeling file rather than a registered "eager" implementation. The class
        # forward is used because offloading hooks wrap the instance forward
        attn_forward = type(module).forward.__globals__.get(
            "eager_attention_forward", _default_eager_attention_forward
        )
        if attn_forward is _default_eager_attention_forward:
            logger.bind(log_once=True).warning(
                "Cannot find `eager_attention_forward` for "
                f"{type(module).__name__}, using the default eager attention"
            )
        return attn_forward

    return ALL_ATTENTION_FUNCTIONS[impl]


def _repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _default_eager_attention_forward(
    module: Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # copy of `eager_attention_forward` from transformers' Llama modeling file
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ----- initialize ----- #


def _hooked_attention(module: Module, *args, **kwargs):
    assert hasattr(module, IMPL_ATTR), (
        f"Using {HOOKED_ATTENTION_NAME} attention implementation, "
        f"but attention module does not have {IMPL_ATTR} submodule."
    )

    return getattr(module, IMPL_ATTR)(module, *args, **kwargs)


def initialize_hooked_attention(model: PreTrainedModel, module: Module):
    """
    Initialize `QuantizedAttentionImpl` and `QuantizedKVCache` instances
    attached to attention. Assumes that only one model is hooked at a time.

    :param model: parent model of attention module
    :param module: attention module to initialize with
    """
    if not hasattr(module, IMPL_ATTR):
        module.register_module(IMPL_ATTR, QuantizedAttentionImpl(model.config))

    if model.config._attn_implementation != HOOKED_ATTENTION_NAME:
        QuantizedAttentionImpl._original_impl = model.config._attn_implementation
        original_mask = ALL_MASK_ATTENTION_FUNCTIONS[model.config._attn_implementation]

        ALL_ATTENTION_FUNCTIONS.register(HOOKED_ATTENTION_NAME, _hooked_attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register(HOOKED_ATTENTION_NAME, original_mask)
        model.set_attn_implementation(HOOKED_ATTENTION_NAME)
        assert model.config._attn_implementation == HOOKED_ATTENTION_NAME

    initialize_hooked_kv_cache(model, module)


# ----- hooks ----- #


def register_query_hook(
    module: Module, hook: Callable[[Module, Tensor], Optional[Tensor]]
) -> RemovableHandle:
    """
    Register a hook which takes post-rope query states as an argument and
    returns the modified query states or `None`

    :param module: attention module to add hook to
    :param hook: query hook function
    """
    impl: QuantizedAttentionImpl = getattr(module, IMPL_ATTR)

    def _hook(impl: QuantizedAttentionImpl, args, kwargs):
        bound = inspect.signature(impl.forward).bind(*args, **kwargs)
        value = hook(module, bound.arguments["query"])
        if value is not None:
            bound.arguments["query"] = value

        return bound.args, bound.kwargs

    return impl.register_forward_pre_hook(_hook, with_kwargs=True)
