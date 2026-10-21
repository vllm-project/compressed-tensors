# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from typing import Callable, Optional

from compressed_tensors.modeling.kvcache import initialize_hooked_kv_cache
from compressed_tensors.offload import unwrap_offload_forward
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.utils import getattr_chain
from compressed_tensors.utils.internal import InternalModule
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
_QUANTIZABLE_ATTR = "_ct_quantizable_attention"
_ORIGINAL_ATTENTION_ATTR = "_ct_original_attention"


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

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self._quantize_query = False

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
        if quant_args is not None and quant_enabled and self._quantize_query:
            query = forward_quantize(module, query, "q", quant_args)

        # original attention
        return _call_original_attention(
            module,
            query,
            key,
            value,
            *args,
            **kwargs,
        )


# ----- initialize ----- #


def _hooked_attention(module: Module, *args, **kwargs):
    if not hasattr(module, IMPL_ATTR):
        # attention modules which can be quantized are marked by
        # `initialize_hooked_attention` and need an impl. The others, for example
        # layers that reuse the kv cache of an earlier layer (gemma-4) or CLIP-style
        # vision encoders, use the model's original attention implementation
        if getattr(module, _QUANTIZABLE_ATTR, True):
            raise ValueError(
                f"Using {HOOKED_ATTENTION_NAME} attention implementation, but "
                f"{type(module).__name__} has no {IMPL_ATTR} submodule. Attention "
                "quantization and attention transforms have to target every attention "
                "module that can be quantized (attention modules with key/value "
                "projections and a cache argument)"
            )
        return _call_original_attention(module, *args, **kwargs)

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
        original_mask = ALL_MASK_ATTENTION_FUNCTIONS[model.config._attn_implementation]
        # decide once which attention modules can be quantized, since
        # `is_cached_attention_module` can't be traced by torch.compile
        from compressed_tensors.quantization.lifecycle.initialize import (  # circ dep
            _is_attention_module,
            is_cached_attention_module,
        )

        for submodule in model.modules():
            if _is_attention_module(submodule) or (
                "attention" in type(submodule).__name__.lower()
                and hasattr(submodule, "q_proj")
            ):
                setattr(
                    submodule,
                    _ORIGINAL_ATTENTION_ATTR,
                    _get_original_attention(submodule, model.config),
                )
                setattr(
                    submodule,
                    _QUANTIZABLE_ATTR,
                    is_cached_attention_module(submodule),
                )

        ALL_ATTENTION_FUNCTIONS.register(HOOKED_ATTENTION_NAME, _hooked_attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register(HOOKED_ATTENTION_NAME, original_mask)
        model.set_attn_implementation(HOOKED_ATTENTION_NAME)
        assert model.config._attn_implementation == HOOKED_ATTENTION_NAME

    initialize_hooked_kv_cache(model, module)


def _call_original_attention(module: Module, *args, **kwargs):
    attention = getattr(module, _ORIGINAL_ATTENTION_ATTR, None)
    if attention is None:
        raise ValueError(
            f"Cannot resolve original attention for {type(module).__name__}"
        )
    return attention(module, *args, **kwargs)


def _get_original_attention(
    module: Module, config: PretrainedConfig
) -> Optional[Callable]:
    """Resolve attention before replacing its backend, including model-local eager."""
    config = getattr(module, "config", config)
    backend = config._attn_implementation
    with unwrap_offload_forward(module):
        forward = inspect.unwrap(module.forward)
        forward = getattr(forward, "__func__", forward)
        namespace = getattr(forward, "__globals__", {})
    registry = namespace.get("ALL_ATTENTION_FUNCTIONS", ALL_ATTENTION_FUNCTIONS)
    attention = registry.get(backend)
    if attention is None and backend in (None, "eager"):
        # Eager is a model-specific default, not a global registry entry. For
        # example, CLIP and Gemma use different implementations and arguments.
        attention = namespace.get("eager_attention_forward")
    if attention is None and backend in (None, "eager"):
        # Some modules compute attention inline and never dispatch through the
        # registry. They don't need a callback unless the hooked function runs.
        return None
    if not callable(attention):
        raise ValueError(
            f"Cannot resolve {backend!r} attention for {type(module).__name__}"
        )
    return attention


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
