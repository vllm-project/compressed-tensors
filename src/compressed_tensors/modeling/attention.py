# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from collections.abc import MutableMapping
from threading import Lock
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
_ORIGINAL_ATTENTION_ATTR = "_ct_original_attention"
_IMPL_REQUIRED_ATTR = "_ct_attention_impl_required"
# Aliases contain no model state and always retain the same mask implementation.
_HOOKED_BACKENDS: dict[str, tuple[str, Optional[Callable]]] = {}
_DISPATCH_LOCK = Lock()


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
        if getattr(module, _IMPL_REQUIRED_ATTR, True):
            raise ValueError(
                f"Using {HOOKED_ATTENTION_NAME}, but {type(module).__name__} has no "
                f"{IMPL_ATTR} submodule. Attention quantization must target every "
                "attention module with key/value projections and a cache argument"
            )
        return _call_original_attention(module, *args, **kwargs)
    return getattr(module, IMPL_ATTR)(module, *args, **kwargs)


def initialize_hooked_attention(model: PreTrainedModel, module: Module):
    """
    Initialize query/cache hooks, preserving each component's native attention.

    :param model: parent model of attention module
    :param module: attention module to initialize with
    """
    from compressed_tensors.quantization.lifecycle.initialize import (  # circ dep
        is_cached_attention_module,
    )

    config = getattr(module, "config", model.config)
    if (
        not callable(getattr(module, _ORIGINAL_ATTENTION_ATTR, None))
        or config._attn_implementation not in _HOOKED_BACKENDS
    ):
        candidates = list(_attention_modules_with_configs(model))
        if not any(candidate is module for candidate, _ in candidates):
            candidates.append((module, config))
        plans = []
        # Resolve all callbacks and registries before changing model state.
        for candidate, owner_config in candidates:
            backend = owner_config._attn_implementation or "eager"
            backend, mask = _HOOKED_BACKENDS.get(
                backend, (backend, ALL_MASK_ATTENTION_FUNCTIONS.get(backend))
            )
            attention, registries = _resolve_attention(candidate, backend)
            if candidate is module and attention is None:
                raise ValueError(
                    f"Cannot resolve original attention for {type(candidate).__name__}"
                )
            required = is_cached_attention_module(candidate) or getattr(
                candidate, _IMPL_REQUIRED_ATTR, False
            )
            plans.append(
                (
                    candidate,
                    owner_config,
                    attention,
                    registries,
                    backend,
                    mask,
                    required,
                )
            )

        for (
            candidate,
            owner_config,
            attention,
            registries,
            backend,
            mask,
            required,
        ) in plans:
            if attention is not None:
                alias = _register_attention_dispatch(backend, mask)
                for registry in registries:
                    registry[alias] = _hooked_attention
                # The public setter propagates to child configs, overwriting mixed
                # text/vision backends. Change only this component's owning config.
                owner_config._attn_implementation_internal = alias
            setattr(candidate, _ORIGINAL_ATTENTION_ATTR, attention)
            setattr(candidate, _IMPL_REQUIRED_ATTR, required)

    setattr(module, _IMPL_REQUIRED_ATTR, True)
    if not hasattr(module, IMPL_ATTR):
        module.register_module(IMPL_ATTR, QuantizedAttentionImpl(model.config))
    initialize_hooked_kv_cache(model, module)


def _attention_modules_with_configs(model: PreTrainedModel):
    pending = [(model, model.config)]
    seen = set()
    while pending:
        module, config = pending.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        config = getattr(module, "config", config)
        if not isinstance(module, InternalModule) and any(
            "attention" in cls.__name__.lower() for cls in type(module).__mro__
        ):
            yield module, config
        pending.extend((child, config) for child in reversed(list(module.children())))


def _resolve_attention(module: Module, backend: str):
    # Delegating overrides have user globals. Find the actual dispatching forward
    # in the MRO, retaining its model-specific eager function and local registry.
    with unwrap_offload_forward(module):
        forwards = [module.forward] + [
            cls.__dict__["forward"]
            for cls in type(module).__mro__
            if "forward" in cls.__dict__
        ]
        attention, registries = None, []
        for forward in forwards:
            forward = inspect.unwrap(forward)
            forward = getattr(forward, "__func__", forward)
            code = getattr(forward, "__code__", None)
            if code is None or "ALL_ATTENTION_FUNCTIONS" not in code.co_names:
                continue
            namespace = forward.__globals__
            registry = namespace.get("ALL_ATTENTION_FUNCTIONS", ALL_ATTENTION_FUNCTIONS)
            if not isinstance(registry, MutableMapping):
                raise ValueError(
                    f"Cannot register hooked attention for {type(module).__name__}"
                )
            if attention is None:
                attention = registry.get(backend)
                if attention is None and backend == "eager":
                    attention = namespace.get("eager_attention_forward")
                if not callable(attention) or attention is _hooked_attention:
                    raise ValueError(
                        f"Cannot resolve {backend!r} attention for "
                        f"{type(module).__name__}"
                    )
            if not any(registry is existing for existing in registries):
                registries.append(registry)
        # Inline attention never uses the registry and remains unchanged unless
        # selected for query hooks, which initialization rejects above.
        return attention, registries


def _register_attention_dispatch(backend: str, mask: Optional[Callable]) -> str:
    with _DISPATCH_LOCK:
        for alias, (original_backend, original_mask) in _HOOKED_BACKENDS.items():
            if backend == original_backend and mask is original_mask:
                return alias
        alias = (
            HOOKED_ATTENTION_NAME
            if backend == "sdpa" and HOOKED_ATTENTION_NAME not in _HOOKED_BACKENDS
            else f"{HOOKED_ATTENTION_NAME}_{len(_HOOKED_BACKENDS)}"
        )
        ALL_ATTENTION_FUNCTIONS.register(alias, _hooked_attention)
        if mask is not None:
            ALL_MASK_ATTENTION_FUNCTIONS.register(alias, mask)
        _HOOKED_BACKENDS[alias] = (backend, mask)
        return alias


def _call_original_attention(module: Module, *args, **kwargs):
    attention = getattr(module, _ORIGINAL_ATTENTION_ATTR, None)
    if attention is None:
        raise ValueError(
            f"Cannot resolve original attention for {type(module).__name__}"
        )
    return attention(module, *args, **kwargs)


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
