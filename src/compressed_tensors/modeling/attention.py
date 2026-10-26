# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from collections.abc import MutableMapping
from contextlib import contextmanager
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
    "override_hooked_attention_backend",
    "register_query_hook",
    "IMPL_ATTR",
]


IMPL_ATTR = "impl"
HOOKED_ATTENTION_NAME = "ct_hooked_attention"
_QUANTIZABLE_ATTR = "_ct_quantizable_attention"
_ORIGINAL_ATTENTION_ATTR = "_ct_original_attention"
_IMPL_REQUIRED_ATTR = "_ct_attention_impl_required"
_STRICT_ATTENTION_ATTR = "_ct_attention_quantization_required"
# A hooked name always retains the same mask contract. It contains no model state,
# so a second model cannot overwrite the mask used by an already hooked model.
_HOOKED_BACKENDS: dict[str, tuple[str, Optional[Callable]]] = {}
_DISPATCH_LOCK = Lock()
_DEFAULT_SDPA_MASK = ALL_MASK_ATTENTION_FUNCTIONS.get("sdpa")


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
        quantizable = getattr(module, _QUANTIZABLE_ATTR, True)
        if getattr(module, _IMPL_REQUIRED_ATTR, quantizable):
            raise ValueError(
                f"Using {HOOKED_ATTENTION_NAME} attention implementation, but "
                f"{type(module).__name__} has no {IMPL_ATTR} submodule. Attention "
                "quantization and attention transforms have to target every attention "
                "module that can be quantized (attention modules with key/value "
                "projections and a cache argument)"
            )
        return _call_original_attention(module, *args, **kwargs)

    return getattr(module, IMPL_ATTR)(module, *args, **kwargs)


def initialize_hooked_attention(
    model: PreTrainedModel, module: Module, *, allow_unselected: bool = False
):
    """
    Attach query/cache hooks while preserving each component's attention backend.

    Attention and mask functions must use the same backend. A multimodal model can
    have different backends on its text and vision configs, so each owning config
    receives a hooked name paired with its original mask function.

    :param model: parent model of attention module
    :param module: attention module to initialize with
    :param allow_unselected: permit unselected modules to use original attention
        for transform-only dispatch. The caller must validate transform dependencies.
        Direct setup and attention quantization retain the strict all-cache-owner
        requirement, including after a prior transform-only initialization.
    """
    from compressed_tensors.quantization.lifecycle.initialize import (  # circ dep
        is_cached_attention_module,
    )

    config = getattr(module, "config", model.config)
    initialized = callable(getattr(module, _ORIGINAL_ATTENTION_ATTR, None))
    needs_dispatch = (
        not initialized or config._attn_implementation not in _HOOKED_BACKENDS
    )
    if needs_dispatch:
        # Resolve every callback before changing modules, configs or registries.
        # In particular, do not leave a partially hooked model on resolution failure.
        candidates = list(_attention_modules_with_configs(model))
        if not any(candidate is module for candidate, _ in candidates):
            candidates.append((module, config))
        plans = []
        for candidate, owner_config in candidates:
            attention = _get_original_attention(candidate, owner_config)
            if candidate is module and attention is None:
                raise ValueError(
                    f"Cannot resolve original attention for {type(candidate).__name__}"
                )
            backend, mask = _original_backend(owner_config)
            if attention is not None:
                for namespace in _forward_namespaces(candidate):
                    registry = namespace.get("ALL_ATTENTION_FUNCTIONS")
                    if registry is not None and not isinstance(
                        registry, MutableMapping
                    ):
                        raise ValueError(
                            "Cannot register hooked attention for "
                            f"{type(candidate).__name__}"
                        )
            plans.append(
                (
                    candidate,
                    owner_config,
                    attention,
                    backend,
                    mask,
                    is_cached_attention_module(candidate),
                )
            )

        aliases = {}
        for candidate, owner_config, attention, backend, mask, quantizable in plans:
            if attention is not None:
                alias = _register_attention_dispatch(backend, mask)
                aliases[id(owner_config)] = (owner_config, alias)
                # Model-local registries may override the global attention mapping.
                for namespace in _forward_namespaces(candidate):
                    registry = namespace.get("ALL_ATTENTION_FUNCTIONS")
                    if registry is not None:
                        registry[alias] = _hooked_attention
            setattr(candidate, _ORIGINAL_ATTENTION_ATTR, attention)
            setattr(candidate, _QUANTIZABLE_ATTR, quantizable)

        for owner_config, alias in aliases.values():
            # The public config setter propagates to subconfigs, overwriting their
            # distinct backends. Transformers also uses this internal assignment
            # when changing one owning config without changing its descendants.
            owner_config._attn_implementation_internal = alias

    strict = not allow_unselected or getattr(model, _STRICT_ATTENTION_ATTR, False)
    if needs_dispatch or strict != getattr(model, _STRICT_ATTENTION_ATTR, None):
        for candidate, _ in _attention_modules_with_configs(model):
            required = getattr(candidate, _IMPL_REQUIRED_ATTR, False)
            quantizable = getattr(candidate, _QUANTIZABLE_ATTR, True)
            setattr(
                candidate, _IMPL_REQUIRED_ATTR, required or (strict and quantizable)
            )
    # A selected query transform needs its impl even when the module only reads a
    # shared cache and cannot itself own quantized key/value projections.
    setattr(module, _IMPL_REQUIRED_ATTR, True)
    setattr(model, _STRICT_ATTENTION_ATTR, strict)
    if not hasattr(module, IMPL_ATTR):
        module.register_module(IMPL_ATTR, QuantizedAttentionImpl(model.config))
    initialize_hooked_kv_cache(model, module)


def _attention_modules_with_configs(model: PreTrainedModel):
    """Find attention and its nearest owning config, including subclass overrides."""
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


def _original_backend(config: PretrainedConfig) -> tuple[str, Optional[Callable]]:
    backend = config._attn_implementation
    if backend in _HOOKED_BACKENDS:
        return _HOOKED_BACKENDS[backend]
    if isinstance(backend, str) and backend.startswith(HOOKED_ATTENTION_NAME):
        raise ValueError(f"Cannot resolve original attention backend for {backend!r}")
    backend = backend or "eager"
    # Custom backends without a mask entry intentionally receive no generated mask
    # in Transformers. Keep that behavior instead of assigning an arbitrary mask.
    return backend, ALL_MASK_ATTENTION_FUNCTIONS.get(backend)


def _register_attention_dispatch(backend: str, mask: Optional[Callable]) -> str:
    with _DISPATCH_LOCK:
        for name, (original_backend, original_mask) in _HOOKED_BACKENDS.items():
            if backend == original_backend and mask is original_mask:
                return name
        if (
            backend == "sdpa"
            and mask is _DEFAULT_SDPA_MASK
            and HOOKED_ATTENTION_NAME not in _HOOKED_BACKENDS
        ):
            name = HOOKED_ATTENTION_NAME
        else:
            name = f"{HOOKED_ATTENTION_NAME}_{len(_HOOKED_BACKENDS)}"
        ALL_ATTENTION_FUNCTIONS.register(name, _hooked_attention)
        if mask is not None:
            ALL_MASK_ATTENTION_FUNCTIONS.register(name, mask)
        _HOOKED_BACKENDS[name] = (backend, mask)
        return name


def _call_original_attention(module: Module, *args, **kwargs):
    attention = getattr(module, _ORIGINAL_ATTENTION_ATTR, None)
    if attention is None:
        raise ValueError(
            f"Cannot resolve original attention for {type(module).__name__}"
        )
    return attention(module, *args, **kwargs)


def _forward_namespaces(module: Module):
    # A delegating override has the user's globals, not the parent model's eager
    # implementation or local registry. Prefer the effective override, then look
    # through its MRO without replacing model-specific math with a generic fallback.
    with unwrap_offload_forward(module):
        forwards = [module.forward]
        forwards.extend(
            cls.__dict__["forward"]
            for cls in type(module).__mro__
            if "forward" in cls.__dict__
        )
        seen = set()
        namespaces = []
        for forward in forwards:
            forward = inspect.unwrap(forward)
            forward = getattr(forward, "__func__", forward)
            code = getattr(forward, "__code__", None)
            if code is None or "ALL_ATTENTION_FUNCTIONS" not in code.co_names:
                continue
            namespace = getattr(forward, "__globals__", {})
            if id(namespace) not in seen:
                seen.add(id(namespace))
                namespaces.append(namespace)
    return namespaces


def _get_original_attention(
    module: Module, config: PretrainedConfig, *, backend: Optional[str] = None
) -> Optional[Callable]:
    """Resolve model-local eager and registry overrides before replacing dispatch."""
    if backend is None:
        backend, _ = _original_backend(config)
    for namespace in _forward_namespaces(module):
        registry = namespace.get("ALL_ATTENTION_FUNCTIONS", ALL_ATTENTION_FUNCTIONS)
        attention = registry.get(backend)
        if attention is None and backend == "eager":
            attention = namespace.get("eager_attention_forward")
        if not callable(attention) or attention is _hooked_attention:
            raise ValueError(
                f"Cannot resolve {backend!r} attention for {type(module).__name__}"
            )
        return attention
    # Inline attention never calls the hooked interface. It needs no callback
    # unless it is selected for query work, which initialization rejects.
    return None


@contextmanager
def override_hooked_attention_backend(model: PreTrainedModel, backend: str):
    """Temporarily select a backend without bypassing installed query hooks.

    Hooks installed inside the context remain active on exit. Existing callbacks
    and component backend choices are restored, including on nested/exception exits.
    This context does not install quantization or transforms on an unhooked model.
    """
    if not isinstance(backend, str) or backend.startswith(HOOKED_ATTENTION_NAME):
        raise ValueError(f"Expected an original attention backend, got {backend!r}")
    candidates = list(_attention_modules_with_configs(model))
    configs = {id(config): config for _, config in candidates}
    pending = [model.config]
    seen = set()
    while pending:
        config = pending.pop()
        if id(config) in seen:
            continue
        seen.add(id(config))
        configs[id(config)] = config
        pending.extend(
            subconfig
            for name in getattr(config, "sub_configs", {})
            if (subconfig := getattr(config, name, None)) is not None
        )
    original = {}
    for key, config in configs.items():
        original_backend, mask = _original_backend(config)
        original[key] = (config, config._attn_implementation, original_backend, mask)
    callbacks = {
        id(module): getattr(module, _ORIGINAL_ATTENTION_ATTR, None)
        or _get_original_attention(module, config)
        for module, config in candidates
    }
    temporary = {
        key: (config, backend, backend, ALL_MASK_ATTENTION_FUNCTIONS.get(backend))
        for key, config in configs.items()
    }
    _set_attention_backends(model, temporary)
    try:
        yield
    finally:
        _set_attention_backends(model, original, callbacks)


def _set_attention_backends(model, choices, callbacks=None):
    candidates = list(_attention_modules_with_configs(model))
    plans = []
    active_configs = set()
    for module, config in candidates:
        choice = choices.get(id(config))
        if choice is None:
            # A new component with a new config has no pre-context backend to
            # restore. Preserve its own valid backend rather than guessing one.
            continue
        _, _, backend, mask = choice
        callback = (callbacks or {}).get(id(module))
        if callback is None:
            callback = _get_original_attention(module, config, backend=backend)
        active = callable(getattr(module, _ORIGINAL_ATTENTION_ATTR, None)) or hasattr(
            module, IMPL_ATTR
        )
        if active and callback is None:
            raise ValueError(
                f"Cannot resolve original attention for {type(module).__name__}"
            )
        if active:
            active_configs.add(id(config))
        registries = [
            namespace["ALL_ATTENTION_FUNCTIONS"]
            for namespace in _forward_namespaces(module)
            if "ALL_ATTENTION_FUNCTIONS" in namespace
        ]
        if active and any(
            not isinstance(registry, MutableMapping) for registry in registries
        ):
            raise ValueError(
                f"Cannot register hooked attention for {type(module).__name__}"
            )
        plans.append((module, config, callback, registries, active))

    aliases = {
        key: _register_attention_dispatch(backend, mask)
        for key, (_, _, backend, mask) in choices.items()
        if key in active_configs
    }
    for module, config, callback, registries, active in plans:
        if active:
            for registry in registries:
                registry[aliases[id(config)]] = _hooked_attention
            setattr(module, _ORIGINAL_ATTENTION_ATTR, callback)
    for key, (config, plain_backend, _, _) in choices.items():
        config._attn_implementation_internal = aliases.get(key, plain_backend)


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
