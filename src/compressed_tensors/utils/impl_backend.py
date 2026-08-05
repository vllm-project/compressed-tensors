# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import os
from typing import Callable

from loguru import logger


__all__ = ["ImplBackend"]


ENFORCE_EAGER = os.environ.get("CT_ENFORCE_EAGER", False)
if ENFORCE_EAGER:
    logger.warning(
        "CT_ENFORCE_EAGER is set to True, meaning that compressed-tensors will "
        "use eager pytorch implementations where possible. Remove this flag for "
        "improved performance."
    )


class ImplBackend:
    """
    Priority-based backend dispatch registry.

    Functions can register multiple implementations (backends) under a shared
    name, each with an availability requirement and a priority. Decorating a
    function with ``entrypoint(name)`` turns it into a dispatch wrapper whose body
    serves as the eager/torch fallback.

    Usage::

        @ImplBackend.register("my_op", req=lambda x: x.is_cuda, priority=0)
        def my_op_cuda(x): ...

        @ImplBackend.entrypoint("my_op")
        def my_op(x):
            # torch/eager fallback — runs when no registered backend matches
            ...

    To call a specific backend directly (e.g. in tests)::

        ImplBackend.call("my_op_cuda", x)
    """

    _backends: dict[str, list[tuple[Callable, Callable, int]]] = {}
    _fn_registry: dict[str, Callable] = {}  # fn.__name__ -> backend_fn

    @classmethod
    def register(cls, name: str, req: Callable[..., bool], priority: int | str):
        """
        Decorator that registers a backend implementation.

        The decorated function's `__name__` is recorded in a global registry
        and must be unique across all registered backends.

        :param name: operation name shared across all backends for this function
        :param req: callable returning True when this backend is usable; receives
            the same positional and keyword arguments as the dispatch wrapper, so
            requirements can inspect actual inputs (e.g. `lambda x: x.is_cuda`)
        :param priority: lower values are tried first (higher priority).
            Set to ``"disable"`` to register the function in the registry
            (so it can still be called directly via :meth:`call`) without
            adding it as a dispatch candidate.
        """

        def decorator(backend_fn: Callable) -> Callable:
            cls._add_to_registery(backend_fn)

            if priority == "disable":
                return backend_fn

            if name not in cls._backends:
                cls._backends[name] = []
            cls._backends[name].append((backend_fn, req, priority))
            cls._backends[name].sort(key=lambda entry: entry[2])
            return backend_fn

        return decorator

    @classmethod
    def call(cls, fn_name: str, *args, **kwargs):
        """
        Call a specific registered backend directly by its function name,
        bypassing availability checks. Useful for testing individual backends
        in isolation.

        :param fn_name: `__name__` of the backend function to call
        """
        if fn_name not in cls._fn_registry:
            available = list(cls._fn_registry.keys())
            raise KeyError(
                f"No registered backend named '{fn_name}'. " f"Available: {available}"
            )
        return cls._fn_registry[fn_name](*args, **kwargs)

    @classmethod
    def entrypoint(cls, name: str) -> Callable:
        """
        Decorator that turns a function into a dispatch wrapper.

        The decorated function serves as the eager/torch fallback: it is called
        only when no registered backend for `name` satisfies its requirement.
        Registered backends are tried in priority order (lowest value first) and
        each receives the same arguments as the wrapper.

        :param name: operation name whose registered backends will be tried first
        """

        def decorator(fallback_fn: Callable) -> Callable:
            cls._add_to_registery(fallback_fn)

            @functools.wraps(fallback_fn)
            def wrapper(*args, **kwargs):
                if not ENFORCE_EAGER:
                    for backend_fn, req, _ in cls._backends.get(name, []):
                        if req(*args, **kwargs):
                            return backend_fn(*args, **kwargs)
                return fallback_fn(*args, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def _add_to_registery(cls, fn: Callable):
        fn_name = fn.__name__
        if fn_name in cls._fn_registry:
            raise ValueError(
                f"A backend with function name '{fn_name}' is already "
                "registered. Backend function names must be unique across "
                "all ops."
            )
        cls._fn_registry[fn_name] = fn
