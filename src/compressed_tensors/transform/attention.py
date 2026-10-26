# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import numbers

import torch
from compressed_tensors.transform.transform_args import TransformLocation
from compressed_tensors.utils.match import match_named_modules


def _scope(name, module):
    parts = name.split(".")
    first_index = next(
        (i for i, part in enumerate(parts) if part.isdigit()), len(parts)
    )
    return id(getattr(module, "config", None)), tuple(parts[:first_index])


def _producer(reader_name, reader, nodes):
    candidates = [
        (name, module)
        for name, module in nodes
        if _scope(name, module) == _scope(reader_name, reader)
        and not getattr(module, "is_kv_shared_layer", False)
    ]
    source_index = getattr(reader, "kv_shared_layer_index", None)
    if isinstance(source_index, numbers.Integral) and not isinstance(
        source_index, bool
    ):
        candidates = [
            (n, m)
            for n, m in candidates
            if getattr(m, "layer_idx", None) == source_index
        ]
    elif getattr(reader, "layer_type", None) is not None:
        candidates = [
            (n, m)
            for n, m in candidates
            if getattr(m, "layer_type", None) == reader.layer_type
        ]
        stored = [
            (n, m) for n, m in candidates if getattr(m, "store_full_length_kv", False)
        ]
        if stored:
            candidates = stored
        elif candidates:
            candidates = [
                max(candidates, key=lambda item: getattr(item[1], "layer_idx", -1))
            ]
    else:
        candidates = []
    if len(candidates) != 1:
        raise ValueError(
            "Cannot resolve the key-cache producer for shared attention "
            f"{reader_name!r}"
        )
    return candidates[0]


def validate_attention_transforms(model, config):
    """Validate shared-cache query/key bases before any transform mutates the model."""
    # HF checkpoint loading can create modules under a default meta device. The
    # metadata proxy must materialize its matrices to compare the planned bases.
    with torch.device("cpu"):
        _validate_attention_transforms(model, config)


def _validate_attention_transforms(model, config):
    from compressed_tensors.transform.factory.base import (
        TransformBase,
        TransformFactory,
    )

    q_location = TransformLocation.Q_ATTN
    k_location = TransformLocation.K_CACHE
    planned = []
    for name, scheme in config.config_groups.items():
        matches = [
            (module_name, module, arg)
            for arg in scheme.apply
            for module_name, module in match_named_modules(
                model, arg.targets, arg.ignore
            )
        ]
        attention_matches = [
            (n, m, a) for n, m, a in matches if a.location in (q_location, k_location)
        ]
        if not attention_matches:
            continue
        for module_name, module, arg in attention_matches:
            dim = getattr(module, "head_dim", None)
            block = scheme.head_dim
            if (
                not isinstance(block, numbers.Integral)
                or isinstance(block, bool)
                or block <= 0
            ):
                raise ValueError(
                    f"Attention transform {name!r} needs a positive head_dim"
                )
            if isinstance(dim, numbers.Integral) and (dim <= 0 or dim % block):
                raise ValueError(
                    f"Attention transform {name!r} head_dim ({block}) must divide "
                    f"{module_name!r} head_dim ({dim})"
                )
        planned.append((name, scheme, matches))

    if not planned:
        return
    active_scopes = {
        _scope(name, module)
        for _, _, matches in planned
        for name, module, arg in matches
        if arg.location in (q_location, k_location)
    }
    nodes = [
        (name, module)
        for name, module in model.named_modules()
        if _scope(name, module) in active_scopes
        and (
            (hasattr(module, "head_dim") and hasattr(module, "layer_idx"))
            or getattr(module, "is_kv_shared_layer", False)
        )
    ]
    readers = [
        (name, module)
        for name, module in nodes
        if getattr(module, "is_kv_shared_layer", False)
    ]
    if not readers:
        return

    # Each built-in factory owns its generator. Replaying construction with a CPU
    # metadata proxy preserves its argument order without touching model weights or
    # consuming the global RNG, including an offline entry which primes its cache.
    basis = {q_location: {}, k_location: {}}
    matrices = {}

    def matrix(transform, dim):
        # Offloaded parameter reads can return fresh objects; the transform itself
        # has a stable identity throughout this validation pass.
        key = (id(transform), dim)
        if key not in matrices:
            with torch.no_grad():
                block = transform.weight.shape[0]
                value = transform.weight.detach().to(device="cpu", dtype=torch.float64)
                if transform.scheme.type in ("hadamard", "random-hadamard"):
                    if transform.perm is not None:
                        perm = transform.perm.detach().to(device="cpu")
                        value = value[perm][:, perm]
                    if transform.args.inverse:
                        value = value.T
                    value = value / math.sqrt(block)
                error = torch.zeros_like(value)
                if transform.scheme.type == "random-matrix" and transform.args.inverse:
                    # Inverses are computed in float64 and then stored in the
                    # configured precision. Bound only that final rounding step.
                    unit_roundoff = torch.finfo(transform.weight.dtype).eps / 2
                    error = value.abs() * unit_roundoff / (1 - unit_roundoff)
                value = torch.block_diag(*([value] * (dim // block)))
                error = torch.block_diag(*([error] * (dim // block)))
            if not torch.isfinite(value).all() or not torch.isfinite(error).all():
                raise ValueError(
                    "Attention query/key transform contains nonfinite values"
                )
            matrices[key] = (value, error)
        return matrices[key]

    def compose(left, right):
        value_l, error_l = left
        value_r, error_r = right
        value = value_l @ value_r
        error = value_l.abs() @ error_r + error_l @ value_r.abs() + error_l @ error_r
        if not torch.isfinite(value).all() or not torch.isfinite(error).all():
            raise ValueError("Attention query/key transform composition is nonfinite")
        return value, error

    def append(module_name, module, transform):
        dim = getattr(module, "head_dim", None)
        if not isinstance(dim, numbers.Integral) or isinstance(dim, bool) or dim <= 0:
            raise ValueError(f"Cannot resolve query/key head_dim for {module_name!r}")
        block = transform.weight.shape[0]
        if dim % block:
            raise ValueError(
                f"Query/key transform block {block} does not divide "
                f"{module_name!r} head_dim {dim}"
            )
        value = matrix(transform, dim)
        previous = basis[transform.args.location].get(module_name)
        basis[transform.args.location][module_name] = (
            value if previous is None else compose(previous, value)
        )

    supported = ("hadamard", "random-hadamard", "random-matrix")
    retained = []
    for module_name, module in nodes:
        for child in module.children():
            if isinstance(child, TransformBase) and child.args.location in basis:
                if child.scheme.type not in supported:
                    raise ValueError(
                        "Cannot validate custom shared-cache query/key transform"
                    )
                append(module_name, module, child)

    for name, scheme, matches in planned:
        if scheme.type not in supported:
            raise ValueError(
                f"Cannot validate {scheme.type!r} query/key transforms "
                "for shared attention"
            )
        factory = TransformFactory.from_scheme(scheme, name=name)
        retained.append(factory)
        proxy = torch.nn.Module()
        proxy.register_parameter("weight", torch.nn.Parameter(torch.empty(0)))
        for module_name, module, arg in matches:
            transform = factory.create_transform(proxy, arg)
            retained.append(transform)
            if arg.location in basis:
                if hasattr(module, f"{name}_{arg.location}"):
                    raise ValueError(
                        f"Query/key transform {name!r} is already installed "
                        f"on {module_name!r}"
                    )
                append(module_name, module, transform)

    edges = [
        (name, module, *_producer(name, module, nodes)) for name, module in readers
    ]
    # Producers also consume their own keys; all affected local and shared dot
    # products must retain a matching basis, not only readers without an impl.
    edges.extend(
        (name, module, name, module)
        for name, module in nodes
        if not getattr(module, "is_kv_shared_layer", False)
    )
    checked = set()
    for reader_name, reader, producer_name, producer in edges:
        query = basis[q_location].get(reader_name)
        key = basis[k_location].get(producer_name)
        if query is None and key is None:
            continue
        dim = reader.head_dim
        identity = torch.eye(dim, dtype=torch.float64)
        identity_basis = (identity, torch.zeros_like(identity))
        query = identity_basis if query is None else query
        key = identity_basis if key is None else key
        signature = id(query), id(key)
        if signature in checked:
            continue
        same_shape = query[0].shape == key[0].shape
        if same_shape:
            product, rounding_bound = compose(query, (key[0].T, key[1].T))
            matches = torch.all(
                (product - identity).abs() <= 5e-6 + 5e-6 * identity + rounding_bound
            )
        else:
            matches = False
        if not matches:
            raise ValueError(
                f"Attention query/key transform mismatch: query at {reader_name!r} "
                f"must match the keys from {producer_name!r}. Target the actual "
                "producer and all its consumers with compatible transforms"
            )
        checked.add(signature)
