# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from typing import Literal, overload

import torch
from compressed_tensors import TRANSFORM_CONFIG_NAME
from compressed_tensors.offload import OffloadCache
from compressed_tensors.transform import TransformConfig, TransformFactory
from compressed_tensors.transform.factory.base import TransformBase
from pydantic import ValidationError
from transformers import PretrainedConfig


__all__ = ["apply_transform_config"]


_TRANSFORM_STATE_ATTR = "_ct_applied_transform_state"


def _serialize_transform_config(config: TransformConfig | dict) -> dict:
    return (
        config.model_dump() if isinstance(config, TransformConfig) else deepcopy(config)
    )


def _same_transform_config(
    first: TransformConfig | dict, second: TransformConfig | dict
):
    first = _serialize_transform_config(first)
    second = _serialize_transform_config(second)
    return first == second and tuple(first["config_groups"]) == tuple(
        second["config_groups"]
    )


def _prepare_restore_config(
    config: TransformConfig | dict,
) -> tuple[TransformConfig, TransformConfig | dict]:
    """Validate executable transforms while retaining offline-only annotations."""
    raw = _serialize_transform_config(config)
    metadata = None
    try:
        parsed = TransformConfig.model_validate(raw)
    except ValidationError as error:
        # An entirely fused group needs no runtime interpretation. Preserve extra
        # annotations there, but do not ignore unknown online execution semantics.
        groups = raw.get("config_groups", {}) if isinstance(raw, dict) else {}
        offline_groups = (
            {
                name
                for name, scheme in groups.items()
                if isinstance(scheme, dict)
                and isinstance(scheme.get("apply"), list)
                and all(
                    isinstance(arg, dict)
                    and arg.get("location") in ("weight_input", "weight_output")
                    for arg in scheme["apply"]
                )
            }
            if isinstance(groups, dict)
            else set()
        )
        errors = error.errors()
        if not all(
            item["type"] == "extra_forbidden"
            and len(item["loc"]) >= 3
            and item["loc"][0] == "config_groups"
            and item["loc"][1] in offline_groups
            for item in errors
        ):
            raise
        sanitized = deepcopy(raw)
        for item in errors:
            parent = sanitized
            for key in item["loc"][:-1]:
                parent = parent[key]
            del parent[item["loc"][-1]]
        parsed = TransformConfig.model_validate(sanitized)
        metadata = raw

    execution = parsed.model_copy(deep=True)
    execution.config_groups = {
        name: scheme
        for name, scheme in execution.config_groups.items()
        if any(arg.is_online() for arg in scheme.apply)
    }
    return execution, metadata if metadata is not None else parsed


def _hook_state(model: torch.nn.Module) -> dict:
    return {
        name: {
            key: tuple(value)
            for key, value in module.__dict__.items()
            if key.startswith("_forward_") and isinstance(value, dict)
        }
        for name, module in model.named_modules()
    }


@contextmanager
def _restore_transaction(model: torch.nn.Module):
    """Roll back module/hook registration and config changes after failed restore."""
    snapshots = []
    seen = set()
    pending = list(model.modules())
    while pending:
        obj = pending.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        attributes = obj.__dict__.copy()
        mappings = [
            (value, value.copy())
            for key, value in attributes.items()
            if isinstance(value, dict) and key not in ("_parameters", "_buffers")
        ]
        snapshots.append((obj, attributes, mappings))
        pending.extend(
            value
            for value in attributes.values()
            if isinstance(value, PretrainedConfig)
        )
    try:
        yield
    except BaseException:
        for obj, attributes, mappings in reversed(snapshots):
            for mapping, values in mappings:
                mapping.clear()
                mapping.update(values)
            obj.__dict__.clear()
            obj.__dict__.update(attributes)
        raise


@overload
def apply_transform_config(
    model: torch.nn.Module,
    config: TransformConfig,
    *,
    restore: bool = False,
) -> None:
    ...


@overload
def apply_transform_config(
    model: torch.nn.Module,
    config: dict,
    *,
    restore: Literal[True],
) -> None:
    ...


def apply_transform_config(
    model: torch.nn.Module,
    config: TransformConfig | dict,
    *,
    restore: bool = False,
):
    """
    Apply a transform config to a model. Weight transforms are fused into weights, while
    activation transforms are attached as submodules and trigger via pytorch hooks

    :param model: model to apply config to
    :param config: transform config to apply, or a saved dictionary when restore=True
    :param restore: restore online modules without reapplying fused weight transforms
    """
    previous_state = getattr(model, _TRANSFORM_STATE_ATTR, None)
    if restore:
        execution, metadata = _prepare_restore_config(config)
        state = previous_state
        if state is not None:
            if not _same_transform_config(state["config"], metadata):
                raise ValueError("A different transform config is already applied")
            modules = dict(model.named_modules())
            hooks = _hook_state(model)
            if any(
                not isinstance(modules.get(name), TransformBase)
                for name in state["modules"]
            ) or any(
                not set(keys).issubset(hooks.get(name, {}).get(key, ()))
                for name, values in state["hooks"].items()
                for key, keys in values.items()
            ):
                raise ValueError(
                    "Previously applied transform modules or hooks are missing"
                )
            return
        if any(isinstance(module, TransformBase) for module in model.modules()):
            raise ValueError("Cannot restore over transforms without application state")
    else:
        execution = metadata = config
        previous_config = getattr(model, TRANSFORM_CONFIG_NAME, None)
        if previous_config is not None:
            combined = _serialize_transform_config(previous_config)
            incoming = _serialize_transform_config(config)
            duplicate_groups = set(combined["config_groups"]) & set(
                incoming["config_groups"]
            )
            if duplicate_groups:
                raise ValueError(
                    f"Transform groups already applied: {sorted(duplicate_groups)}"
                )
            combined["config_groups"].update(incoming["config_groups"])
            _, metadata = _prepare_restore_config(combined)

    from compressed_tensors.transform.attention import validate_attention_transforms

    validate_attention_transforms(model, execution)
    hooks_before = _hook_state(model)
    with _restore_transaction(model) if restore else nullcontext():
        for name, scheme in execution.config_groups.items():
            factory = TransformFactory.from_scheme(scheme, name=name)
            if restore:
                factory.apply_to_model(model, restore=True)
            else:
                factory.apply_to_model(model)

        # declare shared transform parameters as tied weights for save_pretrained compat
        _register_tied_transform_weights(model)

        # Keep the full metadata: offline transforms are already fused, not absent.
        setattr(model, TRANSFORM_CONFIG_NAME, metadata)
        setattr(
            model,
            _TRANSFORM_STATE_ATTR,
            {
                "config": _serialize_transform_config(metadata),
                "modules": tuple(
                    name
                    for name, module in model.named_modules()
                    if isinstance(module, TransformBase)
                ),
                "hooks": {
                    name: {
                        key: tuple(
                            (set(keys) - set(hooks_before.get(name, {}).get(key, ())))
                            | set(
                                (previous_state or {})
                                .get("hooks", {})
                                .get(name, {})
                                .get(key, ())
                            )
                        )
                        for key, keys in values.items()
                    }
                    for name, values in _hook_state(model).items()
                },
            },
        )


def _register_tied_transform_weights(model: torch.nn.Module):
    """
    Scan for transform submodules that share parameters and register them as tied
    weights via ``_tied_weights_keys``. This allows ``save_pretrained`` in
    transformers v5+ to handle shared tensors without raising an error.

    Sharing is detected via the ``id`` of each parameter. For offloaded modules this
    must be done within :meth:`OffloadCache.disable_onloading`, otherwise each access
    onloads a *fresh* copy whose ``id`` differs across modules that share the same
    offloaded weight, and the sharing would go undetected.
    """
    # Map parameter id -> first full parameter name that owns it
    first_seen: dict[int, str] = {}

    # inspect offloaded tensors directly so shared weights keep a stable identity
    with OffloadCache.disable_onloading():
        for module_name, module in model.named_modules():
            if not isinstance(module, TransformBase):
                continue

            tied_keys: dict[str, str] = {}
            for key in getattr(module, "_dynamic_tied_weights_keys", []):
                param = getattr(module, key, None)
                if param is None:
                    continue

                param_id = id(param)
                full_key = f"{module_name}.{key}" if module_name else key

                if param_id not in first_seen:
                    first_seen[param_id] = full_key
                else:
                    # This parameter was already registered under a different module;
                    # mark it as tied so save_pretrained knows to deduplicate
                    tied_keys[key] = first_seen[param_id]

            if tied_keys:
                module._tied_weights_keys = tied_keys
                if hasattr(model, "all_tied_weights_keys"):
                    model.all_tied_weights_keys.update(
                        {
                            f"{module_name}.{key}": source
                            for key, source in tied_keys.items()
                        }
                    )
