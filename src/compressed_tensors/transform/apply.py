# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors import TRANSFORM_CONFIG_NAME
from compressed_tensors.transform import TransformConfig, TransformFactory
from compressed_tensors.transform.factory.base import TransformBase


__all__ = ["apply_transform_config"]


def apply_transform_config(model: torch.nn.Module, config: TransformConfig):
    """
    Apply a transform config to a model. Weight transforms are fused into weights, while
    activation transforms are attached as submodules and trigger via pytorch hooks

    :param model: model to apply config to
    :param config: transform config to apply
    """
    for name, scheme in config.config_groups.items():
        factory = TransformFactory.from_scheme(scheme, name=name)
        factory.apply_to_model(model)

    # declare shared transform parameters as tied weights for save_pretrained compat
    _register_tied_transform_weights(model)

    # attach config to model for compression/serialization
    setattr(model, TRANSFORM_CONFIG_NAME, config)


def _register_tied_transform_weights(model: torch.nn.Module):
    """
    Scan for transform submodules that share parameters and register them as tied
    weights via ``_tied_weights_keys``. This allows ``save_pretrained`` in
    transformers v5+ to handle shared tensors without raising an error.
    """
    # Map parameter id -> first full parameter name that owns it
    first_seen: dict[int, str] = {}

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
