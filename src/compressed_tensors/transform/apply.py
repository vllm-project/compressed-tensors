# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict

import torch
from compressed_tensors import TRANSFORM_CONFIG_NAME
from compressed_tensors.transform import TransformConfig, TransformFactory


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

    # attach config to model for compression/serialization
    setattr(model, TRANSFORM_CONFIG_NAME, config)

    # populate `_tied_weights_keys` for proper loading by transformers
    _update_transforms_tied_weights(model)


def _update_transforms_tied_weights(model: torch.nn.Module):
    """
    This function updates the `_tied_weights_keys` and `all_tied_weights_keys`
    attributes of the given model with transform weights.

    This function is needed because transformers only knows which weights are shared
    via the `_tied_weights_keys` attributes. These attributes are used to tie
    weights after the model has loaded.

    CompressedTensors does not enforce a particular weight is the source weight :.
    We rely on correctness of the following mapping in PreTrainedModel.tie_weights():
    ```
    B -> A
    C -> A
    D -> A

    Where any of A,B,C,D might be the loaded source weight
    ```
    This property is tested by `test_modeling_utils::BaseModelWithMultipleTiedWeights`
    """
    from compressed_tensors.transform import TransformBase

    # 1. find which transform weights are shared
    # create mapping: tensor_hash -> key
    weight_to_keys: dict[int, str] = defaultdict(list)
    for name, module in model.named_modules():
        if isinstance(module, TransformBase):
            for param_name, param_hash in module.tied_weights_hash.items():
                param_fqn = f"{name}.{param_name}" if name else param_name
                weight_to_keys[param_hash].append(param_fqn)

    # 2. assign each group of shared weights to the same value
    # create tied weights: key -> tied_keys[0]
    transform_tied_weights_keys = {}
    for keys in weight_to_keys.values():
        keys = list(keys)
        for key in keys[1:]:  # skip A -> A
            transform_tied_weights_keys[key] = keys[0]

    # 3. update tied weights attributes
    if getattr(model, "_tied_weights_keys", None) is None:
        model._tied_weights_keys = {}
    model._tied_weights_keys.update(transform_tied_weights_keys)
    model.all_tied_weights_keys = model._tied_weights_keys
