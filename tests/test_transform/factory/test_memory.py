# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter

import pytest
import torch
from compressed_tensors.offload import (
    disable_offloading,
    disable_onloading,
    offload_model,
)
from compressed_tensors.transform import (
    TransformArgs,
    TransformBase,
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)
from tests.test_transform.conftest import TransformableModel
from tests.testing_utils import requires_gpu


@requires_gpu
@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("requires_grad", (True, False))
# @pytest.mark.parametrize("offload", (True, False))
@pytest.mark.parametrize("offload", (True,))
def test_memory_sharing(type, randomize, requires_grad, offload):
    # load model (maybe with offloading)
    model = TransformableModel(2, 2, 4, 4, 8, 8)
    if offload:
        offload_model(model, torch.device("cuda"))

    # add transforms to model
    config = TransformConfig(
        config_groups={
            "": TransformScheme(
                type=type,
                randomize=randomize,
                requires_grad=requires_grad,
                apply=[
                    TransformArgs(targets="Linear", location="input"),
                    TransformArgs(targets="Linear", location="output"),
                ],
            )
        }
    )
    apply_transform_config(model, config)

    for context in disable_onloading, disable_offloading:
        with context():
            weights = [
                m.weight for m in model.modules() if isinstance(m, TransformBase)
            ]
            weight_to_count = Counter(weights)
            size_to_weight = {weight.size(0): weight for weight in weight_to_count}

            assert len(weight_to_count) == len(size_to_weight) == 3
            assert weight_to_count[size_to_weight[2]] == 3
            assert weight_to_count[size_to_weight[4]] == 4
            assert weight_to_count[size_to_weight[8]] == 3
