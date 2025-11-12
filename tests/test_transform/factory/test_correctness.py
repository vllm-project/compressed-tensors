# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformFactory,
    TransformScheme,
    apply_transform_config,
)
from compressed_tensors.utils import offloaded_dispatch
from tests.test_transform.conftest import MockAttention, MockAttentionModel
from tests.testing_utils import requires_accelerate, requires_gpu


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard", "random-matrix"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("head_dim", (None, 2, 4))
@pytest.mark.parametrize("input_batch_size", (1, 5, 17))
def test_correctness_linear(type, randomize, head_dim, input_batch_size):
    size = (4, 8)
    module = torch.nn.Linear(*size, bias=False)
    scheme = TransformScheme(type=type, randomize=randomize, head_dim=head_dim)
    factory = TransformFactory.from_scheme(scheme, name="")

    input_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="input", inverse=True)
    )
    w_in_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="weight_input")
    )
    w_out_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="weight_output")
    )
    output_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="output", inverse=True)
    )

    input = torch.rand((input_batch_size, 5, size[0]))
    true_output = input @ module.weight.T
    input_transformed = input_tfm(input)
    weight_transformed = w_out_tfm(w_in_tfm(module.weight))
    output = output_tfm(input_transformed @ weight_transformed.T)
    assert torch.allclose(true_output, output, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard", "random-matrix"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("embed_loc", ("weight_output", "output"))
@pytest.mark.parametrize("linear_loc", ("input", "weight_input"))
def test_correctness_embedding(type, randomize, embed_loc, linear_loc):
    model = torch.nn.Sequential(
        torch.nn.Embedding(2, 4),
        torch.nn.Linear(4, 8, bias=False),
    )

    input = torch.randint(high=1, low=0, size=(17, 5, 2))
    true_output = model(input)

    config = TransformConfig(
        config_groups={
            "": TransformScheme(
                type=type,
                randomize=randomize,
                apply=[
                    TransformArgs(targets="Embedding", location=embed_loc),
                    TransformArgs(targets="Linear", location=linear_loc, inverse=True),
                ],
            )
        }
    )
    apply_transform_config(model, config)

    # compare outputs
    output = model(input)
    assert torch.allclose(true_output, output, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard", "random-matrix"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("input_batch_size", (1, 5, 17))
def test_correctness_model(
    type, randomize, input_batch_size, model_apply, offload=False
):
    # load model
    model = model_apply[0]
    if offload:
        model = offloaded_dispatch(model, torch.device("cuda"))

    # get output
    input = torch.rand((input_batch_size, 5, model.fcs[0].in_features))
    if offload:
        input = input.to(torch.device("cuda"))
    true_output = model(input)

    # apply transforms
    config = TransformConfig(
        config_groups={
            "": TransformScheme(type=type, randomize=randomize, apply=model_apply[1])
        }
    )
    apply_transform_config(model, config)

    # compare outputs
    output = model(input)
    assert torch.allclose(true_output, output, atol=1e-5, rtol=0.0)


@requires_gpu
@requires_accelerate()
@pytest.mark.parametrize("type", ("hadamard", "random-hadamard", "random-matrix"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("input_batch_size", (1, 5, 17))
def test_correctness_model_offload(type, randomize, input_batch_size, model_apply):
    test_correctness_model(type, randomize, input_batch_size, model_apply, offload=True)


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard", "random-matrix"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("head_dim", (4, 8))
@pytest.mark.parametrize("input_batch_size", (1, 5, 17))
def test_correctness_attention_heads(type, randomize, head_dim, input_batch_size):
    hidden_size = 64
    num_attention_heads = 8

    attention = MockAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=head_dim,
    )

    input = torch.rand(input_batch_size, 5, hidden_size)
    true_output = attention(input)

    config = TransformConfig(
        config_groups={
            "R2": TransformScheme(
                type=type,
                randomize=randomize,
                head_dim=head_dim,
                apply=[
                    TransformArgs(targets="v_proj", location="weight_output"),
                    TransformArgs(
                        targets="o_proj", location="weight_input", inverse=True
                    ),
                ],
            )
        }
    )
    apply_transform_config(attention, config)

    output = attention(input)
    assert torch.allclose(true_output, output, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("head_dim", (4, 8))
@pytest.mark.parametrize("input_batch_size", (1, 5, 17))
def test_correctness_query_key_locations(type, randomize, head_dim, input_batch_size):
    hidden_size = 64
    num_attention_heads = 8

    model = MockAttentionModel(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=head_dim,
    )

    input = torch.rand(input_batch_size, 5, hidden_size)
    true_output = model(input)

    config = TransformConfig(
        config_groups={
            "R3": TransformScheme(
                type=type,
                randomize=randomize,
                head_dim=head_dim,
                apply=[
                    TransformArgs(targets="self_attn", location="q_attn"),
                    TransformArgs(targets="self_attn", location="k_cache"),
                ],
            )
        }
    )
    apply_transform_config(model, config)

    output = model(input)
    assert torch.allclose(true_output, output, atol=1e-5, rtol=0.0)


@requires_gpu
@pytest.mark.parametrize("cuda_default", (True, False))
def test_random_matrix_device_handling(cuda_default):
    """
    Test that random-matrix transforms can be created
    on CUDA.
    """
    seed = 0
    size = (4, 8)

    cur_default = torch.get_default_device()
    if cuda_default:
        torch.set_default_device("cuda")
    module = torch.nn.Linear(*size, bias=False).cuda()
    scheme = TransformScheme(type="random-matrix", randomize=True)
    factory = TransformFactory.from_scheme(scheme, name="", seed=seed)

    # Create transforms - this should work despite CPU generator and CUDA module
    input_tfm = factory.create_transform(
        module, TransformArgs(targets="Linear", location="input", inverse=True)
    )

    # Verify transforms work correctly on CUDA
    input = torch.rand((5, 3, size[0])).cuda()
    input_tfm(input)

    # Verify that transforms were created on CUDA
    assert input_tfm.weight.device.type == "cuda"
    torch.set_default_device(cur_default)
