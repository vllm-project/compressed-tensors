# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

import pytest
import torch
from compressed_tensors.offload import set_onload_device
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformFactory,
    TransformScheme,
    apply_transform_config,
)
from tests.test_transform.conftest import MockAttention, MockAttentionModel
from tests.testing_utils import requires_gpu


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


class _SubclassedEmbedding(torch.nn.Embedding):
    pass


class _SubclassedLinear(torch.nn.Linear):
    pass


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard", "random-matrix"))
@pytest.mark.parametrize("randomize", (True, False))
def test_correctness_module_subclasses(type, randomize):
    model = torch.nn.Sequential(
        _SubclassedEmbedding(2, 4),
        _SubclassedLinear(4, 8, bias=False),
    )

    input = torch.randint(high=2, low=0, size=(17, 5, 2))
    true_output = model(input)

    config = TransformConfig(
        config_groups={
            "": TransformScheme(
                type=type,
                randomize=randomize,
                apply=[
                    TransformArgs(targets="Embedding", location="weight_output"),
                    TransformArgs(
                        targets="Linear", location="weight_input", inverse=True
                    ),
                    TransformArgs(targets="Linear", location="weight_output"),
                    TransformArgs(targets="Linear", location="output", inverse=True),
                ],
            )
        }
    )
    apply_transform_config(model, config)

    # compare outputs
    output = model(input)
    assert torch.allclose(true_output, output, atol=1e-5, rtol=0.0)


def test_apply_transform_config_respects_module_scope():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 4, bias=False),
    )
    config = TransformConfig(
        config_groups={
            "input": TransformScheme(
                type="hadamard",
                apply=[TransformArgs(targets="Linear", location="input")],
            )
        }
    )

    apply_transform_config(model, config, modules=[model[0]])

    assert "input_input" in model[0]._modules
    assert "input_input" not in model[1]._modules


@pytest.mark.parametrize("scope_type", [list, iter])
def test_factory_module_scope_can_be_reused_across_transform_arguments(scope_type):
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 4, bias=False),
    )
    inputs = torch.randn(2, 4)
    expected = model(inputs)
    scheme = TransformScheme(
        type="hadamard",
        apply=[
            TransformArgs(targets="Linear", location="input"),
            TransformArgs(targets="Linear", location="weight_input", inverse=True),
        ],
    )
    factory = TransformFactory.from_scheme(scheme, name="rotation")

    factory.apply_to_model(model, use_tqdm=False, modules=scope_type([model[1]]))

    assert "rotation_input" not in model[0]._modules
    assert "rotation_input" in model[1]._modules
    torch.testing.assert_close(model(inputs), expected)


def test_apply_transform_config_reuses_factories_across_scopes():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 4, bias=False),
    )
    config = TransformConfig(
        config_groups={
            "input": TransformScheme(
                type="hadamard",
                apply=[TransformArgs(targets="Linear", location="input")],
            )
        }
    )
    factories = {
        "input": TransformFactory.from_scheme(
            config.config_groups["input"], name="input"
        )
    }

    apply_transform_config(model, config, modules=[model[0]], factories=factories)
    apply_transform_config(model, config, modules=[model[1]], factories=factories)

    assert model[0].input_input.weight is model[1].input_input.weight


@pytest.mark.parametrize("type", ["hadamard", "random-hadamard", "random-matrix"])
def test_shared_factories_preserve_paired_transforms_across_mixed_width_scopes(type):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 8, bias=False),
        torch.nn.Linear(8, 8, bias=False),
    ).double()
    inputs = torch.randn(2, 4, dtype=torch.float64)
    expected = model(inputs)
    eager_model = deepcopy(model)
    scheme = TransformScheme(
        type=type,
        apply=[
            TransformArgs(targets="0", location="weight_output"),
            TransformArgs(targets="1", location="weight_input", inverse=True),
            TransformArgs(targets="1", location="weight_output"),
            TransformArgs(targets="2", location="weight_input", inverse=True),
        ],
    )
    config = TransformConfig(config_groups={"rotation": scheme})
    factories = {"rotation": TransformFactory.from_scheme(scheme, name="rotation")}

    apply_transform_config(eager_model, config)
    for module in model:
        apply_transform_config(model, config, modules=[module], factories=factories)

    torch.testing.assert_close(model(inputs), expected, atol=1e-10, rtol=0)
    for actual, eager in zip(model.parameters(), eager_model.parameters()):
        torch.testing.assert_close(actual, eager, atol=1e-10, rtol=0)


@requires_gpu
@pytest.mark.parametrize("type", ("hadamard", "random-hadamard", "random-matrix"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("input_batch_size", (1, 5, 17))
@pytest.mark.parametrize("offload", (True, False))
def test_correctness_model(type, randomize, input_batch_size, model_apply, offload):
    # load model
    model = model_apply[0]
    if offload:
        set_onload_device(model, torch.device("cuda"))

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


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard", "random-matrix"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("head_dim", (None, 2, 4))
@pytest.mark.parametrize("input_batch_size", (1, 5))
def test_correctness_linear_with_bias(type, randomize, head_dim, input_batch_size):
    """Test that WEIGHT_OUTPUT transforms correctly handle bias.

    For WEIGHT_OUTPUT on a Linear with bias:
        y' = R @ (W @ x + b) = (R @ W) @ x + (R @ b)
    Both weight and bias must be transformed.
    """
    size = (4, 8)
    module = torch.nn.Linear(*size, bias=True)

    input = torch.rand((input_batch_size, 5, size[0]))

    # Apply WEIGHT_OUTPUT (transforms weight AND bias) then WEIGHT_INPUT inverse
    config = TransformConfig(
        config_groups={
            "": TransformScheme(
                type=type,
                randomize=randomize,
                head_dim=head_dim,
                apply=[
                    TransformArgs(targets="0", location="weight_output"),
                    TransformArgs(targets="1", location="weight_input", inverse=True),
                ],
            )
        }
    )
    model = torch.nn.Sequential(module, torch.nn.Linear(size[1], 16, bias=False))
    true_output_full = model(input)
    apply_transform_config(model, config)

    output = model(input)
    assert torch.allclose(true_output_full, output, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("type", ("hadamard", "random-hadamard", "random-matrix"))
@pytest.mark.parametrize("randomize", (True, False))
@pytest.mark.parametrize("head_dim", (4, 8))
@pytest.mark.parametrize("input_batch_size", (1, 5))
def test_correctness_attention_heads_with_bias(
    type, randomize, head_dim, input_batch_size
):
    """Test R2 head-wise rotation with attention bias (e.g. Qwen2 v_proj).

    When v_proj has bias and R2 WEIGHT_OUTPUT is applied, the bias must also
    be rotated for the o_proj WEIGHT_INPUT inverse to correctly cancel out.
    """
    hidden_size = 64
    num_attention_heads = 8

    attention = MockAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=head_dim,
    )
    # Add bias to v_proj to simulate Qwen2-like architecture
    attention.v_proj.bias = torch.nn.Parameter(
        torch.randn(attention.v_proj.out_features)
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
