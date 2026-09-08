# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# flake8: noqa
import pytest
import torch


def compressed_tensors_config_available():
    try:
        from transformers.utils.quantization_config import (  # noqa: F401
            CompressedTensorsConfig,
        )

        return True
    except ImportError:
        return False


_is_compressed_tensors_config_available = compressed_tensors_config_available()


def requires_hf_quantizer():
    return pytest.mark.skipif(
        not _is_compressed_tensors_config_available,
        reason="requires transformers>=4.45 to support CompressedTensorsHfQuantizer",
    )


def get_random_mat(M, K, dtype) -> "torch.Tensor":
    """
    :param M: number of rows
    :param K: number of columns
    :param dtype: data type of the matrix
    :return: random matrix of shape (M, K) with non-zero values
    """
    import torch
    from compressed_tensors.quantization import FP8_DTYPE

    rand_tensor_dtype = dtype
    if dtype in [torch.int8, FP8_DTYPE]:
        rand_tensor_dtype = torch.float16
    mat = torch.rand(M, K, dtype=rand_tensor_dtype).cuda()
    mat = mat.masked_fill_(mat == 0, 1)
    return mat.to(dtype)


def generate_pruned_semi_structured_mat(M, K, dtype) -> "torch.Tensor":
    """
    :param M: number of rows
    :param K: number of columns
    :param dtype: data type of the matrix
    :return: random matrix of shape (M, K) with 2:4 sparsity pattern
    """
    import torch
    from compressed_tensors.quantization import FP8_E4M3_DATA

    mask = torch.Tensor([0, 0, 1, 1]).tile((M, K // 4)).bool()
    rand_tensor_dtype = dtype
    if dtype in [torch.int8, FP8_E4M3_DATA.dtype]:
        rand_tensor_dtype = torch.float16
    mat = torch.rand(M, K, dtype=rand_tensor_dtype)
    mat = mat.masked_fill_(mat == 0, 1)
    if dtype == FP8_E4M3_DATA.dtype:
        # some float8_e4m3fn operations are not supported on CPU
        mat = mat.cuda()
        mask = mask.cuda()
    mat = mat * mask
    return mat.to(dtype)


def induce_sparsity(tensor, sparsity_ratio) -> "torch.Tensor":
    """
    Makes a tensor sparse by zeroing out a given fraction
    of its smallest absolute values.

    :param: weight_tensor (torch.Tensor): The input weight tensor.
    :param: sparsity_ratio (float): Fraction of weights to be zeroed
        (0 <= sparsity_ratio <= 1).
    :returns: torch.Tensor: Sparse version of the input tensor.
    """
    import torch

    if not (0 <= sparsity_ratio <= 1):
        raise ValueError("Sparsity ratio must be between 0 and 1.")

    # Flatten the tensor and compute the threshold for sparsity
    flattened = tensor.view(-1)
    k = int(sparsity_ratio * flattened.numel())

    if k > 0:
        threshold = torch.topk(flattened.abs(), k, largest=False).values.max()
        sparse_tensor = torch.where(
            tensor.abs() > threshold, tensor, torch.zeros_like(tensor)
        )
    else:
        sparse_tensor = tensor

    return sparse_tensor


def is_gpu_available():
    """
    :return: True if an accelerator device is available, False otherwise
    """
    try:
        import torch  # noqa: F401

        return torch.accelerator.device_count() > 0
    except ImportError:
        return False


def requires_version(package_name: str, req_version: str) -> pytest.MarkDecorator:
    """
    Pytest decorator to skip if the installed package version doesn't satisfy
    a version requirement. Supports operator prefixes: >, >=, <, <=, ==, !=.
    A bare version (no operator) is treated as >=.

    Usage:
    @requires_version("transformers", ">=4.45.0")
    @requires_version("transformers", ">5.15")
    @requires_version("transformers", "<6.0")
    """
    import operator
    import re
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version

    ops = {
        ">=": operator.ge,
        ">": operator.gt,
        "<=": operator.le,
        "<": operator.lt,
        "==": operator.eq,
        "!=": operator.ne,
    }

    match = re.match(r"^(>=|<=|!=|>|<|==)?(.+)$", req_version)
    op_str = match.group(1) or ">="
    ver_str = match.group(2)
    compare = ops[op_str]

    try:
        installed = version(package_name)
    except PackageNotFoundError:
        return pytest.mark.skip(
            reason=f"{package_name} is not installed",
        )

    return pytest.mark.skipif(
        not compare(Version(installed), Version(ver_str)),
        reason=(f"{package_name} {op_str}{ver_str} required, " f"found {installed}"),
    )


def requires_gpu(test_case_or_num):
    """
    Pytest decorator to skip based on number of available GPUs.

    Designed for backwards compatibility with the old requires_gpu decorator
    Usage:
    @requires_gpu
    def test_something():
        # only runs if there is at least 1 GPU available
        pass

    @requires_gpu(2)
    def test_something_else():
        # only runs if there are at least 2 GPUs available
        pass
    """
    if isinstance(test_case_or_num, int):
        num_required_gpus = test_case_or_num
    else:
        num_required_gpus = 1

    decorator = pytest.mark.skipif(
        (torch.accelerator.device_count() < num_required_gpus),
        reason=f"Not enough GPUs available, {num_required_gpus} GPUs required",
    )
    if isinstance(test_case_or_num, int):
        return decorator
    else:
        return decorator(test_case_or_num)
