# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from copy import deepcopy

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization.quant_args import (
    FP8_E4M3_DATA,
    DynamicType,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from pydantic import BaseModel, ConfigDict, model_validator


__all__ = [
    "QuantizationScheme",
    "preset_name_to_scheme",
    "is_preset_scheme",
]


class QuantizationScheme(BaseModel, use_enum_values=True):
    """
    Set of QuantizationArgs defining how the weights, inputs and outputs of target list
    of modules should be quantized

    :param targets: list of modules to apply the QuantizationArgs to, can be layer
    names, layer types or a regular expression, typically ["Linear"]
    :param weights: quantization config for layer weights
    :param input_activations: quantization config for layer inputs
    :param output_activations: quantization config for layer outputs
    :param format: CompressionFormat for the layer
    """

    targets: list[str]
    weights: QuantizationArgs | None = None
    input_activations: QuantizationArgs | None = None
    output_activations: QuantizationArgs | None = None
    format: CompressionFormat | None = None

    @model_validator(mode="after")
    def validate_model_after(model: "QuantizationScheme") -> "QuantizationScheme":
        inputs = model.input_activations
        outputs = model.output_activations
        weights = model.weights
        format = model.format

        if inputs is not None:
            if inputs.strategy not in (
                QuantizationStrategy.TOKEN,
                QuantizationStrategy.TENSOR,
                QuantizationStrategy.GROUP,
                QuantizationStrategy.TENSOR_GROUP,
                QuantizationStrategy.ATTN_HEAD,
            ):
                raise NotImplementedError(
                    f"Using {inputs.strategy} strategy is not supported for "
                    "activation quantization"
                )

            if inputs.actorder is not None:
                raise ValueError("Cannot apply actorder to input activations")

        if outputs is not None:
            if outputs.actorder is not None:
                raise ValueError("Cannot apply actorder to output activations")

        if format == CompressionFormat.mixed_precision:
            raise ValueError(
                "mixed-precision cannot be set as a format for a QuantizationScheme"
            )

        if (
            inputs
            and weights
            and weights.strategy == QuantizationStrategy.GROUP
            and inputs.strategy == QuantizationStrategy.GROUP
            and weights.group_size != inputs.group_size
        ):
            warnings.warn(
                "Using GROUP strategy for both weights and input_activations "
                f"with different group sizes ({weights.group_size} vs "
                f"{inputs.group_size}) may complicate fused kernel implementations. "
                "Consider using TENSOR_GROUP strategy for both or matching group"
                " sizes.",
                UserWarning,
                stacklevel=2,
            )

        return model

    model_config = ConfigDict(extra="forbid")


"""
Pre-Set Quantization Scheme Args
"""


def _int_wnam(weight_bits: int, act_bits: int = 16) -> dict:
    if weight_bits < 2 or weight_bits > 8:
        raise ValueError(f"weight_bits must be 2-8, got {weight_bits}")
    if act_bits not in (4, 8, 16):
        raise ValueError(f"act_bits must be 4, 8, or 16, got {act_bits}")
    if weight_bits > act_bits:
        raise ValueError(
            f"weight_bits ({weight_bits}) must be <= act_bits ({act_bits})"
        )
    scheme = dict(
        weights=QuantizationArgs(
            num_bits=weight_bits,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
            symmetric=True,
            dynamic=False,
        ),
    )
    if act_bits < 16:
        scheme["input_activations"] = QuantizationArgs(
            num_bits=act_bits,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TOKEN,
            symmetric=True,
            dynamic=True,
        )
    return scheme


def preset_name_to_scheme(name: str, targets: list[str]) -> QuantizationScheme:
    """
    :param name: preset quantization settings name. must exist in upper case in
        PRESET_SCHEMES
    :param targets: list of quantization targets to be passed to the Scheme
    :return: new QuantizationScheme for a given name with the given targets
    """
    name = name.upper()

    if name not in PRESET_SCHEMES:
        raise KeyError(
            f"Unknown preset scheme name {name}, "
            f"available names: {list(PRESET_SCHEMES.keys())}"
        )

    scheme_args = deepcopy(PRESET_SCHEMES[name])  # deepcopy to avoid args references
    return QuantizationScheme(
        targets=targets,
        **scheme_args,
    )


def is_preset_scheme(name: str) -> bool:
    """
    :param name: preset quantization settings name
    :return: True if the name is a preset scheme name
    """
    return name.upper() in PRESET_SCHEMES


UNQUANTIZED = dict()

NVFP4A16 = dict(
    weights=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        symmetric=True,
        dynamic=False,
        group_size=16,
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=FP8_E4M3_DATA.dtype,
    )
)


NVFP4A16_BLOCK = dict(
    weights=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_BLOCK,
        symmetric=True,
        dynamic=False,
        block_structure=[16, 16],
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=FP8_E4M3_DATA.dtype,
    )
)


NVFP4 = dict(
    weights=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        symmetric=True,
        dynamic=False,
        group_size=16,
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=FP8_E4M3_DATA.dtype,
    ),
    input_activations=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        symmetric=True,
        dynamic=DynamicType.LOCAL,
        group_size=16,
        observer="static_minmax",
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=FP8_E4M3_DATA.dtype,
    ),
)

MXFP4A16 = dict(
    weights=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    )
)

MXFP4 = dict(
    weights=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    ),
    input_activations=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        dynamic=True,
        symmetric=True,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    ),
)

MXFP8A16 = dict(
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    )
)

MXFP8 = dict(
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    ),
    input_activations=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        dynamic=True,
        symmetric=True,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    ),
)


# Integer WxAy schemes (weight_bits <= act_bits)
W2A4 = _int_wnam(2, 4)
W2A8 = _int_wnam(2, 8)
W2A16 = _int_wnam(2)
W3A4 = _int_wnam(3, 4)
W3A8 = _int_wnam(3, 8)
W3A16 = _int_wnam(3)
W4A4 = _int_wnam(4, 4)
W4A8 = _int_wnam(4, 8)
W4A16 = _int_wnam(4)
W5A8 = _int_wnam(5, 8)
W5A16 = _int_wnam(5)
W6A8 = _int_wnam(6, 8)
W6A16 = _int_wnam(6)
W7A8 = _int_wnam(7, 8)
W7A16 = _int_wnam(7)
W8A16 = _int_wnam(8)

# W8A8 uses CHANNEL strategy for weights (distinct from the generic WxAy template)
INT8_W8A8 = dict(
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    ),
    input_activations=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    ),
)

# 4 bit integer weights only asymmetric quantization
W4A16_ASYM = dict(
    weights=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=128,
        symmetric=False,
        dynamic=False,
    ),
)

# 4 bit integer weights and 8 bit FP activations quantization
W4AFP8 = dict(
    weights=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=128,
        symmetric=True,
        dynamic=False,
    ),
    input_activations=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
        observer=None,
    ),
)

# FP8 weights and FP8 activations quantization
FP8 = dict(
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    ),
    input_activations=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
        observer="static_minmax",
    ),
)

# FP8 weights and FP8 dynamic activations quantization
FP8_DYNAMIC = dict(
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    ),
    input_activations=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    ),
)

# Block‐wise FP8 (deepseekv3-style quantization):
# static 128x128 per‐block weights and
# dynamic per‐token‐group activations
FP8_BLOCK = dict(
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.BLOCK,
        symmetric=True,
        dynamic=False,
        block_structure=[128, 128],
    ),
    input_activations=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=True,
        group_size=128,
    ),
)

PRESET_SCHEMES: dict[str, dict] = {
    # Unquantized (no-op)
    "UNQUANTIZED": UNQUANTIZED,
    # Special-cased integer schemes
    "W4A16_ASYM": W4A16_ASYM,
    "W8A8": INT8_W8A8,
    "INT8": INT8_W8A8,  # alias for W8A8
    "W4AFP8": W4AFP8,
    # Float weight and activation schemes
    "FP8": FP8,
    "FP8_DYNAMIC": FP8_DYNAMIC,
    "FP8_BLOCK": FP8_BLOCK,
    "NVFP4A16": NVFP4A16,
    "NVFP4A16_BLOCK": NVFP4A16_BLOCK,
    "NVFP4": NVFP4,
    "MXFP4A16": MXFP4A16,
    "MXFP4": MXFP4,
    "MXFP8A16": MXFP8A16,
    "MXFP8": MXFP8,
    # Integer WxAy schemes (weight_bits x act_bits, weight_bits <= act_bits)
    "W2A4": W2A4,
    "W2A8": W2A8,
    "W2A16": W2A16,
    "W3A4": W3A4,
    "W3A8": W3A8,
    "W3A16": W3A16,
    "W4A4": W4A4,
    "W4A8": W4A8,
    "W4A16": W4A16,
    "W5A8": W5A8,
    "W5A16": W5A16,
    "W6A8": W6A8,
    "W6A16": W6A16,
    "W7A8": W7A8,
    "W7A16": W7A16,
    "W8A16": W8A16,
}
