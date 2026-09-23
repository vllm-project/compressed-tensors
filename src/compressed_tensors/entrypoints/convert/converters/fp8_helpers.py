# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import re
from collections.abc import Iterable

import torch
from compressed_tensors.utils.match import match_name
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_safetensors_header,
)


__all__ = [
    "SCALE_PARAM_NAMES",
    "classify_targets",
    "generalize_name",
    "encode_fp8_scale",
    "encode_fp4_scale",
]

# safetensors dtype strings (as found in the file header) for quantized weights
_FP8_WEIGHT_DTYPES = {"F8_E4M3", "F8_E5M2"}
_FP4_WEIGHT_DTYPES = {"I8", "U8"}  # two packed e2m1 values per byte
# scale param names that different fp8 checkpoints use for the per-block weight scale
SCALE_PARAM_NAMES = ("weight_scale_inv", "scale")
# checkpoint scales stored as float8_e8m0fnu (a raw biased exponent byte); absent on
# torch versions that predate the dtype
_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)


def classify_targets(
    model_name_or_path: str, ignore: Iterable[str]
) -> tuple[list[str], list[str]]:
    """
    Read the safetensors headers and bucket every quantized module (a ``.weight``
    that has an accompanying scale) into fp8 vs fp4 targets based on the stored
    weight dtype. Module names are generalized into regex targets by replacing
    numeric indices (layer/expert ids) with ``\\d+`` so the resulting config stays
    compact.

    Only tensor headers are inspected, never the tensor data, so this stays cheap
    even for multi-hundred-GB checkpoints.
    """
    weight_dtypes, all_names = _read_weight_dtypes(model_name_or_path)

    fp8_patterns: set[str] = set()
    fp4_patterns: set[str] = set()
    for weight_name, dtype in weight_dtypes.items():
        module_name = weight_name[: -len(".weight")]
        if not any(f"{module_name}.{s}" in all_names for s in SCALE_PARAM_NAMES):
            continue  # unquantized weight (e.g. norms), skip
        if any(match_name(module_name, ign) for ign in ignore):
            continue
        if dtype in _FP8_WEIGHT_DTYPES:
            fp8_patterns.add(generalize_name(module_name))
        elif dtype in _FP4_WEIGHT_DTYPES:
            fp4_patterns.add(generalize_name(module_name))

    return sorted(fp8_patterns), sorted(fp4_patterns)


def _read_weight_dtypes(
    model_name_or_path: str,
) -> tuple[dict[str, str], set[str]]:
    """
    Return ``(weight_name -> safetensors dtype string, set of all tensor names)``
    by reading only safetensors headers. Local directories are read file-by-file;
    Hub stubs use the metadata API to avoid downloading weights.
    """
    weight_dtypes: dict[str, str] = {}
    all_names: set[str] = set()

    if os.path.exists(model_name_or_path):
        for rel_path, path in get_checkpoint_files(model_name_or_path).items():
            if not rel_path.endswith(".safetensors"):
                continue
            for name, info in get_safetensors_header(path).items():
                if name == "__metadata__":
                    continue
                all_names.add(name)
                if name.endswith(".weight"):
                    weight_dtypes[name] = info["dtype"]
    else:
        from huggingface_hub import HfApi

        metadata = HfApi().get_safetensors_metadata(model_name_or_path)
        for file_metadata in metadata.files_metadata.values():
            for name, tensor in file_metadata.tensors.items():
                all_names.add(name)
                if name.endswith(".weight"):
                    weight_dtypes[name] = tensor.dtype

    return weight_dtypes, all_names


def generalize_name(module_name: str) -> str:
    """Turn a concrete module name into a regex target, replacing numeric indices
    with ``\\d+``, e.g. ``model.layers.3.experts.7.w1`` ->
    ``re:.*layers\\.\\d+\\.experts\\.\\d+\\.w\\d+$``.
    """
    pattern = re.sub(r"\d+", lambda _m: r"\d+", module_name).replace(".", r"\.")
    return f"re:.*{pattern}$"


def encode_fp8_scale(scale: torch.Tensor) -> torch.Tensor:
    """
    The ``float-quantized`` block compressor multiplies the weight by the scale
    directly and has no ``float8_e8m0fnu`` kernel, so UE8M0 scales must be promoted
    to fp32 (a lossless power-of-2 cast). fp32 scales (V3-style) pass through
    unchanged.
    """
    if _E8M0_DTYPE is not None and scale.dtype == _E8M0_DTYPE:
        return scale.to(torch.float32)
    return scale


def encode_fp4_scale(scale: torch.Tensor) -> torch.Tensor:
    """
    Compressed-tensors' MXFP4 compressor stores weight_scale as the raw E8M0
    biased-exponent byte (uint8) and decodes it as ``2 ** (byte - 127)``. The
    checkpoint's ``float8_e8m0fnu`` scale already holds exactly that byte in its
    bit pattern, so reinterpret rather than value-cast (which would build the wrong
    integer from the float value).
    """
    if _E8M0_DTYPE is not None and scale.dtype == _E8M0_DTYPE:
        return scale.view(torch.uint8)
    if scale.dtype == torch.uint8:
        return scale
    # float32 scales (V3-style): encode to E8M0 biased exponent
    return (
        (torch.log2(scale).floor().to(torch.int32) + 127).clamp(0, 255).to(torch.uint8)
    )
