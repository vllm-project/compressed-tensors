# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.quant_scheme import (
    FP8,
    FP8_BLOCK,
    FP8_DYNAMIC,
    MXFP8,
    NVFP4,
    NVFP4A16,
)
from compressed_tensors.utils.match import match_name
from compressed_tensors.utils.safetensors_load import (
    find_config_path,
    get_quantization_config,
)
from huggingface_hub import list_repo_files
from transformers.file_utils import CONFIG_NAME
from transformers.utils import cached_file


__all__ = ["ModelOptConverter"]


@dataclass(frozen=True)
class _ModelOptAlgo:
    """
    Declarative mapping of one ModelOpt ``quant_algo`` onto compressed-tensors:
    the target QuantizationScheme, checkpoint format, the ModelOpt params a
    quantized module carries, and the params its converted weight requires.

    :param aliases: ModelOpt ``quant_algo`` spellings selecting this algo
    :param scheme: builds QuantizationScheme kwargs (weights /
        input_activations) from the ModelOpt group_size
    :param format: compressed-tensors format for the resulting config group
    :param param_names: ModelOpt qparam names (besides ``weight``) carried by
        a module quantized with this algo
    :param weight_param: name of the module's weight param after conversion
    :param weight_dtypes: dtypes the converted weight may take; empty skips
        the dtype check
    :param required_scales: converted scale params that must accompany the
        converted weight
    :param packs_int_weight: whether the unpacked int8 weight is repacked into
        ``weight_packed``/``weight_shape`` (compressed-tensors pack-quantized)
    :param drops_input_scale: whether a (legacy) ModelOpt ``input_scale`` is
        dropped instead of converted
    """

    aliases: tuple[str, ...]
    scheme: Callable[[int | None], dict[str, Any]]
    format: CompressionFormat
    param_names: tuple[str, ...] = ()
    weight_param: str = "weight"
    weight_dtypes: tuple[torch.dtype, ...] = ()
    required_scales: tuple[str, ...] = ()
    packs_int_weight: bool = False
    drops_input_scale: bool = False


def _nvfp4_scheme(group_size: int | None) -> dict[str, Any]:
    if group_size not in (None, 16):
        raise ValueError(f"NVFP4 requires group_size 16, got {group_size}")
    return dict(NVFP4)


def _w4a16_nvfp4_scheme(group_size: int | None) -> dict[str, Any]:
    if group_size not in (None, 16):
        raise ValueError(f"W4A16_NVFP4 requires group_size 16, got {group_size}")
    return dict(NVFP4A16)


def _fp8_block_scheme(group_size: int | None) -> dict[str, Any]:
    scheme = dict(FP8_BLOCK)
    if group_size is not None and group_size != 128:
        scheme["weights"] = scheme["weights"].model_copy(
            update={"block_structure": [group_size, group_size]}
        )
    return scheme


def _int8_w8a8_scheme(group_size: int | None) -> dict[str, Any]:
    # ModelOpt INT8_SQ (smoothquant): per-channel weight scale plus a static
    # per-tensor input scale, kept as-is by the int-quantized format
    return dict(
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
            strategy=QuantizationStrategy.TENSOR,
            symmetric=True,
            dynamic=False,
        ),
    )


def _int8_group_scheme(group_size: int | None) -> dict[str, Any]:
    return dict(
        weights=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=group_size or 128,
            symmetric=True,
            dynamic=False,
        ),
    )


def _int8_channel_scheme(group_size: int | None) -> dict[str, Any]:
    # ModelOpt INT8_WO: weight-only per-channel int8
    return dict(
        weights=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False,
        ),
    )


_ALGOS: dict[str, _ModelOptAlgo] = {
    "nvfp4": _ModelOptAlgo(
        aliases=("NVFP4",),
        scheme=_nvfp4_scheme,
        format=CompressionFormat.nvfp4_pack_quantized,
        param_names=("input_scale", "weight_scale", "weight_scale_2"),
        weight_param="weight_packed",
        weight_dtypes=(torch.uint8,),
        required_scales=("weight_scale", "weight_global_scale"),
    ),
    "w4a16_nvfp4": _ModelOptAlgo(
        aliases=("W4A16_NVFP4",),
        scheme=_w4a16_nvfp4_scheme,
        format=CompressionFormat.nvfp4_pack_quantized,
        param_names=("weight_scale", "weight_scale_2"),
        weight_param="weight_packed",
        weight_dtypes=(torch.uint8,),
        required_scales=("weight_scale", "weight_global_scale"),
        drops_input_scale=True,
    ),
    "fp8": _ModelOptAlgo(
        aliases=("FP8",),
        scheme=lambda group_size: dict(FP8),
        format=CompressionFormat.float_quantized,
        param_names=("weight_scale", "input_scale"),
        weight_param="weight",
        weight_dtypes=(torch.float8_e4m3fn,),
        required_scales=("weight_scale", "input_scale"),
    ),
    "fp8_per_channel": _ModelOptAlgo(
        aliases=("FP8_PER_CHANNEL_PER_TOKEN",),
        scheme=lambda group_size: dict(FP8_DYNAMIC),
        format=CompressionFormat.float_quantized,
        param_names=("weight_scale",),
        weight_param="weight",
        weight_dtypes=(torch.float8_e4m3fn,),
        required_scales=("weight_scale",),
    ),
    "fp8_block": _ModelOptAlgo(
        aliases=("FP8_PB_WO", "FP8_BLOCK_SCALES"),
        scheme=_fp8_block_scheme,
        format=CompressionFormat.float_quantized,
        param_names=("weight_scale",),
        weight_param="weight",
        weight_dtypes=(torch.float8_e4m3fn,),
        required_scales=("weight_scale_inv",),
    ),
    "int8_static": _ModelOptAlgo(
        aliases=("INT8_SQ", "W8A8_SQ_PER_CHANNEL"),
        scheme=_int8_w8a8_scheme,
        format=CompressionFormat.int_quantized,
        param_names=("weight_scale", "input_scale"),
        weight_param="weight",
        weight_dtypes=(torch.int8,),
        required_scales=("weight_scale", "input_scale"),
    ),
    "int8_group": _ModelOptAlgo(
        aliases=("INT8_PB_WO",),
        scheme=_int8_group_scheme,
        format=CompressionFormat.pack_quantized,
        param_names=("weight_scale",),
        weight_param="weight_packed",
        weight_dtypes=(torch.int32,),
        required_scales=("weight_scale",),
        packs_int_weight=True,
    ),
    "int8_channel": _ModelOptAlgo(
        aliases=("INT8_WO", "W8A16"),
        scheme=_int8_channel_scheme,
        format=CompressionFormat.pack_quantized,
        param_names=("weight_scale",),
        weight_param="weight_packed",
        weight_dtypes=(torch.int32,),
        required_scales=("weight_scale",),
        packs_int_weight=True,
    ),
    "mxfp8": _ModelOptAlgo(
        aliases=("MXFP8",),
        scheme=lambda group_size: dict(MXFP8),
        format=CompressionFormat.mxfp8_quantized,
        param_names=("weight_scale",),
        weight_param="weight",
        weight_dtypes=(torch.float8_e4m3fn,),
        required_scales=("weight_scale",),
    ),
}


_ALGO_FOR_ALIAS = {
    alias.upper(): canonical
    for canonical, algo in _ALGOS.items()
    for alias in algo.aliases
}

# ModelOpt algos compressed-tensors cannot faithfully represent: AWQ-style
# formats repack int4 along the output dimension (compressed-tensors packs
# densely along the input dimension) and fold a pre_quant_scale into the
# preceding layernorm; GGML block codecs have no compressed-tensors scheme;
# MXFP4/MXFP6 use a different FP4/MX packing than nvfp4-pack-quantized.
_UNSUPPORTED_ALGOS = {
    "W4A16_AWQ",
    "W4A8_AWQ",
    "NVFP4_AWQ",
    "NVFP4_SVD",
    "NVFP4_SVDQUANT",
    "W4A8_NVFP4_FP8",
    "W4A8_MXFP4_FP8",
    "MXFP4",
    "W4A16_MXFP4",
    "MXFP6",
    "W4A16_MXFP6",
    "IQ1_S",
    "IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "Q8_0",
}

_KV_CACHE_QUANT_ALGOS = {
    "FP8": QuantizationArgs(
        num_bits=8, type=QuantizationType.FLOAT, symmetric=True, dynamic=False
    ),
    "INT8": QuantizationArgs(
        num_bits=8, type=QuantizationType.INT, symmetric=True, dynamic=False
    ),
}

# ModelOpt qparams that only belong to quantized modules; a non-ignored
# module carrying one of these without a quantization assignment indicates
# an untargeted or orphan qparam
_SOURCE_QPARAMS = {"input_scale", "weight_scale", "weight_scale_2"}


def _canonicalize_algo(quant_algo: str) -> str:
    """Map a ModelOpt quant_algo string onto its canonical algo key."""
    algo = str(quant_algo).upper()
    if algo in _ALGO_FOR_ALIAS:
        return _ALGO_FOR_ALIAS[algo]
    supported = sorted(
        alias for algo_spec in _ALGOS.values() for alias in algo_spec.aliases
    )
    if algo in _UNSUPPORTED_ALGOS:
        raise NotImplementedError(
            f"ModelOpt quant_algo {quant_algo!r} is not supported by "
            f"ModelOptConverter; supported algos: {supported}"
        )
    raise NotImplementedError(
        f"Unknown ModelOpt quant_algo {quant_algo!r}; supported algos: {supported}"
    )


def _coerce_kv_cache_scheme(
    kv_cache_scheme: QuantizationArgs | dict[str, Any] | None,
) -> QuantizationArgs | None:
    if isinstance(kv_cache_scheme, dict):
        return QuantizationArgs(**kv_cache_scheme)
    return kv_cache_scheme


def _parse_kv_cache_scheme(config: dict[str, Any]) -> QuantizationArgs | None:
    """Read the kv cache scheme from a ModelOpt quantization config.

    config.json style stores a compressed-tensors style ``kv_cache_scheme``
    dict, while the legacy hf_quant_config.json stores a
    ``kv_cache_quant_algo`` string.
    """
    scheme = config.get("kv_cache_scheme")
    if isinstance(scheme, dict) and scheme:
        return QuantizationArgs(**scheme)

    quant_algo = config.get("kv_cache_quant_algo")
    if quant_algo is None:
        return None
    quant_algo = str(quant_algo).upper()
    if quant_algo not in _KV_CACHE_QUANT_ALGOS:
        raise ValueError(
            f"Unsupported ModelOpt kv_cache_quant_algo {quant_algo!r}; "
            f"supported: {sorted(_KV_CACHE_QUANT_ALGOS)}"
        )
    return _KV_CACHE_QUANT_ALGOS[quant_algo]


def _modelopt_ignore_to_ct(ignore: Iterable[str]) -> list[str]:
    """Convert ModelOpt exclude patterns to compressed-tensors ignore entries.

    ModelOpt matches ``ignore``/``exclude_modules`` entries with fnmatch
    wildcards, while compressed-tensors matches exact names or ``re:``-prefixed
    regexes, so wildcard entries are translated to anchored regexes.
    """
    converted = []
    for entry in ignore:
        if any(char in entry for char in "*?["):
            pattern = re.escape(entry).replace(r"\*", ".*").replace(r"\?", ".")
            converted.append("re:" + pattern)
        else:
            converted.append(entry)
    return converted


def _is_modelopt_config(config: Any) -> bool:
    """Whether a parsed quantization config dict is in ModelOpt format."""
    if not isinstance(config, dict):
        return False
    if "quantization" in config:  # hf_quant_config.json layout
        return True
    quant_method = config.get("quant_method", "")
    return isinstance(quant_method, str) and quant_method.lower().startswith("modelopt")


def _unwrap_modelopt_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Unwrap the hf_quant_config.json layout, ``{"producer": ..., "quantization":
    {...}}``."""
    if isinstance(config, dict) and isinstance(config.get("quantization"), dict):
        return config["quantization"]
    return config


def _merge_modelopt_configs(
    config_quant: dict[str, Any] | None, hf_quant: dict[str, Any] | None
) -> dict[str, Any] | None:
    """
    Merge the two ModelOpt config sources a checkpoint may carry.

    config.json's ``quantization_config`` can be a lossy projection of
    ModelOpt's own export: entries for modules the projection walked late or
    not at all (e.g. MTP layers) may be dropped from its ``quantized_layers``
    even though the checkpoint carries their quantized tensors, whose
    conversion hf_quant_config.json still describes. When both are
    mixed-precision configs, the config.json entries win per key and the
    hf_quant_config.json entries supply what is missing; the ignore lists are
    unioned.

    :param config_quant: config.json's ``quantization_config``, or None
    :param hf_quant: parsed hf_quant_config.json contents, or None
    :return: the merged ModelOpt config, or the single available source
    """
    if config_quant is None:
        return hf_quant
    if hf_quant is None:
        return config_quant

    base = _unwrap_modelopt_config(config_quant) or {}
    supplement = _unwrap_modelopt_config(hf_quant) or {}
    if not (
        str(base.get("quant_algo", "")).upper() == "MIXED_PRECISION"
        and str(supplement.get("quant_algo", "")).upper() == "MIXED_PRECISION"
    ):
        # only mixed-precision configs carry per-layer entries to merge
        return config_quant

    merged = dict(base)
    quantized_layers = dict(supplement.get("quantized_layers") or {})
    quantized_layers.update(base.get("quantized_layers") or {})
    merged["quantized_layers"] = quantized_layers

    ignore = list(base.get("ignore", base.get("exclude_modules")) or [])
    for entry in supplement.get("ignore", supplement.get("exclude_modules")) or []:
        if entry not in ignore:
            ignore.append(entry)
    merged["ignore"] = ignore

    return merged


class ModelOptConverter(Converter):
    """
    Convert a ModelOpt-quantized checkpoint into compressed-tensors format.

    Generalizes ModelOptNvfp4Converter to every ModelOpt quantization format
    that compressed-tensors can represent. Modules quantized with each ModelOpt
    ``quant_algo`` are converted to the corresponding compressed-tensors
    convention, and the checkpoint's quantization config is rewritten as a
    compressed-tensors QuantizationConfig with one config group per algo:

    - ``NVFP4`` / ``W4A16_NVFP4``: ``weight`` -> ``weight_packed``,
      ``1 / input_scale`` -> ``input_global_scale`` (W4A4 only) and
      ``1 / weight_scale_2`` -> ``weight_global_scale`` (nvfp4-pack-quantized)
    - ``FP8`` / ``FP8_PER_CHANNEL_PER_TOKEN`` / ``MXFP8``: tensor names already
      match compressed-tensors, only the config is rewritten
    - ``FP8_PB_WO``: ``weight_scale`` [out_blocks, 1, in_blocks, 1] ->
      ``weight_scale_inv`` [out_blocks, in_blocks]
    - ``INT8_SQ``: tensor names already match compressed-tensors
    - ``INT8_PB_WO`` / ``INT8_WO``: int8 ``weight`` repacked into int32
      ``weight_packed`` + ``weight_shape`` (pack-quantized)
    - ``MIXED_PRECISION``: each entry of ``quantized_layers`` is converted
      according to its own ``quant_algo``

    Two construction modes are supported. Recommended: build from the
    checkpoint's own ModelOpt quantization config, so that targets, the ignore
    list, the kv cache scheme and per-module algos are all derived from it:

    ```python
    converter = ModelOptConverter.from_pretrained("nvidia/...-NVFP4")
    convert_checkpoint("nvidia/...-NVFP4", save_directory, converter)
    ```

    Alternatively, apply a single quant_algo to manually targeted modules,
    mirroring ModelOptNvfp4Converter:

    ```python
    converter = ModelOptConverter(quant_algo="FP8", targets=["Linear"])
    ```

    :param quantization_config: ModelOpt quantization config dict, i.e. the
        ``quantization_config`` of the checkpoint's config.json (see
        ``from_pretrained``/``from_modelopt_config``) or the contents of its
        hf_quant_config.json. When given, all other arguments must be left at
        their defaults
    :param quant_algo: ModelOpt quant algo applied to every targeted module,
        required when ``quantization_config`` is not given
    :param ignore: modules to leave unquantized, exact names or ``re:`` regexes
    :param targets: modules to convert, exact names, ``re:`` regexes or module
        types; empty targets all (non-ignored) modules
    :param kv_cache_scheme: kv cache quantization args applied to ``k_scale``
        and ``v_scale`` tensors
    :param group_size: ModelOpt group size for algos that use one
        (e.g. ``INT8_PB_WO``), defaulting to the algo's canonical value
    """

    def __init__(
        self,
        quantization_config: dict[str, Any] | None = None,
        quant_algo: str | None = None,
        ignore: Iterable[str] = tuple(),
        targets: Iterable[str] = tuple(),
        kv_cache_scheme: QuantizationArgs | dict[str, Any] | None = None,
        group_size: int | None = None,
    ):
        if quantization_config is not None:
            manual_args = [
                name
                for name, value in (
                    ("quant_algo", quant_algo),
                    ("targets", list(targets)),
                    ("ignore", list(ignore)),
                    ("group_size", group_size),
                )
                if value
            ] + (["kv_cache_scheme"] if kv_cache_scheme is not None else [])
            if manual_args:
                raise ValueError(
                    f"{', '.join(manual_args)} cannot be combined with "
                    "quantization_config; they are derived from it"
                )
            self._init_from_config(quantization_config)
        else:
            if quant_algo is None:
                raise ValueError(
                    "quant_algo is required when quantization_config is not given"
                )
            self._algo = _canonicalize_algo(quant_algo)
            self._group_size = group_size
            self.targets = list(targets)
            self.ignore = list(ignore)
            self.kv_cache_scheme = _coerce_kv_cache_scheme(kv_cache_scheme)
            self._module_algos: dict[str, tuple[str, int | None]] = {}
            self._container_entries: set[str] = set()
            # eagerly validate scheme construction (e.g. NVFP4 group_size)
            _ALGOS[self._algo].scheme(self._group_size)

    @classmethod
    def from_modelopt_config(
        cls, modelopt_config: dict[str, Any]
    ) -> "ModelOptConverter":
        """
        Build a converter from a ModelOpt quantization config dict, i.e. the
        ``quantization_config`` field of the checkpoint's config.json or the
        contents of its hf_quant_config.json.

        :param modelopt_config: ModelOpt quantization config dict
        :return: converter derived from the config
        """
        return cls(quantization_config=modelopt_config)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | os.PathLike,
        model_kwargs: dict | None = None,
    ) -> "ModelOptConverter":
        """
        Read a checkpoint's ModelOpt quantization config and build the
        converter from it. The config is read from config.json's
        ``quantization_config`` (falling back to ``text_config`` and
        ``compression_config``), then from hf_quant_config.json. When both are
        available and ModelOpt-format, their per-layer entries are merged (see
        :func:`_merge_modelopt_configs`), so tensors which only
        hf_quant_config.json describes are still converted.

        :param model_name_or_path: local checkpoint directory or HuggingFace
            model stub
        :param model_kwargs: optional kwargs forwarded to Hugging Face Hub
            lookups (``list_repo_files`` / ``cached_file``) when
            ``model_name_or_path`` is a Hub id, e.g. ``revision``, ``token``.
            Ignored for local paths
        :return: converter derived from the checkpoint's quantization config
        """
        config_quant = None
        hf_quant = None

        if os.path.exists(model_name_or_path):
            config_path = find_config_path(model_name_or_path)
            if config_path is not None:
                quant_config = get_quantization_config(config_path)
                if _is_modelopt_config(quant_config):
                    config_quant = quant_config
            hf_config_path = os.path.join(
                os.path.dirname(config_path or model_name_or_path),
                "hf_quant_config.json",
            )
            if os.path.exists(hf_config_path):
                with open(hf_config_path, "r") as file:
                    hf_quant = json.load(file)
        else:
            hub_kwargs = model_kwargs or {}
            repo_files = list_repo_files(model_name_or_path, **hub_kwargs)
            if CONFIG_NAME in repo_files:
                config_path = cached_file(model_name_or_path, CONFIG_NAME, **hub_kwargs)
                quant_config = get_quantization_config(config_path)
                if _is_modelopt_config(quant_config):
                    config_quant = quant_config
            if "hf_quant_config.json" in repo_files:
                hf_config_path = cached_file(
                    model_name_or_path, "hf_quant_config.json", **hub_kwargs
                )
                with open(hf_config_path, "r") as file:
                    hf_quant = json.load(file)

        if config_quant is None:
            # only hf_quant_config.json describes the checkpoint
            if not _is_modelopt_config(hf_quant):
                raise ValueError(
                    f"Could not find a ModelOpt quantization config for "
                    f"{model_name_or_path}"
                )
            return cls(quantization_config=hf_quant)

        merged = _merge_modelopt_configs(config_quant, hf_quant)
        return cls(quantization_config=merged)

    def _init_from_config(self, quantization_config: dict[str, Any]):
        config = dict(quantization_config)
        if "quantization" in config:
            # hf_quant_config.json layout: {"producer": ..., "quantization": {...}}
            inner = config["quantization"]
            if not isinstance(inner, dict):
                raise ValueError(
                    "Expected 'quantization' to be a dict in the ModelOpt config"
                )
            config = inner

        quant_algo = config.get("quant_algo")
        if not quant_algo:
            raise ValueError("ModelOpt quantization config is missing quant_algo")

        self.kv_cache_scheme = _parse_kv_cache_scheme(config)
        self.ignore = _modelopt_ignore_to_ct(
            config.get("ignore", config.get("exclude_modules", []))
        )
        self._module_algos: dict[str, tuple[str, int | None]] = {}
        self._group_size = None

        if str(quant_algo).upper() == "MIXED_PRECISION":
            quantized_layers = config.get("quantized_layers") or {}
            if not quantized_layers:
                raise ValueError(
                    "MIXED_PRECISION quantization config requires a non-empty "
                    "'quantized_layers' mapping"
                )
            default_group_size = config.get("group_size")
            for module, entry in quantized_layers.items():
                if isinstance(entry, dict):
                    module_algo = entry.get("quant_algo")
                    module_group_size = entry.get("group_size", default_group_size)
                else:
                    module_algo, module_group_size = entry, default_group_size
                if not module_algo:
                    raise ValueError(
                        f"quantized_layers entry for {module!r} is missing quant_algo"
                    )
                self._module_algos[module] = (
                    _canonicalize_algo(module_algo),
                    module_group_size,
                )
            self._algo = None
            self.targets = []
            self._container_entries: set[str] = set()
        else:
            self._algo = _canonicalize_algo(quant_algo)
            self._group_size = config.get("group_size")
            self.targets = self._config_targets(config)
            # eagerly validate scheme construction (e.g. NVFP4 group_size)
            _ALGOS[self._algo].scheme(self._group_size)

    @staticmethod
    def _config_targets(config: dict[str, Any]) -> list[str]:
        """
        Single-algo configs carry compressed-tensors style config_groups whose
        targets list the quantized modules ("Linear", regexes or exact names).
        Default to all Linear layers, like ModelOpt itself.
        """
        targets = []
        for group in (config.get("config_groups") or {}).values():
            targets.extend(group.get("targets") or [])
        return targets or ["Linear"]

    def _assignment_for(self, module_name: str) -> tuple[str, int | None] | None:
        """
        Resolve the (canonical algo, group_size) a module converts with, or
        None when the module is not quantized (ignored or untargeted).

        In mixed-precision mode modules are looked up by exact name in the
        quantized_layers mapping. Entries may also name *container* modules --
        e.g. a fused experts block whose per-expert projections carry the
        quantized tensors -- in which case a module resolves through the
        longest entry which is an anchored prefix of its name; otherwise
        targets/ignore are matched like ModelOptNvfp4Converter.
        """
        if self._module_algos:
            assignment = self._module_algos.get(module_name)
            if assignment is None:
                prefix_hits = [
                    entry
                    for entry in self._module_algos
                    if module_name.startswith(entry + ".")
                ]
                if prefix_hits:
                    # container entry: its per-expert projections resolve
                    # through it. Remember it, so the built quantization
                    # config can target the projections it covers
                    container = max(prefix_hits, key=len)
                    self._container_entries.add(container)
                    assignment = self._module_algos[container]
            return assignment

        if self._is_ignored(module_name):
            return None
        if module_name.endswith("norm"):
            # parity with match_quantizable_tensors: norms are never quantized
            return None
        if (
            self.targets
            and "Linear" not in self.targets
            and not any(match_name(module_name, target) for target in self.targets)
        ):
            return None
        return (self._algo, self._group_size)

    def _is_ignored(self, module_name: str) -> bool:
        return any(match_name(module_name, ign) for ign in self.ignore)

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Map the ModelOpt tensors of each quantized module to the appropriate
        compressed-tensors format. Some tensors require rename, some require
        inversion, some require repacking (see the class docstring for the
        per-algo conventions); unquantized weights are left unchanged.
        """
        for name in list(tensors.keys()):
            if name not in tensors:  # renamed or deleted by an earlier step
                continue
            module_name, _, param_name = name.rpartition(".")

            if param_name in ("k_scale", "v_scale") and (
                self.kv_cache_scheme is not None
            ):
                # the kv cache scheme applies to every attention module,
                # whether or not its weights are quantized; scales are often
                # F32 in ModelOpt, default BF16 in compressed-tensors
                tensors[name] = tensors[name].to(
                    self.kv_cache_scheme.scale_dtype or torch.bfloat16
                )
                continue

            assignment = self._assignment_for(module_name)
            if assignment is None:
                continue
            algo, _ = assignment
            spec = _ALGOS[algo]

            if param_name == "weight":
                self._process_weight(module_name, tensors, spec)
            elif param_name in spec.param_names:
                self._process_qparam(module_name, param_name, tensors, algo)
            elif param_name == "input_scale" and spec.drops_input_scale:
                # W4A16 has no activation quant; genuine exports carry no
                # input_scale, legacy ones are dropped
                del tensors[name]

        return tensors

    def _process_weight(
        self,
        module_name: str,
        tensors: dict[str, torch.Tensor],
        spec: _ModelOptAlgo,
    ):
        name = f"{module_name}.weight"
        weight = tensors[name]

        if spec.packs_int_weight:
            # ModelOpt stores int8 weights unpacked, while pack-quantized
            # stores them densely in int32 alongside their original shape
            if weight.dtype != torch.int8:
                raise ValueError(
                    f"Expected {name} to have dtype torch.int8, got {weight.dtype}"
                )
            if weight.device.type == "meta":
                packed_cols = math.ceil(weight.shape[-1] * 8 / 32)
                packed = torch.empty(
                    (*weight.shape[:-1], packed_cols),
                    dtype=torch.int32,
                    device="meta",
                )
            else:
                packed = pack_to_int32(weight.to(torch.int8), num_bits=8)
            tensors[f"{module_name}.weight_packed"] = packed
            tensors[f"{module_name}.weight_shape"] = torch.tensor(weight.shape)
            del tensors[name]
        elif spec.weight_param == "weight_packed":
            # ModelOpt packs two FP4 values per uint8; other weights (e.g.
            # embeddings and lm_head) remain in their original dtype
            if weight.dtype == torch.uint8:
                tensors[f"{module_name}.weight_packed"] = weight
                del tensors[name]
        # else: weight already matches the compressed-tensors convention

    def _process_qparam(
        self,
        module_name: str,
        param_name: str,
        tensors: dict[str, torch.Tensor],
        algo: str,
    ):
        name = f"{module_name}.{param_name}"

        if param_name == "input_scale":
            if algo == "nvfp4":
                # convert modelopt input_scale x -> 1/x
                # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1070-L1073
                # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1134
                # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L190
                tensors[f"{module_name}.input_global_scale"] = 1 / tensors[name]
                del tensors[name]
            # else: static activation scales already match, e.g. FP8/INT8_SQ
        elif param_name == "weight_scale_2":
            # convert modelopt weight_scale_2 x -> 1/x
            # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/modelopt.py#L1066-L1068
            # https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py#L163-L166
            tensors[f"{module_name}.weight_global_scale"] = 1 / tensors[name]
            del tensors[name]
        elif param_name == "weight_scale":
            if algo == "fp8_block":
                # ModelOpt exports block scales 4-D [out_blocks, 1, in_blocks, 1];
                # compressed-tensors stores them 2-D as weight_scale_inv
                scale = tensors[name]
                if scale.dim() == 4:
                    scale = scale.squeeze(1).squeeze(-1)
                elif scale.dim() != 2:
                    raise ValueError(
                        f"Expected 2-D or 4-D weight_scale for {name}, "
                        f"got shape {tuple(scale.shape)}"
                    )
                tensors[f"{module_name}.weight_scale_inv"] = scale
                del tensors[name]
            # else: weight_scale already matches, e.g. NVFP4/FP8/INT8/MXFP8

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Process, then flag non-ignored qparams without a converted weight and
        targeted weights with an invalid dtype or missing scales. Returns the
        processed tensors so chained converters observe the resulting format.
        """
        tensors = self.process(tensors)

        orphans = []
        for name in tensors:
            module_name, _, param_name = name.rpartition(".")
            if self._is_ignored(module_name):
                continue
            if param_name in _SOURCE_QPARAMS or (
                param_name in ("k_scale", "v_scale") and self.kv_cache_scheme is None
            ):
                assignment = self._assignment_for(module_name)
                # qparams must accompany their module's converted weight;
                # thanks to get_dependencies a missing converted weight in the
                # same job indicates an untargeted or orphan qparam
                converted = assignment is not None and (
                    f"{module_name}.{_ALGOS[assignment[0]].weight_param}" in tensors
                )
                if not converted:
                    orphans.append(name)
        if orphans:
            raise ValueError(
                f"Found {len(orphans)} quantization param(s) without a quantized "
                f"weight, indicating untargeted or orphan qparams: {orphans}"
            )

        for name, tensor in tensors.items():
            module_name, _, param_name = name.rpartition(".")
            assignment = self._assignment_for(module_name)
            if assignment is None:
                continue
            algo, _ = assignment
            spec = _ALGOS[algo]
            if param_name != spec.weight_param:
                continue

            if spec.weight_dtypes and tensor.dtype not in spec.weight_dtypes:
                raise ValueError(
                    f"Expected {name} to have dtype "
                    f"{' or '.join(str(dtype) for dtype in spec.weight_dtypes)}, "
                    f"got {tensor.dtype}"
                )
            for scale_param in spec.required_scales:
                scale_name = f"{module_name}.{scale_param}"
                if scale_name not in tensors:
                    raise ValueError(f"Found {name} without corresponding {scale_name}")

        return tensors

    def get_dependencies(self, weight_name: str) -> set[str]:
        module_name, _, param_name = weight_name.rpartition(".")
        if param_name != "weight":
            return set()

        assignment = self._assignment_for(module_name)
        if assignment is None:
            return set()

        algo, _ = assignment
        spec = _ALGOS[algo]
        deps = {f"{module_name}.{param}" for param in spec.param_names}

        # kv cache scales are deliberately not declared as dependencies: they
        # never require their module's weight for conversion (only a dtype
        # cast, applied wherever the scale itself is scheduled), and ModelOpt
        # exports them only for the layers whose KV cache was calibrated --
        # declaring them required would fail on checkpoints carrying the
        # scheme without per-layer scales

        return deps

    def _build_quant_config(self) -> QuantizationConfig:
        config_groups: dict[str, QuantizationScheme] = {}

        if self._module_algos:
            # mixed precision: one config group per (algo, group_size), with
            # the module names from quantized_layers as targets. Container
            # entries (resolved through a prefix by _assignment_for) cover the
            # per-expert projections beneath them, which in the linearized
            # layout live at {container}.{i}.{proj}, so they target a regex
            # over those projections instead of the container name itself
            modules_by_algo: dict[tuple[str, int | None], list[str]] = defaultdict(list)
            for module, assignment in self._module_algos.items():
                if module in self._container_entries:
                    modules_by_algo[assignment].append(
                        f"re:^{re.escape(module)}"
                        r"\.\d+\.(gate_proj|up_proj|down_proj|gate_up_proj)$"
                    )
                else:
                    modules_by_algo[assignment].append(module)
            for idx, ((algo, group_size), modules) in enumerate(
                sorted(
                    modules_by_algo.items(),
                    key=lambda item: (item[0][0], item[0][1] or 0),
                )
            ):
                spec = _ALGOS[algo]
                config_groups[f"config_group_{idx}"] = QuantizationScheme(
                    targets=sorted(modules),
                    format=spec.format.value,
                    **spec.scheme(group_size),
                )
        else:
            spec = _ALGOS[self._algo]
            config_groups["config_group_0"] = QuantizationScheme(
                targets=self.targets,
                format=spec.format.value,
                **spec.scheme(self._group_size),
            )

        formats = {group.format for group in config_groups.values()}
        format = (
            formats.pop()
            if len(formats) == 1
            else CompressionFormat.mixed_precision.value
        )

        return QuantizationConfig(
            config_groups=config_groups,
            ignore=self.ignore,
            kv_cache_scheme=self.kv_cache_scheme,
            format=format,
            quantization_status=QuantizationStatus.COMPRESSED.value,
        )

    def update_config(
        self, config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        quant_config = self._build_quant_config()
        if config is not None:
            config.merge(quant_config)
            return config
        return quant_config
