# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
import warnings
from collections import defaultdict
from enum import Enum

from compressed_tensors.config import CompressionFormat, ExpertSparsityConfig
from compressed_tensors.quantization.quant_args import DynamicType, QuantizationArgs
from compressed_tensors.quantization.quant_scheme import (
    QuantizationScheme,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.utils import is_module_quantized
from compressed_tensors.utils import find_unique_name
from compressed_tensors.utils.consolidate import consolidate_module_names
from compressed_tensors.utils.match import match_name
from pydantic import BaseModel, ConfigDict, Field
from torch.nn import Module


__all__ = [
    "QuantizationStatus",
    "QuantizationConfig",
    "LIFECYCLE_ORDER",
    "DEFAULT_QUANTIZATION_METHOD",
    "DEFAULT_QUANTIZATION_FORMAT",
]


def _map_to_checkpoint_names(model: Module, ignore_list: list[str]) -> list[str]:
    """Translate ignore list entries from HF module names to checkpoint names.

    Transformers v5 may rename weight keys on load (e.g. vision_embedder ->
    embed_vision).  The ignore list is built from ``model.named_modules()``
    which uses HF names, but safetensors keys use checkpoint names.  This
    applies the same reverse mapping that ``save_pretrained`` uses for weights.
    """
    weight_conversions = getattr(model, "_weight_conversions", None)
    if not weight_conversions:
        return ignore_list

    inverted = [conv.reverse_transform() for conv in reversed(weight_conversions)]

    result = []
    for name in ignore_list:
        # Some conversions match a complete weight key or a trailing dot.
        # This assumes Linear-style `.weight` parameters; conversions specific
        # to other parameter names are not handled.
        name += ".weight"
        for rev in inverted:
            renamed, matched = rev.rename_source_key(name)
            if matched is not None:
                name = renamed
        result.append(name.removesuffix(".weight"))

    return result


def copy_formats_after_compression(config: "QuantizationConfig", model: Module) -> None:
    """Write each module's compression format onto the saved scheme.

    ``compress_module`` sets ``format`` on the scheme object stored on the
    module. ``["Linear"]`` is saved as that same object, so the format is
    already in the config. A module path or a regex is saved as a copy made
    before compression, because the module keeps the targets the user wrote.
    That copy's ``format`` is still unset. This copies the format
    compression already chose. It does not choose a new one.
    """
    from compressed_tensors.utils.match import match_name

    for name, module in model.named_modules():
        live = getattr(module, "quantization_scheme", None)
        if not name or live is None or getattr(live, "format", None) is None:
            continue
        checkpoint_names = _map_to_checkpoint_names(model, [name])
        checkpoint_name = checkpoint_names[0] if checkpoint_names else name
        for group in config.config_groups.values():
            if group is live:
                break
            if any(
                match_name(checkpoint_name, target) or match_name(name, target)
                for target in group.targets
            ):
                group.format = live.format
                break


class _SchemeKey:
    """Dict key for one scheme object.

    ``QuantizationScheme`` is not hashable, and compression assigns
    ``format`` on that object later. Modules that share a scheme object
    share a group.
    """

    def __init__(self, scheme: QuantizationScheme):
        self.scheme = scheme

    def __hash__(self) -> int:
        return id(self.scheme)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SchemeKey) and self.scheme is other.scheme


def _targets_are_class_names(targets: list[str]) -> bool:
    """True when every target is a class name, such as ``Linear``.

    A class name has no ``.`` and is not a ``re:`` pattern. Those targets
    are already short. Folding every matched module into a regex only
    makes the saved config longer.
    """
    return bool(targets) and all(
        target and "." not in target and not target.startswith("re:")
        for target in targets
    )


class QuantizationStatus(str, Enum):
    """
    Enum storing the different states a quantized layer can be in

    - Initialized: Quantization parameters are initialized to empty values
    - Calibration: Quantization parameters are being calibrated, observers are attached
    - Frozen: Quantization parameters are fully calibrated, observers are removed
    - Compressed: All parameters are quantized to target dtype. If the weight param is
        no longer applicable (i.e. if "weight" has been converted to "weight_packed"),
        the weight param is pruned from the module.
    - Decompressed: Parameters are converted back into frozen state. Quantization params
        remain on the module so that the module can be compressed, but params are
        pruned if they are no longer applicable or needed for compression
        (i.e. if "weight_packed" has been converted to "weight").
        Additionally, weight qdq is skipped during forward passes for better performance
    """

    INITIALIZED = "initialized"
    CALIBRATION = "calibration"
    FROZEN = "frozen"
    COMPRESSED = "compressed"
    DECOMPRESSED = "decompressed"

    @classmethod
    def lifecycle_order(cls) -> list["QuantizationStatus"]:
        """
        :return: list of correct quantization lifecycle order
        """
        return

    def __ge__(self, other):
        if other is None:
            return True
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) >= LIFECYCLE_ORDER.index(other)

    def __gt__(self, other):
        if other is None:
            return True
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) > LIFECYCLE_ORDER.index(other)

    def __lt__(self, other):
        if other is None:
            return False
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) < LIFECYCLE_ORDER.index(other)

    def __le__(self, other):
        if other is None:
            return False
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return LIFECYCLE_ORDER.index(self) <= LIFECYCLE_ORDER.index(other)


LIFECYCLE_ORDER = [
    QuantizationStatus.INITIALIZED,
    QuantizationStatus.CALIBRATION,
    QuantizationStatus.FROZEN,
    QuantizationStatus.COMPRESSED,
    QuantizationStatus.DECOMPRESSED,
]

DEFAULT_QUANTIZATION_METHOD = "compressed-tensors"
DEFAULT_QUANTIZATION_FORMAT = "fakequant"


class QuantizationConfig(BaseModel):
    """
    Full configuration specifying how a model is quantized. Each quantized layer is
    mapped to a QuantizationScheme in config_groups.

    :param config_groups: dict of QuantizationSchemes specifying the quantization
    settings for each quantized layer. A group could also be a reference to
    a predefined scheme name, mapped to a list of its target layers/classes
    :param quant_method: a constant used to differentiate compressed-tensors
    quantization from other quantization configs
    :param format: specifies how the quantized model is stored on disk
    :quantization_status: specifies the current status of all quantized layers. It is
        assumed all layers are in the same state.
    :param kv_cache_scheme: optional QuantizationArgs, that specify the
        quantization of the kv cache. If None, kv cache is not quantized.
        When applying kv cache quantization to transformer AutoModelForCausalLM,
        the kv_cache_scheme gets converted into a QuantizationScheme that:
            - targets the `q_proj` and `k_proj` modules of the model. The outputs
              of those modules are the keys and values that might be cached
            - quantizes the outputs of the aforementioned layers, so that
              keys and values are compressed before storing them in the cache
        There is an explicit assumption that the model contains modules with
        `k_proj` and `v_proj` in their names. If this is not the case
        and kv_cache_scheme != None, the quantization of kv cache will fail
    :global_compression_ratio: optional informational config to report the model
        compression ratio achieved by the quantization config
    :ignore: optional list of layers to ignore from config_groups. Layers in this list
        are not quantized even if they match up with a target in config_groups
    :param moe_sparsity_config: optional ExpertSparsityConfig describing per-layer
        MoE expert sparsity. If None, the model has no expert sparsity.
    """

    config_groups: dict[str, QuantizationScheme | list[str]]
    quant_method: str = DEFAULT_QUANTIZATION_METHOD
    kv_cache_scheme: QuantizationArgs | None = None
    format: str = DEFAULT_QUANTIZATION_FORMAT
    quantization_status: QuantizationStatus = QuantizationStatus.INITIALIZED
    global_compression_ratio: float | None = None
    ignore: list[str] | None = Field(default_factory=list)
    moe_sparsity_config: ExpertSparsityConfig | None = None

    def model_post_init(self, __context):
        """
        updates any quantization schemes defined as presets to be fully loaded
        schemes
        """
        for group_name, targets_or_scheme in self.config_groups.items():
            if isinstance(targets_or_scheme, QuantizationScheme):
                continue  # scheme already defined
            self.config_groups[group_name] = preset_name_to_scheme(
                name=group_name,
                targets=targets_or_scheme,
            )

    def to_dict(self):
        # for compatibility with HFQuantizer
        return self.model_dump()

    @staticmethod
    def from_pretrained(
        model: Module, format: str | list | None = None
    ) -> "QuantizationConfig | None":
        """
        Converts a model into its associated QuantizationConfig based on the
        QuantizationScheme attached to each quantized module

        :param model: model to calculate quantization scheme of
        :return: filled out QuantizationScheme for the input model
        """
        from compressed_tensors.modeling import IMPL_ATTR
        from compressed_tensors.quantization.lifecycle.initialize import (
            is_cached_attention_module,
        )

        # One scheme object is one group. Insertion order is the group order.
        grouped: dict[_SchemeKey, list[str]] = defaultdict(list)

        # use any status from modules (in practice, use the last module)
        model_status = None

        # set of all quantized types
        # this is later used to create the ignore list
        quantization_type_names: set[str] = set()

        # maps types to names which are not quantized
        # this is later used to create the ignore list
        ignore: dict[str, list[str]] = defaultdict(list)

        # this keeps track of any kvcache schemes
        kv_cache_scheme: QuantizationArgs | None = None

        for name, submodule in model.named_modules():
            layer_type: str = get_vllm_module_type(type(submodule).__name__)
            is_cached_attention = is_cached_attention_module(submodule)

            # add config group if quantized non-attention or attention quant
            has_config_group = is_module_quantized(submodule) and (
                not is_cached_attention or hasattr(submodule, IMPL_ATTR)
            )
            # only add kvcache if quant attention (which always implies kvcache)
            has_kv_cache = is_module_quantized(submodule) and is_cached_attention

            if has_config_group:
                # add to running set of schemes/layer_type_names
                model_status = getattr(submodule, "quantization_status", model_status)
                quantization_type_names.add(layer_type)
                scheme = submodule.quantization_scheme
                module_names = grouped[_SchemeKey(scheme)]
                if name:
                    module_names.append(name)

            if has_kv_cache:
                model_status = getattr(submodule, "quantization_status", model_status)
                kv_cache_scheme = submodule.quantization_scheme.input_activations

            if not has_config_group:
                # add non-quantized layers to the ignore list
                if layer_type not in ignore:
                    ignore[layer_type] = []
                ignore[layer_type].append(name)

        if len(grouped) == 0 and kv_cache_scheme is None:  # No quantized layers
            return None

        # create ignore list, only include layers whose class has ever been targeted
        consolidated_ignore = []
        for layer_type, ignore_names in ignore.items():
            if layer_type in quantization_type_names:
                # specific layers of a quantized type are ignored
                consolidated_ignore += ignore_names
            # else we leave it off the ignore list, doesn't fall under any of the
            # existing quantization schemes so it won't be quantized

        consolidated_ignore = _map_to_checkpoint_names(model, consolidated_ignore)
        # Ignores stay concrete. Older vLLM rewrites those names and leaves
        # "re:" entries unchanged. Ignore regexes wait until that vLLM is
        # the one in use.

        # Class-name targets such as "Linear" stay as written. Replacing
        # them lists every module and the config gets longer for no benefit.
        # Module paths and regexes are checkpoint names, then folded.
        # The module keeps its own scheme. A copied scheme with a checkpoint
        # regex in its place leaves qparams behind after decompression.
        config_groups = {}
        for idx, (key, module_names) in enumerate(grouped.items()):
            scheme = key.scheme
            group_name = "group_" + str(idx)
            if _targets_are_class_names(scheme.targets):
                config_groups[group_name] = scheme
                continue
            checkpoint_names = _map_to_checkpoint_names(model, module_names)
            config_groups[group_name] = scheme.model_copy(
                update={"targets": consolidate_module_names(checkpoint_names)}
            )

        # infer format
        if format is None:
            if model_status == QuantizationStatus.COMPRESSED:
                format = CompressionFormat.int_quantized.value
            else:
                format = CompressionFormat.dense.value
        elif isinstance(format, list):
            format = (
                CompressionFormat.mixed_precision.value
                if len(format) > 1
                else format[0]
            )

        return QuantizationConfig(
            config_groups=config_groups,
            quantization_status=model_status,
            kv_cache_scheme=kv_cache_scheme,
            global_compression_ratio=None,
            format=format,
            ignore=consolidated_ignore,
        )

    def requires_calibration_data(self):
        if self.kv_cache_scheme is not None:
            return True

        for _, scheme in self.config_groups.items():
            if scheme.weights is not None:
                if scheme.weights.observer == "imatrix_mse":
                    return True
            if scheme.input_activations is not None:
                if scheme.input_activations.dynamic in (False, DynamicType.LOCAL):
                    return True
            if scheme.output_activations is not None:
                if not scheme.output_activations.dynamic:
                    return True

        return False

    def merge(self, config: "QuantizationConfig") -> None:
        """
        Merge another QuantizationConfig into self, modifying in-place. The current
        quant config (self) will take precedence over the second quant config,
        i.e. the config groups will be appended rather than inserted before. Python's
        json stdlib and pydantic both respect order when serializing/deserializing.

        Because QuantizationConfig has a global ignore list while targets are scoped to
        a given scheme, merging is not a straightforward task. A warning will be raised
        when this function is called to indicate that invalid quantization configs are
        possible when using this method, particularly when complex ignore lists are
        used. For best results, use complex targets lists over complex ignore lists.

        If a new quant format is added, the global format will be set to
        "mixed-precision".

        The QuantizationStatus will be set to whichever is largest.

        Excluding regexes, any names in the ignore list that are targeted in the new
        quant config will be pruned.

        :param config: QuantizationConfig to merge into self.
        """
        warnings.warn(
            "Attempting to merge quantization configs. This is not a straightforward "
            "task and can lead to quantization configs that fail to load. For best "
            "results, use complex targets lists instead of complex ingore lists"
        )

        pruned_ignore_list = []
        for ign in self.ignore:
            if ign.startswith("re:"):
                pruned_ignore_list.append(ign)
                continue
            if any(
                match_name(ign, target)
                for scheme in config.config_groups.values()
                for target in scheme.targets
            ):
                continue
            pruned_ignore_list.append(ign)
        self.ignore = pruned_ignore_list

        for scheme_name, scheme in config.config_groups.items():
            new_scheme_name = find_unique_name(scheme_name, self.config_groups.keys())

            self.config_groups[new_scheme_name] = scheme

        unique_formats = set(scheme.format for scheme in self.config_groups.values())
        self.format = (
            next(iter(unique_formats))
            if len(unique_formats) == 1
            else CompressionFormat.mixed_precision.value
        )

        if config.quantization_status > self.quantization_status:
            self.quantization_status = config.quantization_status

    # TODO set `extra="forbid"` when upstream transformers is compatible
    model_config = ConfigDict(extra="ignore")


def get_vllm_module_type(module_type: str) -> str:
    """
    Returns a string representing the module type used when loading in vLLM.
    This is typically going to be the same as the `torch.nn.Module` type,
    however specific cases like MoE gate layers need to be treated like "Linear"
    layers for the purposes of config matching.
    """
    if "ExpertMLP" not in module_type and (
        "Router" in module_type
        or "Gating" in module_type
        or re.search(r"Gate(?![a-z])", module_type)
    ):
        module_type = "Linear"

    return module_type
