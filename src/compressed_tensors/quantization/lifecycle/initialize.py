# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import math
import numbers

import torch
from compressed_tensors.modeling import (
    IMPL_ATTR,
    KV_CACHE_ATTR,
    QuantizedAttentionImpl,
    QuantizedKVCache,
)
from compressed_tensors.offload import disable_onloading, unwrap_offload_forward
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationMetadata,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle.forward import set_forward_quantized
from compressed_tensors.quantization.utils import strategy_cdiv
from compressed_tensors.utils import (
    get_execution_device,
    get_head_dim,
    get_num_attn_heads,
    get_num_kv_heads,
)
from compressed_tensors.utils.helpers import deprecated
from loguru import logger
from torch.nn import Module, Parameter
from transformers import PretrainedConfig


__all__ = [
    "initialize_module_for_quantization",
    "is_attention_module",
    "is_cached_attention_module",
    "initialize_qparams",
    "initialize_attn_qparams",
]

_ATTENTION_PROJECTION_NAMES = (
    "k_proj",
    "v_proj",
    "qkv_proj",
    "kv_b_proj",
    "q_a_proj",
    "kv_proj",
    "q",
)
_KV_CACHE_PROJECTION_NAMES = ("k_proj", "v_proj", "qkv_proj", "kv_b_proj")


def initialize_module_for_quantization(
    module: Module,
    scheme: QuantizationScheme | None = None,
    force_zero_point: bool = True,
):
    """
    Attaches appropriate scales, zero points, and observers to a layer
    given its target quantization scheme.

    Previously initialized scales and zero points will be removed from
    module if they no longer apply to the scheme

    :param module: module to set for calibration
    :param scheme: scheme to use for quantization. if None is provided,
        will attempt to use scheme stored in the module under `quantization_scheme`,
        if not provided, the layer will be skipped
    :param force_zero_point: whether to force initialization of a zero point for
        symmetric quantization
    """
    from compressed_tensors.linear.compressed_linear import CompressedLinear  # circ dep

    scheme = scheme or getattr(module, "quantization_scheme", None)
    if scheme is None:
        return

    QuantizationMetadata.clear_all_qparams(module)

    if is_cached_attention_module(module):
        initialize_attn_qparams(module, scheme, force_zero_point)

    elif isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        with disable_onloading():
            weight = module.weight

        if scheme.input_activations is not None:
            initialize_qparams(
                module,
                "input",
                scheme.input_activations,
                observed_shape=weight.shape[-1:],
                observed_dtype=weight.dtype,
                force_zero_point=force_zero_point,
            )

        if scheme.weights is not None:
            initialize_qparams(
                module,
                "weight",
                scheme.weights,
                observed_shape=weight.shape,
                observed_dtype=weight.dtype,
                force_zero_point=force_zero_point,
            )

        if scheme.output_activations is not None:
            initialize_qparams(
                module,
                "output",
                scheme.output_activations,
                observed_shape=weight.shape[:-1],
                observed_dtype=weight.dtype,
                force_zero_point=force_zero_point,
            )

        # CompressedLinear has its own forward method that handles decompression
        # Don't override it with the quantized forward
        if not isinstance(module, CompressedLinear):
            with unwrap_offload_forward(module):
                set_forward_quantized(module)

    else:
        raise ValueError(f"Quantization of module type {type(module)} is not supported")

    module.quantization_scheme = scheme
    module.quantization_status = QuantizationStatus.INITIALIZED


def _is_attention_module(module: Module) -> bool:
    class_name = type(module).__name__.lower()
    if "attention" not in class_name:
        return False

    with disable_onloading():
        return any(hasattr(module, name) for name in _ATTENTION_PROJECTION_NAMES)


@deprecated("is_cached_attention_module")
def is_attention_module(module: Module) -> bool:
    return _is_attention_module(module)


def is_cached_attention_module(module: Module) -> bool:
    if not _is_attention_module(module):
        return False

    with disable_onloading():
        if not any(hasattr(module, name) for name in _KV_CACHE_PROJECTION_NAMES):
            return False

    try:
        parameters = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        logger.warning(
            "Unable to inspect an attention module's forward signature; "
            "skipping KV cache quantization for uninspectable modules",
            log_once=True,
        )
        return False

    return "past_key_value" in parameters or "past_key_values" in parameters


def initialize_qparams(
    module: Module,
    base_name: str,
    quantization_args: QuantizationArgs,
    observed_shape: tuple[int | None, ...],
    observed_dtype: torch.dtype,
    force_zero_point: bool = True,
):
    """
    Initialize quantization parameters for a given basename according to the passed
    quantization args. The shape and dtype of the observed weight/activation must also
    be provided.

    Scales will always be initialized. Global scales are initialized depending on args.
    Zero points will be initialized if not symmetric or if `force_zero_point` is True.

    :param module: module to register qparams to
    :param base_name: base name of qparams, for example "input", "weight", "k", "v"
    :param quantization_args: arguments for quantization
    :param observed_shape: last (right-most) known dimensions of the observed weight/act
    :param observed_dtype: dtype of the observed weight/actt
    :param force_zero_point: force the zero_point parameter to be initialized
    """
    strategy = quantization_args.strategy
    dynamic = quantization_args.dynamic
    device = get_execution_device(module)  # avoid performing intialization ops on cpu

    # Skip all intialization for fully dynamic quantization
    if dynamic is True:
        return

    # 0. Create global scale for tensor-group quantization
    if strategy == QuantizationStrategy.TENSOR_GROUP:
        init_global_scale = Parameter(
            torch.empty(1, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        module.register_parameter(f"{base_name}_global_scale", init_global_scale)

    # Skip scale/zp initialization for locally dynamic quantization
    if dynamic == DynamicType.LOCAL:
        return

    # 1. Infer expected scale/zp shape
    if strategy == QuantizationStrategy.TENSOR:
        expected_shape = (1,)

    elif strategy == QuantizationStrategy.TOKEN:
        raise ValueError("Cannot perform static token quantization")

    elif strategy == QuantizationStrategy.CHANNEL:
        if len(observed_shape) < 2:
            raise ValueError("Channel quant requires at least 2 observed dimensions")

        expected_shape = (observed_shape[-2], 1)

    elif strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        assert quantization_args.group_size is not None
        if len(observed_shape) < 1:
            raise ValueError("Group quant requires at least 1 observed dimension")

        group_size = quantization_args.group_size
        num_groups = strategy_cdiv(observed_shape[-1], group_size, strategy)
        expected_shape = (*observed_shape[:-1], num_groups)

    elif strategy == QuantizationStrategy.BLOCK:
        assert quantization_args.block_structure is not None
        if len(observed_shape) < 2:
            raise ValueError("Block quant requires at least 2 observed dimensions")

        block_structure = quantization_args.block_structure

        # NOTE: vllm kernels for block-quantization do not require
        # num_rows to be evenly divisible by block_structure[-2],
        # but num_cols does need to be evenly divisible by block_structure[-1]
        num_rows = math.ceil(observed_shape[-2] / block_structure[-2])
        num_cols = strategy_cdiv(observed_shape[-1], block_structure[-1], strategy)
        expected_shape = (num_rows, num_cols)

    elif strategy == QuantizationStrategy.ATTN_HEAD:
        # (batch_size, num_attention_heads, seq_len, head_dim)
        if len(observed_shape) < 3:
            raise ValueError("Attention quant requires at least 3 observed dimensions")

        expected_shape = (observed_shape[-3], 1, 1)

    else:
        assert False, f"Unknown strategy {strategy}"

    # 2. Identify quantization scale and zp dtype
    scale_dtype = observed_dtype
    if scale_dtype not in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ]:
        scale_dtype = torch.float16

    # 3. Initializes scale/zp for the module
    init_scale = Parameter(
        torch.empty(expected_shape, dtype=scale_dtype, device=device),
        requires_grad=False,
    )
    module.register_parameter(f"{base_name}_scale", init_scale)

    if force_zero_point or not quantization_args.symmetric:
        init_zero_point = Parameter(
            torch.zeros(
                expected_shape, device=device, dtype=quantization_args.zp_dtype
            ),
            requires_grad=False,
        )
        module.register_parameter(f"{base_name}_zero_point", init_zero_point)


def initialize_attn_qparams(
    module: Module, scheme: QuantizationScheme, force_zero_point: bool
):
    """Initlaize k_scale, v_scale for self_attn"""

    impl: QuantizedAttentionImpl | None = getattr(module, IMPL_ATTR, None)
    kv_cache: QuantizedKVCache | None = getattr(module, KV_CACHE_ATTR, None)

    if impl is None and kv_cache is None:
        raise ValueError(
            f"Attention module has quantization scheme but no {IMPL_ATTR} "
            f"or {KV_CACHE_ATTR} attributes. Please ensure that these "
            "attributes are initialized using `apply_quantization_config`."
        )

    _validate_attention_scheme(scheme)

    # extract shapes from the config of this attention module's decoder layer
    is_heterogeneous = getattr(kv_cache.config, "is_heterogeneous", False)
    config = _get_layer_config(kv_cache.config, module)
    num_attn_heads = get_num_attn_heads(config)
    num_kv_heads = get_num_kv_heads(config)
    head_dim = get_head_dim(config)
    _validate_num_kv_heads(
        module, scheme.input_activations, num_kv_heads, strict=is_heterogeneous
    )

    # (batch_size, num_heads, slen, head_dim)
    q_observed_shape = (num_attn_heads, None, head_dim)
    kv_observed_shape = (num_kv_heads, None, head_dim)
    observed_dtype = next(module.parameters()).dtype

    if impl is not None:
        initialize_qparams(
            module,
            "q",
            scheme.input_activations,
            observed_shape=q_observed_shape,
            observed_dtype=observed_dtype,
            force_zero_point=force_zero_point,
        )

    if kv_cache is not None:
        initialize_qparams(
            module,
            "k",
            scheme.input_activations,
            observed_shape=kv_observed_shape,
            observed_dtype=observed_dtype,
            force_zero_point=force_zero_point,
        )
        initialize_qparams(
            module,
            "v",
            scheme.input_activations,
            observed_shape=kv_observed_shape,
            observed_dtype=observed_dtype,
            force_zero_point=force_zero_point,
        )


def _validate_attention_scheme(scheme: QuantizationScheme):
    if scheme.weights is not None:
        raise ValueError(
            "Cannot apply weight quantization to attention. "
            "Instead, target the (q|k|v)_proj submodule layers of attention"
        )

    if scheme.input_activations is None:
        raise ValueError(
            "Cannot apply attention quantization without specifying input activations"
        )

    if scheme.output_activations is not None:
        raise ValueError("Cannot apply output quantization to attention")

    args = scheme.input_activations
    if args.dynamic is False and args.strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        strategy = QuantizationStrategy(args.strategy).value
        raise ValueError(
            f"Static {strategy} quantization is not supported for attention or the "
            "kv cache. Use the tensor or attn_head strategy, or dynamic quantization"
        )


def _get_layer_config(config: PretrainedConfig, module: Module) -> PretrainedConfig:
    """
    Get the config of the decoder layer which owns an attention module

    Heterogeneous configs (transformers>=5.14, for example gemma-4) vary attributes
    such as `head_dim` and `num_key_value_heads` per layer, and transformers raises
    when they are read from the global config. Read them from the config of the
    module's own layer instead, selected by the module's `layer_idx`.

    :param config: decoder config of the model
    :param module: attention module
    :return: config of the module's layer, or `config` if it is homogeneous
    """
    if not getattr(config, "is_heterogeneous", False):
        return config

    layer_idx = getattr(module, "layer_idx", None)
    num_layers = len(config.per_layer_config)
    is_index = isinstance(layer_idx, numbers.Integral) and not isinstance(
        layer_idx, bool
    )
    if not is_index or not 0 <= layer_idx < num_layers:
        raise ValueError(
            f"Cannot initialize attention quantization for {type(module).__name__}: "
            f"the model config varies {sorted(config.per_layer_attributes)} per "
            f"layer, but the module's layer_idx ({layer_idx!r}) does not select one "
            f"of its {num_layers} layers"
        )

    return config.per_layer_config[int(layer_idx)]


def _validate_num_kv_heads(
    module: Module, args: QuantizationArgs, num_kv_heads: int, strict: bool
):
    """
    Check that the config's number of key/value heads matches the attention module
    before shaping per-head kv cache scales by it.

    For per-layer configs a mismatch is an error. For other configs it only logs a
    warning and keeps the config-shaped scales, as before this check existed: some
    models initialize today despite a mismatch, for example the Idefics2 perceiver
    attention, which is hooked with the decoder config, while others fail later when
    calibrated or run, for example gemma-4 26B/31B on transformers<5.15.

    :param module: attention module
    :param args: input activation quantization args of the attention module
    :param num_kv_heads: number of key/value heads read from the config
    :param strict: raise instead of warning when the counts disagree
    """
    if args.strategy != QuantizationStrategy.ATTN_HEAD or args.dynamic is not False:
        return

    head_dim = getattr(module, "head_dim", None)
    out_features = getattr(getattr(module, "k_proj", None), "out_features", None)
    if not isinstance(head_dim, int) or not isinstance(out_features, int):
        return
    if head_dim <= 0 or out_features % head_dim != 0:
        return

    module_kv_heads = out_features // head_dim
    if module_kv_heads != num_kv_heads:
        message = (
            f"attn_head quantization of {type(module).__name__} shapes its kv cache "
            f"scales for {num_kv_heads} key/value heads from the config, but its "
            f"k_proj projects to {module_kv_heads} heads of size {head_dim}"
        )
        if strict:
            raise ValueError(message)
        logger.warning(
            f"{message}. The scales keep the config's shape, so this module may "
            "fail to calibrate or run; the tensor strategy does not depend on the "
            "number of heads",
            log_once=True,
        )
