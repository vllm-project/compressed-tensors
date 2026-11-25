# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import re
from collections import OrderedDict
from typing import Optional

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
from compressed_tensors.utils.safetensors_load import get_checkpoint_files
from safetensors import safe_open


__all__ = ["GemmaConverter"]


# Gemma's static per-tensor activation scales, renamed to CT input_scale/output_scale.
_IN_ACT_SUFFIX = ".input_activation_scale"
_OUT_ACT_SUFFIX = ".output_activation_scale"
_ACT_SUFFIXES = (_IN_ACT_SUFFIX, _OUT_ACT_SUFFIX)

# KV-cache scales are out of scope for weight-only conversion; always dropped.
_KV_DROP_SUFFIXES = (".k_cache_scale", ".v_cache_scale")

# safetensors header dtype strings for the packed weight storage.
_QUANT_STORAGE_DTYPES = {"U8", "I8"}


def _activation_args() -> QuantizationArgs:
    # gemma's static per-tensor `clamp(round(x/scale), -128, 127) * scale` is exactly
    # CT int8 symmetric QDQ. Execution stays weight-only (activation fake-quantized in
    # float around the sub-byte weight matmul).
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        symmetric=True,
        strategy=QuantizationStrategy.TENSOR,
        dynamic=False,
    )


def _unpack_gemma_uint8(packed: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Unpack a gemma uint8-packed weight into signed int8 values.

    Stores ``8 // num_bits`` symmetric values per byte, low-index value in the low
    bits, each offset by ``2 ** (num_bits - 1)``. The contiguous column ordering
    matches :func:`pack_to_int32`, so re-packing the result is value-exact.
    """
    packed = packed.to(torch.uint8)
    pack_factor = 8 // num_bits
    mask = (1 << num_bits) - 1
    offset = 1 << (num_bits - 1)

    values = [
        (((packed >> (i * num_bits)) & mask).to(torch.int16) - offset)
        for i in range(pack_factor)
    ]
    stacked = torch.stack(values, dim=-1)
    out = stacked.reshape(*packed.shape[:-1], packed.shape[-1] * pack_factor)
    return out.to(torch.int8)


def _dequantize_dense(
    weight: torch.Tensor, scale: torch.Tensor, plan: "_ModulePlan"
) -> torch.Tensor:
    """Reconstruct a dense bf16 ``[out, in]`` weight from a quantized gemma weight."""
    if weight.dtype == torch.int8:
        vals = weight.to(torch.float32)
    else:
        vals = _unpack_gemma_uint8(weight, plan.num_bits).to(torch.float32)
    scale = scale.to(torch.float32)
    out_f, in_f = plan.unpacked_shape
    if plan.group_size:
        ng = in_f // plan.group_size
        dense = (vals.view(out_f, ng, plan.group_size) * scale.view(out_f, ng, 1)).view(
            out_f, in_f
        )
    else:
        dense = vals * scale  # scale [out, 1] broadcasts over the input dim
    return dense.to(torch.bfloat16)


class _ModulePlan:
    """Per-module conversion plan, precomputed from checkpoint metadata."""

    __slots__ = (
        "module",
        "weight_key",
        "scale_key",
        "num_bits",
        "strategy",
        "group_size",
        "unpacked_shape",
        "is_embedding",
        "in_scale_key",
        "out_scale_key",
        "dequantize",
    )

    def __init__(
        self,
        module: str,
        weight_key: str,
        scale_key: str,
        num_bits: int,
        strategy: QuantizationStrategy,
        group_size: Optional[int],
        unpacked_shape: list[int],
        is_embedding: bool,
        in_scale_key: Optional[str] = None,
        out_scale_key: Optional[str] = None,
        dequantize: bool = False,
    ):
        self.module = module
        self.weight_key = weight_key
        self.scale_key = scale_key
        self.num_bits = num_bits
        self.strategy = strategy
        self.group_size = group_size
        self.unpacked_shape = unpacked_shape
        self.is_embedding = is_embedding
        # If True, dequantize to a dense bf16 `weight` instead of compressing
        # (e.g. for modules consumed by non-quant-aware layers like HF towers).
        self.dequantize = dequantize
        # Source gemma activation-scale tensor keys, or None if the scale is
        # absent / uncalibrated (zero). Renamed to CT `input_scale`/`output_scale`.
        self.in_scale_key = in_scale_key
        self.out_scale_key = out_scale_key


class GemmaConverter(Converter):
    """Convert gemma "mobile" QAT checkpoints to compressed-tensors pack-quantized.

    The source (e.g. ``google/gemma-4-E*B-it-qat-mobile-transformers``) stores
    symmetric INT weights with per-module bit-widths from a
    ``quantization_config.module_quant_configs`` regex map: linear ``<module>.weight``
    as uint8 (2/4-bit packed) or int8 (8-bit) with per-channel
    ``<module>.weight_scale``, and embeddings as ``<module>.embedding_quantized`` +
    ``<module>.embedding_scale``.

    Each quantized weight is unpacked to signed int8 and re-packed into CT's int32
    ``weight_packed`` (value-exact), and ``quantization_config`` is rewritten for vLLM's
    compressed-tensors loader. Gemma's static per-tensor INT8 activation scales are
    preserved as CT ``input_scale`` / ``output_scale``; uncalibrated (zero) and KV-cache
    scales are dropped. Pass ``include_activation_quant=False`` for a weight-only
    checkpoint, or ``dequantize_targets`` to emit some modules as dense bf16 instead.
    """

    def __init__(
        self,
        plans: "OrderedDict[str, _ModulePlan]",
        ignore: list[str],
        scale_dtype: torch.dtype = torch.bfloat16,
        act_scale_dtype: torch.dtype = torch.float32,
    ):
        self.plans = plans
        self.ignore = ignore
        self.scale_dtype = scale_dtype
        self.act_scale_dtype = act_scale_dtype
        # map of weight_key -> plan for fast lookup in process()
        self._by_weight_key = {p.weight_key: p for p in plans.values()}
        self._scale_keys = {p.scale_key for p in plans.values()}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        include_activation_quant: bool = True,
        dequantize_targets: Optional[list[str]] = None,
    ) -> "GemmaConverter":
        files = get_checkpoint_files(model_name_or_path)

        config_path = files.get("config.json")
        if config_path is None:
            raise ValueError(f"Could not find config.json for {model_name_or_path!r}")
        with open(config_path) as f:
            config = json.load(f)

        q_cfg = config.get("quantization_config")
        if q_cfg is None:
            raise ValueError("Model config does not contain quantization_config")
        if q_cfg.get("quant_method") != "gemma":
            raise ValueError(
                "Model config quant_method is not 'gemma', got "
                f"{q_cfg.get('quant_method')!r}"
            )

        module_quant_configs = q_cfg.get("module_quant_configs")
        if not module_quant_configs:
            raise ValueError(
                "gemma quantization_config is missing 'module_quant_configs'"
            )
        # Preserve order: bit-width resolution is first-match (e.g. layers 0-14 mlp
        # @ 4-bit shadows the all-layers mlp @ 2-bit rule).
        module_quant_configs = OrderedDict(module_quant_configs.items())

        ignore = list(q_cfg.get("modules_to_not_convert") or [])

        shard_paths = [
            path for name, path in files.items() if name.endswith(".safetensors")
        ]
        if not shard_paths:
            raise ValueError(f"No safetensors files found for {model_name_or_path!r}")

        plans = cls._build_plans(
            shard_paths,
            module_quant_configs,
            include_activation_quant,
            dequantize_targets or [],
        )
        if not plans:
            raise ValueError("No quantized gemma modules found in checkpoint")

        return cls(plans=plans, ignore=ignore)

    @staticmethod
    def _resolve_num_bits(
        module: str, module_quant_configs: "OrderedDict[str, dict]"
    ) -> Optional[int]:
        """First-match regex resolution, mirroring the gemma runtime semantics."""
        for pattern, opts in module_quant_configs.items():
            if re.search(pattern, module):
                return int(opts.get("num_bits", 4))
        return None

    @classmethod
    def _build_plans(
        cls,
        shard_paths: list[str],
        module_quant_configs: "OrderedDict[str, dict]",
        include_activation_quant: bool = True,
        dequantize_targets: Optional[list[str]] = None,
    ) -> "OrderedDict[str, _ModulePlan]":
        dequantize_targets = dequantize_targets or []
        # Collect (dtype, shape) for every tensor without loading data. Activation
        # scales are tiny scalars, so read their values to detect uncalibrated (0).
        headers: dict[str, tuple[str, list[int]]] = {}
        act_scales: dict[str, float] = {}
        for path in shard_paths:
            with safe_open(path, framework="pt") as f:
                for name in f.keys():
                    sl = f.get_slice(name)
                    headers[name] = (sl.get_dtype(), list(sl.get_shape()))
                    if include_activation_quant and name.endswith(_ACT_SUFFIXES):
                        act_scales[name] = float(f.get_tensor(name).reshape(-1)[0])

        plans: "OrderedDict[str, _ModulePlan]" = OrderedDict()
        for name, (dtype, shape) in headers.items():
            if name.endswith(".weight"):
                module = name[: -len(".weight")]
                scale_key = f"{module}.weight_scale"
                is_embedding = False
            elif name.endswith(".embedding_quantized"):
                module = name[: -len(".embedding_quantized")]
                scale_key = f"{module}.embedding_scale"
                is_embedding = True
            else:
                continue

            if dtype not in _QUANT_STORAGE_DTYPES:
                continue  # full-precision weight, not quantized
            if scale_key not in headers:
                continue  # no scale -> not a quantized module we handle

            num_bits = cls._resolve_num_bits(module, module_quant_configs)
            if num_bits is None:
                # uint8 storage is ambiguous (2 vs 4 bit) without a bit-width rule.
                if dtype == "I8":
                    num_bits = 8
                else:
                    raise ValueError(
                        f"Quantized module {module!r} (uint8) matches no "
                        "module_quant_configs rule; cannot infer bit-width"
                    )

            out_features = shape[0]
            packed_in = shape[1]
            if dtype == "I8":
                if num_bits != 8:
                    raise ValueError(
                        f"{module!r} stored as int8 but resolved to {num_bits}-bit"
                    )
                unpacked_in = packed_in
            else:  # U8
                unpacked_in = packed_in * (8 // num_bits)
            unpacked_shape = [out_features, unpacked_in]

            scale_groups = headers[scale_key][1]
            num_groups = scale_groups[1] if len(scale_groups) > 1 else 1
            if num_groups <= 1:
                strategy = QuantizationStrategy.CHANNEL
                group_size = None
            else:
                strategy = QuantizationStrategy.GROUP
                if unpacked_in % num_groups != 0:
                    raise ValueError(
                        f"{module!r}: in_features {unpacked_in} not divisible by "
                        f"{num_groups} scale groups"
                    )
                group_size = unpacked_in // num_groups

            # Keep static activation scales only when present and calibrated (!= 0);
            # gemma divides by the scale, so a zero scale means "no activation quant".
            in_key = f"{module}{_IN_ACT_SUFFIX}"
            out_key = f"{module}{_OUT_ACT_SUFFIX}"
            in_scale_key = in_key if act_scales.get(in_key, 0.0) != 0.0 else None
            out_scale_key = out_key if act_scales.get(out_key, 0.0) != 0.0 else None

            dequantize = any(re.search(t, module) for t in dequantize_targets)

            plans[module] = _ModulePlan(
                module=module,
                weight_key=name,
                scale_key=scale_key,
                num_bits=num_bits,
                strategy=strategy,
                group_size=group_size,
                unpacked_shape=unpacked_shape,
                is_embedding=is_embedding,
                in_scale_key=in_scale_key,
                out_scale_key=out_scale_key,
                dequantize=dequantize,
            )

        return plans

    # ------------------------------------------------------------------ #
    # Converter protocol
    # ------------------------------------------------------------------ #
    def get_dependencies(self, weight_name: str) -> set[str]:
        plan = self._by_weight_key.get(weight_name)
        if plan is None:
            return set()
        deps = {plan.scale_key}
        if plan.in_scale_key is not None:
            deps.add(plan.in_scale_key)
        if plan.out_scale_key is not None:
            deps.add(plan.out_scale_key)
        return deps

    def validate(self, tensors: dict[str, torch.Tensor]):
        for name in tensors:
            plan = self._by_weight_key.get(name)
            if plan is None:
                continue
            if plan.scale_key not in tensors:
                raise ValueError(
                    f"Quantized weight {name} is missing its scale {plan.scale_key}"
                )

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            # KV-cache scales are out of scope.
            if name.endswith(_KV_DROP_SUFFIXES):
                continue
            # Activation scales are re-emitted (renamed) by their weight's branch
            # when calibrated; drop them from the generic pass either way.
            if name.endswith(_ACT_SUFFIXES):
                continue
            # Weight scales are consumed alongside their primary weight.
            if name in self._scale_keys:
                continue

            plan = self._by_weight_key.get(name)
            if plan is None:
                out[name] = tensor
                continue

            scale = tensors.get(plan.scale_key)
            if scale is None:
                raise ValueError(
                    f"Quantized weight {name} is missing its scale {plan.scale_key}"
                )

            if plan.dequantize:
                # Reconstruct a dense bf16 `weight`; drop all scales. For non-quant
                # -aware consumers (e.g. HF vision/audio towers).
                out[f"{plan.module}.weight"] = _dequantize_dense(
                    tensor, scale, plan
                ).contiguous()
                continue

            if plan.num_bits == 8:
                # int-quantized: a byte already, so store the signed int8 weight
                # directly under `weight` -- no packing, no weight_shape.
                out[f"{plan.module}.weight"] = tensor.to(torch.int8).contiguous()
            else:
                int8_vals = _unpack_gemma_uint8(tensor, plan.num_bits)
                packed = pack_to_int32(int8_vals.contiguous(), plan.num_bits)
                out[f"{plan.module}.weight_packed"] = packed.contiguous()
                out[f"{plan.module}.weight_shape"] = torch.tensor(
                    plan.unpacked_shape, dtype=torch.int64
                )
            out[f"{plan.module}.weight_scale"] = scale.to(self.scale_dtype).contiguous()

            # Preserve calibrated activation scales as CT input_scale/output_scale.
            for src_key, ct_name in (
                (plan.in_scale_key, "input_scale"),
                (plan.out_scale_key, "output_scale"),
            ):
                if src_key is None:
                    continue
                act_scale = tensors.get(src_key)
                if act_scale is None:
                    raise ValueError(f"{name}: missing activation scale {src_key}")
                out[f"{plan.module}.{ct_name}"] = act_scale.to(
                    self.act_scale_dtype
                ).contiguous()

        return out

    def create_config(self) -> QuantizationConfig:
        # Bucket modules by full scheme (weight bits/strategy/group + which
        # activation scales are present); one config group each.
        buckets: "OrderedDict[tuple, list[str]]" = OrderedDict()
        for plan in self.plans.values():
            if plan.dequantize:
                continue  # reconstructed dense; not a quantized config target
            key = (
                plan.num_bits,
                plan.strategy.value,
                plan.group_size,
                plan.in_scale_key is not None,
                plan.out_scale_key is not None,
            )
            buckets.setdefault(key, []).append(plan.module)

        config_groups: dict[str, QuantizationScheme] = {}
        for idx, (
            (num_bits, strategy, group_size, has_in, has_out),
            modules,
        ) in enumerate(buckets.items()):
            weights = QuantizationArgs(
                num_bits=num_bits,
                type=QuantizationType.INT,
                symmetric=True,
                strategy=QuantizationStrategy(strategy),
                group_size=group_size,
            )
            # 8-bit is a byte: store as int-quantized (plain int8) rather than
            # packing into int32.
            fmt = (
                CompressionFormat.int_quantized.value
                if num_bits == 8
                else CompressionFormat.pack_quantized.value
            )
            # `lm_head` is top-level (text-only) or `language_model.lm_head`
            # (multimodal); a suffix regex matches both.
            targets = sorted("re:.*lm_head$" if m == "lm_head" else m for m in modules)
            config_groups[f"group_{idx}"] = QuantizationScheme(
                targets=targets,
                weights=weights,
                input_activations=_activation_args() if has_in else None,
                output_activations=_activation_args() if has_out else None,
                format=fmt,
            )

        return QuantizationConfig(
            config_groups=config_groups,
            ignore=self.ignore,
            format=CompressionFormat.pack_quantized.value,
            quantization_status=QuantizationStatus.COMPRESSED.value,
        )
