# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert import (
    ModelOptConverter,
    ModelOptNvfp4Converter,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    QuantizationType,
)


def _nvfp4_module(prefix, out=64, inp=32, device="cpu"):
    """ModelOpt NVFP4 params: packed u8 weight + fp8 block scale + f32 scalars"""
    with torch.device(device):
        return {
            f"{prefix}.weight": torch.randint(
                0, 255, (out, inp // 2), dtype=torch.uint8
            ),
            f"{prefix}.weight_scale": torch.rand(out, inp // 16).to(
                torch.float8_e4m3fn
            ),
            f"{prefix}.weight_scale_2": torch.rand(1),
            f"{prefix}.input_scale": torch.rand(1),
        }


def _fp8_module(prefix, out=64, inp=32, device="cpu"):
    """ModelOpt FP8 params: fp8 weight + f32 per-tensor scales"""
    with torch.device(device):
        return {
            f"{prefix}.weight": torch.rand(out, inp).to(torch.float8_e4m3fn),
            f"{prefix}.weight_scale": torch.rand(1),
            f"{prefix}.input_scale": torch.rand(1),
        }


def _fp8_block_module(prefix, out=256, inp=256, device="cpu"):
    """ModelOpt FP8_PB_WO params: fp8 weight + f32 4-D block scales"""
    with torch.device(device):
        return {
            f"{prefix}.weight": torch.rand(out, inp).to(torch.float8_e4m3fn),
            f"{prefix}.weight_scale": torch.rand(out // 128, 1, inp // 128, 1),
        }


def _int8_sq_module(prefix, out=64, inp=32, device="cpu"):
    """ModelOpt INT8_SQ params: int8 weight + f32 per-channel scale + input scale"""
    with torch.device(device):
        return {
            f"{prefix}.weight": torch.randint(-127, 127, (out, inp), dtype=torch.int8),
            f"{prefix}.weight_scale": torch.rand(out),
            f"{prefix}.input_scale": torch.rand(1),
        }


def _int8_pb_wo_module(prefix, out=64, inp=64, group_size=32, device="cpu"):
    """ModelOpt INT8_PB_WO params: int8 weight + f32 per-group scale"""
    with torch.device(device):
        return {
            f"{prefix}.weight": torch.randint(-127, 127, (out, inp), dtype=torch.int8),
            f"{prefix}.weight_scale": torch.rand(out, inp // group_size),
        }


def _mxfp8_module(prefix, out=64, inp=32, device="cpu"):
    """ModelOpt MXFP8 params: fp8 weight + e8m0 (u8) per-32-block scale"""
    with torch.device(device):
        return {
            f"{prefix}.weight": torch.rand(out, inp).to(torch.float8_e4m3fn),
            f"{prefix}.weight_scale": torch.randint(120, 135, (out, inp // 32)).to(
                torch.uint8
            ),
        }


MIXED_PRECISION_CONFIG = {
    "quant_algo": "MIXED_PRECISION",
    "quant_method": "modelopt_mixed",
    "producer": {"name": "modelopt", "version": "0.37.0"},
    "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
    "ignore": ["model.embed_tokens", "vision_model.*"],
    "quantized_layers": {
        "model.layers.0.self_attn.k_proj": {"quant_algo": "NVFP4", "group_size": 16},
        "model.layers.0.mlp.up_proj": {"quant_algo": "NVFP4", "group_size": 16},
        "model.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
        "lm_head": {"quant_algo": "FP8"},
    },
}

SINGLE_ALGO_CONFIG = {
    "quant_algo": "NVFP4",
    "quant_method": "modelopt",
    "producer": {"name": "modelopt", "version": "0.35.0"},
    "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
    "ignore": ["lm_head"],
    "group_size": 16,
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16,
            },
            "input_activations": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16,
            },
        }
    },
}

HF_QUANT_CONFIG = {
    "producer": {"name": "modelopt", "version": "0.35.0"},
    "quantization": {
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": "FP8",
        "group_size": 16,
        "exclude_modules": ["lm_head"],
    },
}


class TestManualModeParity:
    """ModelOptConverter(quant_algo="NVFP4") matches ModelOptNvfp4Converter"""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "converter",
        [
            ModelOptNvfp4Converter(targets=[r"re:.*mlp\..*proj$"]),
            ModelOptConverter(quant_algo="NVFP4", targets=[r"re:.*mlp\..*proj$"]),
        ],
        ids=["modelopt_nvfp4", "modelopt_general"],
    )
    def test_nvfp4_process_parity(self, converter):
        tensors = _nvfp4_module("model.layer0.mlp.up_proj")
        tensors["model.embed_tokens.weight"] = torch.randn(
            128, 128, dtype=torch.bfloat16
        )

        result = converter.process(dict(tensors))

        assert "model.layer0.mlp.up_proj.input_scale" not in result
        assert torch.allclose(
            result["model.layer0.mlp.up_proj.input_global_scale"],
            1 / tensors["model.layer0.mlp.up_proj.input_scale"],
        )
        assert torch.equal(
            result["model.layer0.mlp.up_proj.weight_packed"],
            tensors["model.layer0.mlp.up_proj.weight"],
        )
        assert (
            result["model.layer0.mlp.up_proj.weight_scale"]
            is tensors["model.layer0.mlp.up_proj.weight_scale"]
        )
        assert torch.allclose(
            result["model.layer0.mlp.up_proj.weight_global_scale"],
            1 / tensors["model.layer0.mlp.up_proj.weight_scale_2"],
        )
        # unquantized weights pass through
        assert (
            result["model.embed_tokens.weight"] is tensors["model.embed_tokens.weight"]
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "converter",
        [
            ModelOptNvfp4Converter(targets=[r"re:.*down_proj$"]),
            ModelOptConverter(quant_algo="NVFP4", targets=[r"re:.*down_proj$"]),
        ],
        ids=["modelopt_nvfp4", "modelopt_general"],
    )
    def test_nvfp4_get_dependencies_parity(self, converter):
        deps = converter.get_dependencies("model.layer0.mlp.down_proj.weight")
        assert deps == {
            "model.layer0.mlp.down_proj.input_scale",
            "model.layer0.mlp.down_proj.weight_scale",
            "model.layer0.mlp.down_proj.weight_scale_2",
        }
        assert (
            converter.get_dependencies("model.layer0.mlp.down_proj.weight_scale")
            == set()
        )
        assert converter.get_dependencies("model.layer0.mlp.up_proj.weight") == set()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "converter",
        [
            ModelOptNvfp4Converter(targets=[r"re:.*up_proj$"]),
            ModelOptConverter(quant_algo="NVFP4", targets=[r"re:.*up_proj$"]),
        ],
        ids=["modelopt_nvfp4", "modelopt_general"],
    )
    def test_nvfp4_validate_orphan_parity(self, converter):
        tensors = _nvfp4_module("model.layer0.mlp.up_proj")
        # untargeted module carrying a qparam
        tensors["model.layer0.mlp.down_proj.weight_scale"] = torch.rand(64, 2).to(
            torch.float8_e4m3fn
        )

        with pytest.raises(ValueError, match="orphan qparams"):
            converter.validate(dict(tensors))


class TestManualModeFormats:
    """Per-algo conversion of each supported ModelOpt format"""

    @pytest.mark.unit
    @pytest.mark.parametrize("device", ["cpu", "meta"])
    def test_fp8_passthrough(self, device):
        converter = ModelOptConverter(quant_algo="FP8", targets=[r"re:.*in_proj$"])
        tensors = _fp8_module("model.layer0.mixer.in_proj", device=device)
        tensors["model.embed_tokens.weight"] = torch.ones(
            128, 64, dtype=torch.bfloat16, device=device
        )

        result = converter.validate(dict(tensors))

        for name, tensor in tensors.items():
            assert result[name] is tensor

    @pytest.mark.unit
    def test_fp8_config(self):
        converter = ModelOptConverter(quant_algo="FP8", targets=["Linear"])

        config = converter.update_config(None)

        group = config.config_groups["config_group_0"]
        assert group.format == CompressionFormat.float_quantized.value
        assert group.weights.num_bits == 8
        assert group.weights.type == QuantizationType.FLOAT
        assert group.weights.strategy == QuantizationStrategy.TENSOR
        assert group.input_activations.strategy == QuantizationStrategy.TENSOR
        assert not group.input_activations.dynamic
        assert config.format == CompressionFormat.float_quantized.value

    @pytest.mark.unit
    @pytest.mark.parametrize("device", ["cpu", "meta"])
    def test_fp8_per_channel_passthrough(self, device):
        converter = ModelOptConverter(
            quant_algo="FP8_PER_CHANNEL_PER_TOKEN", targets=[r"re:.*in_proj$"]
        )
        tensors = _fp8_module("model.layer0.mixer.in_proj", device=device)
        # dynamic activations carry no input_scale
        del tensors["model.layer0.mixer.in_proj.input_scale"]

        result = converter.validate(dict(tensors))

        for name, tensor in tensors.items():
            assert result[name] is tensor

    @pytest.mark.unit
    def test_fp8_per_channel_config(self):
        converter = ModelOptConverter(quant_algo="FP8_PER_CHANNEL_PER_TOKEN")

        config = converter.update_config(None)

        group = config.config_groups["config_group_0"]
        assert group.weights.strategy == QuantizationStrategy.CHANNEL
        assert group.input_activations.strategy == QuantizationStrategy.TOKEN
        assert group.input_activations.dynamic
        assert group.format == CompressionFormat.float_quantized.value

    @pytest.mark.unit
    @pytest.mark.parametrize("device", ["cpu", "meta"])
    def test_fp8_block_scales(self, device):
        converter = ModelOptConverter(
            quant_algo="FP8_PB_WO", targets=[r"re:.*in_proj$"]
        )
        tensors = _fp8_block_module("model.layer0.mixer.in_proj", device=device)

        result = converter.validate(dict(tensors))

        assert "model.layer0.mixer.in_proj.weight_scale" not in result
        scale_inv = result["model.layer0.mixer.in_proj.weight_scale_inv"]
        assert scale_inv.shape == (2, 2)
        if device == "cpu":
            assert torch.equal(
                scale_inv, tensors["model.layer0.mixer.in_proj.weight_scale"].squeeze()
            )
        assert (
            result["model.layer0.mixer.in_proj.weight"]
            is tensors["model.layer0.mixer.in_proj.weight"]
        )

    @pytest.mark.unit
    def test_fp8_block_config(self):
        converter = ModelOptConverter(quant_algo="FP8_PB_WO")

        config = converter.update_config(None)

        group = config.config_groups["config_group_0"]
        assert group.weights.strategy == QuantizationStrategy.BLOCK
        assert group.weights.block_structure == [128, 128]
        assert group.format == CompressionFormat.float_quantized.value

    @pytest.mark.unit
    @pytest.mark.parametrize("device", ["cpu", "meta"])
    def test_int8_sq_passthrough(self, device):
        converter = ModelOptConverter(quant_algo="INT8_SQ", targets=[r"re:.*in_proj$"])
        tensors = _int8_sq_module("model.layer0.mixer.in_proj", device=device)

        result = converter.validate(dict(tensors))

        for name, tensor in tensors.items():
            assert result[name] is tensor

    @pytest.mark.unit
    def test_int8_sq_config(self):
        converter = ModelOptConverter(quant_algo="INT8_SQ")

        config = converter.update_config(None)

        group = config.config_groups["config_group_0"]
        assert group.format == CompressionFormat.int_quantized.value
        assert group.weights.type == QuantizationType.INT
        assert group.weights.strategy == QuantizationStrategy.CHANNEL
        assert group.input_activations.type == QuantizationType.INT
        assert group.input_activations.strategy == QuantizationStrategy.TENSOR
        assert not group.input_activations.dynamic

    @pytest.mark.unit
    def test_int8_pb_wo_packing_round_trip(self):
        converter = ModelOptConverter(
            quant_algo="INT8_PB_WO", group_size=32, targets=[r"re:.*in_proj$"]
        )
        tensors = _int8_pb_wo_module("model.layer0.mixer.in_proj")

        result = converter.validate(dict(tensors))

        module = "model.layer0.mixer.in_proj"
        assert f"{module}.weight" not in result
        packed = result[f"{module}.weight_packed"]
        assert packed.dtype == torch.int32
        assert packed.shape == (64, 16)  # 64 inputs packed 4 per int32
        assert torch.equal(result[f"{module}.weight_shape"], torch.tensor((64, 64)))
        assert result[f"{module}.weight_scale"] is tensors[f"{module}.weight_scale"]

        unpacked = unpack_from_int32(packed, 8, (64, 64))
        assert torch.equal(unpacked, tensors[f"{module}.weight"])

    @pytest.mark.unit
    @pytest.mark.parametrize("device", ["cpu", "meta"])
    def test_int8_pb_wo_validate_on_meta(self, device):
        converter = ModelOptConverter(quant_algo="INT8_PB_WO")
        tensors = _int8_pb_wo_module("model.layer0.mixer.in_proj", device=device)

        result = converter.validate(dict(tensors))

        module = "model.layer0.mixer.in_proj"
        assert result[f"{module}.weight_packed"].dtype == torch.int32
        assert result[f"{module}.weight_packed"].shape == (64, 16)
        assert f"{module}.weight" not in result

    @pytest.mark.unit
    @pytest.mark.parametrize("device", ["cpu", "meta"])
    def test_int8_pb_wo_rejects_non_int8_weight(self, device):
        converter = ModelOptConverter(quant_algo="INT8_PB_WO")
        tensors = _int8_pb_wo_module("model.layer0.mixer.in_proj", device=device)
        tensors["model.layer0.mixer.in_proj.weight"] = tensors[
            "model.layer0.mixer.in_proj.weight"
        ].to(torch.bfloat16)

        with pytest.raises(ValueError, match="torch.int8"):
            converter.validate(dict(tensors))

    @pytest.mark.unit
    def test_int8_pb_wo_config(self):
        converter = ModelOptConverter(quant_algo="INT8_PB_WO", group_size=32)

        config = converter.update_config(None)

        group = config.config_groups["config_group_0"]
        assert group.format == CompressionFormat.pack_quantized.value
        assert group.weights.strategy == QuantizationStrategy.GROUP
        assert group.weights.group_size == 32
        assert group.input_activations is None

    @pytest.mark.unit
    @pytest.mark.parametrize("device", ["cpu", "meta"])
    def test_mxfp8_passthrough(self, device):
        converter = ModelOptConverter(quant_algo="MXFP8", targets=[r"re:.*in_proj$"])
        tensors = _mxfp8_module("model.layer0.mixer.in_proj", device=device)

        result = converter.validate(dict(tensors))

        for name, tensor in tensors.items():
            assert result[name] is tensor

    @pytest.mark.unit
    def test_mxfp8_config(self):
        converter = ModelOptConverter(quant_algo="MXFP8")

        config = converter.update_config(None)

        group = config.config_groups["config_group_0"]
        assert group.format == CompressionFormat.mxfp8_quantized.value
        assert group.weights.group_size == 32
        assert group.weights.scale_dtype == torch.uint8

    @pytest.mark.unit
    @pytest.mark.parametrize("device", ["cpu", "meta"])
    def test_w4a16_nvfp4_drops_input_scale(self, device):
        converter = ModelOptConverter(
            quant_algo="W4A16_NVFP4", targets=[r"re:.*up_proj$"]
        )
        tensors = _nvfp4_module("model.layer0.mlp.up_proj", device=device)

        result = converter.validate(dict(tensors))

        module = "model.layer0.mlp.up_proj"
        assert f"{module}.input_scale" not in result
        assert f"{module}.input_global_scale" not in result
        if device == "cpu":
            torch.testing.assert_close(
                result[f"{module}.weight_global_scale"],
                1 / tensors[f"{module}.weight_scale_2"],
            )
        assert result[f"{module}.weight_packed"].dtype == torch.uint8

    @pytest.mark.unit
    def test_w4a16_nvfp4_config(self):
        converter = ModelOptConverter(quant_algo="W4A16_NVFP4")

        config = converter.update_config(None)

        group = config.config_groups["config_group_0"]
        assert group.format == CompressionFormat.nvfp4_pack_quantized.value
        assert group.input_activations is None
        assert group.weights.group_size == 16

    @pytest.mark.unit
    def test_unsupported_algo(self):
        with pytest.raises(NotImplementedError, match="W4A16_AWQ"):
            ModelOptConverter(quant_algo="W4A16_AWQ")
        with pytest.raises(NotImplementedError, match="Unknown ModelOpt quant_algo"):
            ModelOptConverter(quant_algo="NOT_AN_ALGO")

    @pytest.mark.unit
    def test_quant_algo_required(self):
        with pytest.raises(ValueError, match="quant_algo is required"):
            ModelOptConverter()


class TestMixedPrecisionConfig:
    """Config-driven conversion, like nvidia mixed-precision ModelOpt exports"""

    @pytest.fixture
    def converter(self):
        return ModelOptConverter(quantization_config=MIXED_PRECISION_CONFIG)

    @pytest.fixture
    def mixed_tensors(self):
        tensors = {
            "model.embed_tokens.weight": torch.ones(128, 64, dtype=torch.bfloat16),
            "vision_model.fc1.weight": torch.ones(32, 64, dtype=torch.bfloat16),
            "model.layers.0.self_attn.k_proj.k_scale": torch.rand(1),
        }
        tensors.update(_nvfp4_module("model.layers.0.self_attn.k_proj"))
        tensors.update(_nvfp4_module("model.layers.0.mlp.up_proj"))
        tensors.update(_fp8_module("model.layers.0.mixer.in_proj"))
        tensors.update(_fp8_module("lm_head"))
        return tensors

    @pytest.mark.unit
    @pytest.mark.parametrize("device", ["cpu", "meta"])
    def test_process_per_module_algo(self, converter, mixed_tensors, device):
        tensors = {name: tensor.to(device) for name, tensor in mixed_tensors.items()}

        result = converter.validate(dict(tensors))

        # NVFP4 modules converted
        for module in (
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.mlp.up_proj",
        ):
            assert f"{module}.weight_packed" in result
            assert f"{module}.weight_global_scale" in result
            assert f"{module}.input_global_scale" in result
            assert f"{module}.weight_scale_2" not in result
            assert f"{module}.input_scale" not in result
            if device == "cpu":
                torch.testing.assert_close(
                    result[f"{module}.weight_global_scale"],
                    1 / tensors[f"{module}.weight_scale_2"],
                )
                torch.testing.assert_close(
                    result[f"{module}.input_global_scale"],
                    1 / tensors[f"{module}.input_scale"],
                )

        # FP8 modules untouched
        for module in ("model.layers.0.mixer.in_proj", "lm_head"):
            assert result[f"{module}.weight"].dtype == torch.float8_e4m3fn
            assert f"{module}.weight_scale" in result
            assert f"{module}.input_scale" in result
            assert result[f"{module}.weight_scale"] is tensors[f"{module}.weight_scale"]

        # ignored modules untouched, wildcard ignore included
        assert (
            result["model.embed_tokens.weight"] is tensors["model.embed_tokens.weight"]
        )
        assert result["vision_model.fc1.weight"] is tensors["vision_model.fc1.weight"]

        # kv cache scale converted to the scheme's dtype (bf16 default)
        k_scale = result["model.layers.0.self_attn.k_proj.k_scale"]
        assert k_scale.dtype == torch.bfloat16

    @pytest.mark.unit
    def test_get_dependencies_per_module_algo(self, converter):
        assert converter.get_dependencies("model.layers.0.mlp.up_proj.weight") == {
            "model.layers.0.mlp.up_proj.input_scale",
            "model.layers.0.mlp.up_proj.weight_scale",
            "model.layers.0.mlp.up_proj.weight_scale_2",
        }
        assert converter.get_dependencies("lm_head.weight") == {
            "lm_head.weight_scale",
            "lm_head.input_scale",
        }
        # kv cache scales are not dependencies: they convert standalone
        # (dtype cast only) and may be absent for layers whose KV cache was
        # not calibrated
        assert converter.get_dependencies("model.layers.0.self_attn.k_proj.weight") == {
            "model.layers.0.self_attn.k_proj.input_scale",
            "model.layers.0.self_attn.k_proj.weight_scale",
            "model.layers.0.self_attn.k_proj.weight_scale_2",
        }
        # unquantized modules have no dependencies
        assert converter.get_dependencies("model.embed_tokens.weight") == set()

    @pytest.mark.unit
    def test_update_config(self, converter):
        config = converter.update_config(None)

        assert config.quantization_status == QuantizationStatus.COMPRESSED
        assert config.format == CompressionFormat.mixed_precision.value
        assert config.kv_cache_scheme == QuantizationArgs(
            num_bits=8, dynamic=False, type=QuantizationType.FLOAT
        )
        # wildcard ignore translated to a regex
        assert "model.embed_tokens" in config.ignore
        assert "re:vision_model\\..*" in config.ignore

        assert len(config.config_groups) == 2
        nvfp4_group = next(
            group
            for group in config.config_groups.values()
            if group.format == CompressionFormat.nvfp4_pack_quantized.value
        )
        fp8_group = next(
            group
            for group in config.config_groups.values()
            if group.format == CompressionFormat.float_quantized.value
        )

        assert nvfp4_group.targets == [
            "model.layers.0.mlp.up_proj",
            "model.layers.0.self_attn.k_proj",
        ]
        assert nvfp4_group.weights.group_size == 16
        assert nvfp4_group.input_activations.num_bits == 4

        assert fp8_group.targets == [
            "lm_head",
            "model.layers.0.mixer.in_proj",
        ]
        assert fp8_group.weights.num_bits == 8

    @pytest.mark.unit
    def test_validate_rejects_orphan_qparam(self, converter):
        tensors = _fp8_module("model.layers.0.mixer.in_proj")
        # module absent from quantized_layers carrying a qparam
        tensors.update(_fp8_module("model.layers.1.mixer.in_proj"))

        with pytest.raises(ValueError, match="orphan qparams"):
            converter.validate(dict(tensors))

    @pytest.mark.unit
    def test_validate_rejects_missing_scale(self, converter):
        tensors = _nvfp4_module("model.layers.0.mlp.up_proj")
        del tensors["model.layers.0.mlp.up_proj.weight_scale"]

        with pytest.raises(
            ValueError,
            match="without corresponding model.layers.0.mlp.up_proj.weight_scale",
        ):
            converter.validate(dict(tensors))

    @pytest.mark.unit
    def test_config_cannot_combine_manual_args(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            ModelOptConverter(
                quantization_config=MIXED_PRECISION_CONFIG, targets=["Linear"]
            )


class TestContainerEntries:
    """Module-level quantized_layers entries covering per-expert projections.

    ModelOpt exports such entries for fused MoE blocks (e.g. the MTP layer of
    Nemotron-H checkpoints): ``...mixer.experts`` names the experts container,
    while the checkpoint carries its quantized tensors per-expert, at
    ``...mixer.experts.{i}.{up,down}_proj``.
    """

    CONTAINER_CONFIG = {
        "quant_algo": "MIXED_PRECISION",
        "quant_method": "modelopt_mixed",
        "ignore": ["model.layers.1.mlp.gate"],
        "quantized_layers": {
            "model.layers.1.mixer.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
        },
    }

    @pytest.fixture
    def converter(self):
        return ModelOptConverter(quantization_config=self.CONTAINER_CONFIG)

    @pytest.fixture
    def expert_tensors(self):
        tensors = {}
        for expert in (0, 1, 511):
            tensors.update(
                _nvfp4_module(f"model.layers.1.mixer.experts.{expert}.up_proj")
            )
            tensors.update(
                _nvfp4_module(f"model.layers.1.mixer.experts.{expert}.down_proj")
            )
        # the router of the ignored kind is left untouched
        tensors["model.layers.1.mlp.gate.weight"] = torch.ones(8, 64)
        return tensors

    @pytest.mark.unit
    def test_container_entry_converts_per_expert_params(
        self, converter, expert_tensors
    ):
        result = converter.validate(dict(expert_tensors))

        for expert in (0, 1, 511):
            for proj in ("up_proj", "down_proj"):
                module = f"model.layers.1.mixer.experts.{expert}.{proj}"
                assert f"{module}.weight_packed" in result
                assert f"{module}.weight_global_scale" in result
                assert f"{module}.input_global_scale" in result
                assert f"{module}.weight_scale_2" not in result
                assert f"{module}.weight" not in result

    @pytest.mark.unit
    def test_container_entry_dependencies(self, converter):
        module = "model.layers.1.mixer.experts.5.up_proj"
        assert converter.get_dependencies(f"{module}.weight") == {
            f"{module}.input_scale",
            f"{module}.weight_scale",
            f"{module}.weight_scale_2",
        }
        # a module which is neither an entry nor below one has no assignment
        assert converter.get_dependencies("model.layers.1.mlp.gate.weight") == set()

    @pytest.mark.unit
    def test_container_entry_config_target(self, converter, expert_tensors):
        # resolve assignments first, as convert_checkpoint does: the container
        # entry is discovered while building the weight maps
        converter.validate(dict(expert_tensors))

        config = converter.update_config(None)
        (group,) = config.config_groups.values()
        assert group.targets == [
            r"re:^model\.layers\.1\.mixer\.experts"
            r"\.\d+\.(gate_proj|up_proj|down_proj|gate_up_proj)$"
        ]

        # the regex targets the per-expert projections the entry covers
        from compressed_tensors.utils.match import match_name

        assert match_name("model.layers.1.mixer.experts.5.up_proj", group.targets[0])
        assert match_name("model.layers.1.mixer.experts.0.down_proj", group.targets[0])
        assert not match_name("model.layers.1.mixer.experts.gate", group.targets[0])
        assert not match_name("model.layers.1.mixer.experts", group.targets[0])

    @pytest.mark.unit
    def test_prefix_is_anchored(self, converter):
        # an entry must not match a module whose name merely starts with it
        assignment = converter._assignment_for("model.layers.1.mixer.experts_extra.0")
        assert assignment is None


class TestConfigMerging:
    """config.json's quantization_config merged with hf_quant_config.json"""

    @pytest.mark.unit
    def test_merge_supplements_missing_layers(self):
        from compressed_tensors.entrypoints.convert.converters.modelopt import (
            _merge_modelopt_configs,
        )

        config_quant = {
            "quant_algo": "MIXED_PRECISION",
            "quant_method": "modelopt_mixed",
            "ignore": ["lm_head"],
            "quantized_layers": {"model.layers.0.mlp.up_proj": {"quant_algo": "NVFP4"}},
        }
        hf_quant = {
            "producer": {"name": "modelopt"},
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "exclude_modules": ["model.embed_tokens"],
                "quantized_layers": {
                    # config.json's entry wins on conflict
                    "model.layers.0.mlp.up_proj": {"quant_algo": "FP8"},
                    # entry only hf_quant_config.json describes
                    "model.mtp.layers.1.mixer.experts": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                },
            },
        }

        merged = _merge_modelopt_configs(config_quant, hf_quant)

        assert merged["quantized_layers"]["model.layers.0.mlp.up_proj"] == {
            "quant_algo": "NVFP4"
        }
        assert merged["quantized_layers"]["model.mtp.layers.1.mixer.experts"] == {
            "quant_algo": "NVFP4",
            "group_size": 16,
        }
        # ignore lists are unioned
        assert set(merged["ignore"]) == {"lm_head", "model.embed_tokens"}
        # flat layout, not the hf_quant_config wrapper
        assert "quantization" not in merged

    @pytest.mark.unit
    def test_merge_skipped_for_single_algo_configs(self):
        from compressed_tensors.entrypoints.convert.converters.modelopt import (
            _merge_modelopt_configs,
        )

        config_quant = {"quant_algo": "NVFP4", "group_size": 16}
        hf_quant = {
            "quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": {}}
        }

        assert _merge_modelopt_configs(config_quant, hf_quant) is config_quant
        assert _merge_modelopt_configs(None, hf_quant) is hf_quant
        assert _merge_modelopt_configs(config_quant, None) is config_quant

    @pytest.mark.unit
    def test_from_pretrained_merges_both_sources(self, tmp_path):
        # config.json's quantization_config omits the MTP entries which
        # hf_quant_config.json still describes
        config_quant = {
            "quant_algo": "MIXED_PRECISION",
            "quant_method": "modelopt_mixed",
            "ignore": ["lm_head"],
            "quantized_layers": {"model.layers.0.mlp.up_proj": {"quant_algo": "NVFP4"}},
        }
        hf_quant = {
            "producer": {"name": "modelopt"},
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "exclude_modules": [],
                "quantized_layers": {
                    "model.layers.0.mlp.up_proj": {"quant_algo": "NVFP4"},
                    "model.mtp.layers.1.mixer.experts": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                },
            },
        }
        with open(tmp_path / "config.json", "w") as file:
            json.dump({"quantization_config": config_quant}, file)
        with open(tmp_path / "hf_quant_config.json", "w") as file:
            json.dump(hf_quant, file)

        converter = ModelOptConverter.from_pretrained(tmp_path)

        assert "model.layers.0.mlp.up_proj" in converter._module_algos
        assert "model.mtp.layers.1.mixer.experts" in converter._module_algos
        assignment = converter._assignment_for(
            "model.mtp.layers.1.mixer.experts.3.up_proj"
        )
        assert assignment is not None and assignment[0] == "nvfp4"


class TestConfigParsing:
    """Parsing of each ModelOpt config layout"""

    @pytest.mark.unit
    def test_from_modelopt_config(self):
        converter = ModelOptConverter.from_modelopt_config(SINGLE_ALGO_CONFIG)

        assert converter._algo == "nvfp4"
        assert converter._group_size == 16
        assert converter.targets == ["Linear"]
        assert converter.ignore == ["lm_head"]
        assert converter.kv_cache_scheme == QuantizationArgs(
            num_bits=8, dynamic=False, type=QuantizationType.FLOAT
        )

    @pytest.mark.unit
    def test_hf_quant_config_layout(self):
        converter = ModelOptConverter(quantization_config=HF_QUANT_CONFIG)

        assert converter._algo == "nvfp4"
        assert converter._group_size == 16
        # no config_groups in hf_quant_config.json: default to all Linears
        assert converter.targets == ["Linear"]
        assert converter.ignore == ["lm_head"]
        # kv_cache_quant_algo string parsed into args
        assert converter.kv_cache_scheme == QuantizationArgs(
            num_bits=8, dynamic=False, type=QuantizationType.FLOAT
        )

    @pytest.mark.unit
    def test_mixed_config_requires_quantized_layers(self):
        config = {"quant_algo": "MIXED_PRECISION", "quant_method": "modelopt_mixed"}
        with pytest.raises(ValueError, match="quantized_layers"):
            ModelOptConverter(quantization_config=config)

    @pytest.mark.unit
    def test_config_requires_quant_algo(self):
        with pytest.raises(ValueError, match="quant_algo"):
            ModelOptConverter(quantization_config={"quant_method": "modelopt"})

    @pytest.mark.unit
    def test_unsupported_algo_in_config(self):
        config = {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {"model.layer": {"quant_algo": "W4A16_AWQ"}},
        }
        with pytest.raises(NotImplementedError, match="W4A16_AWQ"):
            ModelOptConverter(quantization_config=config)

    @pytest.mark.unit
    def test_from_pretrained_local_config_json(self, tmp_path):
        config = {"quantization_config": SINGLE_ALGO_CONFIG}
        with open(tmp_path / "config.json", "w") as file:
            json.dump(config, file)

        converter = ModelOptConverter.from_pretrained(tmp_path)

        assert converter._algo == "nvfp4"
        assert converter.kv_cache_scheme is not None

    @pytest.mark.unit
    def test_from_pretrained_local_hf_quant_config(self, tmp_path):
        # config.json without a modelopt quantization_config falls back to
        # hf_quant_config.json
        with open(tmp_path / "config.json", "w") as file:
            json.dump({"architectures": ["LlamaForCausalLM"]}, file)
        with open(tmp_path / "hf_quant_config.json", "w") as file:
            json.dump(HF_QUANT_CONFIG, file)

        converter = ModelOptConverter.from_pretrained(tmp_path)

        assert converter._algo == "nvfp4"
        assert converter._group_size == 16

    @pytest.mark.unit
    def test_from_pretrained_rejects_non_modelopt(self, tmp_path):
        config = {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
            }
        }
        with open(tmp_path / "config.json", "w") as file:
            json.dump(config, file)

        with pytest.raises(ValueError, match="Could not find a ModelOpt"):
            ModelOptConverter.from_pretrained(tmp_path)

    @pytest.mark.unit
    def test_from_pretrained_nested_text_config(self, tmp_path):
        # VLMs may nest quantization_config under text_config
        config = {
            "text_config": {"quantization_config": SINGLE_ALGO_CONFIG},
        }
        with open(tmp_path / "config.json", "w") as file:
            json.dump(config, file)

        converter = ModelOptConverter.from_pretrained(tmp_path)

        assert converter._algo == "nvfp4"


class TestConfigChaining:
    """update_config merge behavior with existing configs"""

    @pytest.mark.unit
    def test_update_config_merges(self):
        converter = ModelOptConverter(quant_algo="FP8", targets=[r"re:.*proj$"])
        existing = QuantizationConfig(
            config_groups={
                "prior_group": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(num_bits=8, type="int", symmetric=True),
                )
            },
        )

        merged = converter.update_config(existing)

        assert merged is existing
        assert len(merged.config_groups) == 2

    @pytest.mark.unit
    def test_mixed_precision_single_format(self):
        # mixed-precision config whose algos share one format collapses it
        config = {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.a": {"quant_algo": "FP8"},
                "model.b": {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"},
            },
        }
        converter = ModelOptConverter(quantization_config=config)

        config_out = converter.update_config(None)

        assert config_out.format == CompressionFormat.float_quantized.value
        assert len(config_out.config_groups) == 2

    @pytest.mark.unit
    def test_group_size_splits_config_groups(self):
        config = {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.a": {"quant_algo": "INT8_PB_WO", "group_size": 16},
                "model.b": {"quant_algo": "INT8_PB_WO", "group_size": 32},
            },
        }
        converter = ModelOptConverter(quantization_config=config)

        config_out = converter.update_config(None)

        group_sizes = {
            group.weights.group_size for group in config_out.config_groups.values()
        }
        assert group_sizes == {16, 32}
