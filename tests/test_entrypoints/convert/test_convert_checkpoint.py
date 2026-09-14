# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from compressed_tensors.entrypoints.convert import convert_checkpoint
from compressed_tensors.entrypoints.convert.convert_checkpoint import _resolve_devices
from compressed_tensors.entrypoints.convert.convert_file import (
    convert_file,
    write_checkpoint_quantization_config,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
)
from safetensors.torch import load_file, save_file


_CHECKPOINT_MODULE = "compressed_tensors.entrypoints.convert.convert_checkpoint"
_CONVERT_FILE_MODULE = "compressed_tensors.entrypoints.convert.convert_file"


def _checkpoint_dependencies():
    model_files = {"model.safetensors": "/source/model.safetensors"}
    inverse_weight_map = {"/source/model.safetensors": None}
    return model_files, inverse_weight_map


class NoOpConverter:
    def process(self, tensors):
        return tensors

    def validate(self, tensors):
        return tensors

    def update_config(self, config):
        return config

    def update_model_config(self, model_config):
        return model_config

    def get_dependencies(self, weight_name):
        return set()


class ConfigAppendingConverter(NoOpConverter):
    def __init__(self, group_name):
        self.group_name = group_name

    def update_config(self, config):
        scheme = QuantizationScheme(
            targets=[f"{self.group_name}.weight"],
            weights=QuantizationArgs(num_bits=8),
        )
        if config is None:
            return QuantizationConfig(config_groups={self.group_name: scheme})
        config.config_groups[self.group_name] = scheme
        return config


@patch(f"{_CHECKPOINT_MODULE}.update_safetensors_index")
@patch(f"{_CHECKPOINT_MODULE}.write_checkpoint_quantization_config")
@patch(f"{_CHECKPOINT_MODULE}.exec_jobs_dynamic")
@patch(f"{_CHECKPOINT_MODULE}.exec_jobs")
@patch(f"{_CHECKPOINT_MODULE}.build_inverse_weight_maps")
@patch(f"{_CHECKPOINT_MODULE}.get_weight_map")
@patch(f"{_CHECKPOINT_MODULE}.get_checkpoint_files")
@pytest.mark.parametrize(
    ("device", "resolved_devices"),
    (
        ("cpu", [torch.device("cpu")]),
        (None, [torch.device("cuda:0")]),
    ),
)
def test_convert_checkpoint_preserves_threaded_cpu_path(
    get_checkpoint_files,
    get_weight_map,
    build_inverse_weight_maps,
    exec_jobs,
    exec_jobs_dynamic,
    write_checkpoint_quantization_config,
    update_safetensors_index,
    tmp_path,
    device,
    resolved_devices,
):
    model_files, inverse_weight_map = _checkpoint_dependencies()
    get_checkpoint_files.return_value = model_files
    get_weight_map.return_value = {"weight": "model.safetensors"}
    build_inverse_weight_maps.return_value = {"model.safetensors": inverse_weight_map}
    exec_jobs.side_effect = [[], [(4, {"weight": "model.safetensors"})]]
    converter = Mock()

    with patch(f"{_CHECKPOINT_MODULE}._resolve_devices", return_value=resolved_devices):
        convert_checkpoint(
            "source",
            tmp_path,
            converter,
            max_workers=2,
            device=device,
        )

    assert exec_jobs.call_args_list[0].kwargs == {
        "desc": "Validating",
    }
    assert exec_jobs.call_args_list[0].args[1] == 2
    assert exec_jobs.call_args_list[1].kwargs == {
        "desc": "Converting",
    }
    assert exec_jobs.call_args_list[1].args[1] == 2
    exec_jobs_dynamic.assert_not_called()
    write_checkpoint_quantization_config.assert_called_once_with(tmp_path, [converter])
    update_safetensors_index.assert_called_once_with(
        tmp_path, 4, {"weight": "model.safetensors"}
    )


@patch(f"{_CHECKPOINT_MODULE}.update_safetensors_index")
@patch(f"{_CHECKPOINT_MODULE}.write_checkpoint_quantization_config")
@patch(f"{_CHECKPOINT_MODULE}.exec_jobs_dynamic")
@patch(f"{_CHECKPOINT_MODULE}.exec_jobs", return_value=[])
@patch(f"{_CHECKPOINT_MODULE}.convert_file")
@patch(f"{_CHECKPOINT_MODULE}.build_inverse_weight_maps")
@patch(f"{_CHECKPOINT_MODULE}.get_weight_map")
@patch(f"{_CHECKPOINT_MODULE}.get_checkpoint_files")
def test_convert_checkpoint_schedules_accelerator_jobs(
    get_checkpoint_files,
    get_weight_map,
    build_inverse_weight_maps,
    convert_file_mock,
    exec_jobs,
    exec_jobs_dynamic,
    write_checkpoint_quantization_config,
    update_safetensors_index,
    tmp_path,
):
    model_files, inverse_weight_map = _checkpoint_dependencies()
    get_checkpoint_files.return_value = model_files
    get_weight_map.return_value = {"weight": "model.safetensors"}
    build_inverse_weight_maps.return_value = {"model.safetensors": inverse_weight_map}
    exec_jobs_dynamic.return_value = [(4, {"weight": "model.safetensors"})]
    converter = Mock()
    job_memory_estimator = Mock(return_value=123)
    devices = ["cuda:0", torch.device("cuda:1")]

    convert_checkpoint(
        "source",
        tmp_path,
        converter,
        max_workers=2,
        device=devices,
        job_memory_estimator=job_memory_estimator,
    )

    exec_jobs.assert_called_once()
    job_memory_estimator.assert_called_once_with(inverse_weight_map)
    dynamic_call = exec_jobs_dynamic.call_args
    assert dynamic_call.kwargs["devices"] == [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]
    assert dynamic_call.kwargs["max_workers"] == 2
    assert dynamic_call.kwargs["memory_estimates"] == [123]
    assert dynamic_call.kwargs["desc"] == "Converting"

    dynamic_call.kwargs["jobs"][0](torch.device("cuda:1"))
    convert_file_mock.assert_called_once_with(
        inverse_weight_map,
        Path(tmp_path) / "model.safetensors",
        [converter],
        torch.device("cuda:1"),
    )
    write_checkpoint_quantization_config.assert_called_once_with(tmp_path, [converter])
    update_safetensors_index.assert_called_once_with(
        tmp_path, 4, {"weight": "model.safetensors"}
    )


@patch(f"{_CONVERT_FILE_MODULE}.save_file")
@patch(f"{_CONVERT_FILE_MODULE}.load_tensors_from_inverse_weight_map")
def test_convert_file_loads_tensors_on_requested_device(load_tensors, save_file):
    tensors = {"weight": torch.ones(1)}
    load_tensors.return_value = tensors
    converter = Mock()
    converter.process.return_value = tensors
    inverse_weight_map = {"/source/model.safetensors": None}
    save_path = "/output/model.safetensors"
    device = torch.device("cuda:1")

    total_size, weight_map = convert_file(
        inverse_weight_map,
        save_path,
        [converter],
        device,
    )

    load_tensors.assert_called_once_with(inverse_weight_map, device=device)
    converter.process.assert_called_once_with(tensors)
    save_file.assert_called_once_with(tensors, save_path)
    assert total_size == 4
    assert weight_map == {"weight": "model.safetensors"}


@patch(f"{_CHECKPOINT_MODULE}.get_checkpoint_files")
def test_convert_checkpoint_rejects_empty_device_list(get_checkpoint_files, tmp_path):
    with pytest.raises(ValueError, match="device list cannot be empty"):
        convert_checkpoint("source", tmp_path, Mock(), device=[])

    get_checkpoint_files.assert_not_called()


@patch(f"{_CHECKPOINT_MODULE}.get_checkpoint_files")
def test_convert_checkpoint_requires_estimator_for_accelerator(
    get_checkpoint_files, tmp_path
):
    with pytest.raises(ValueError, match="job_memory_estimator"):
        convert_checkpoint("source", tmp_path, Mock(), device="cuda:0")

    get_checkpoint_files.assert_not_called()


def test_resolve_devices_uses_all_accelerators_by_default():
    with (
        patch(
            f"{_CHECKPOINT_MODULE}.torch.accelerator.current_accelerator",
            return_value=torch.device("cuda"),
        ),
        patch(f"{_CHECKPOINT_MODULE}.torch.accelerator.device_count", return_value=2),
    ):
        devices = _resolve_devices(None)

    assert devices == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_resolve_devices_falls_back_to_cpu():
    with patch(
        f"{_CHECKPOINT_MODULE}.torch.accelerator.current_accelerator",
        return_value=None,
    ):
        devices = _resolve_devices(None)

    assert devices == [torch.device("cpu")]


def test_convert_checkpoint_local_checkpoint_end_to_end(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    tensors = {
        "first.weight": torch.arange(4, dtype=torch.float32),
        "second.weight": torch.ones(2, dtype=torch.float16),
    }
    save_file(tensors, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps({"model_type": "test"}))

    convert_checkpoint(source, output, NoOpConverter(), max_workers=2, device="cpu")

    converted = load_file(output / "model.safetensors")
    assert converted.keys() == tensors.keys()
    for name, tensor in tensors.items():
        assert torch.equal(converted[name], tensor)

    assert json.loads((output / "config.json").read_text()) == {"model_type": "test"}
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == sum(
        tensor.nbytes for tensor in tensors.values()
    )
    assert index["weight_map"] == {
        "first.weight": "model.safetensors",
        "second.weight": "model.safetensors",
    }


def test_convert_checkpoint_creates_output_for_weights_only_checkpoint(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    tensors = {"weight": torch.ones(2)}
    save_file(tensors, source / "model.safetensors")

    convert_checkpoint(source, output, NoOpConverter(), device="cpu")

    assert output.is_dir()
    assert torch.equal(
        load_file(output / "model.safetensors")["weight"], tensors["weight"]
    )
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {"weight": "model.safetensors"}


def test_convert_checkpoint_creates_nested_shard_directories(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    shard_path = source / "nested" / "model.safetensors"
    shard_path.parent.mkdir(parents=True)
    tensors = {"weight": torch.ones(2)}
    save_file(tensors, shard_path)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "nested/model.safetensors"}})
    )

    convert_checkpoint(source, output, NoOpConverter(), device="cpu")

    converted = load_file(output / "nested" / "model.safetensors")
    assert torch.equal(converted["weight"], tensors["weight"])


def test_convert_checkpoint_preserves_config_across_runs(tmp_path):
    source = tmp_path / "source"
    first_output = tmp_path / "first_output"
    second_output = tmp_path / "second_output"
    source.mkdir()
    save_file({"weight": torch.ones(1)}, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps({"model_type": "test"}))

    convert_checkpoint(
        source,
        first_output,
        ConfigAppendingConverter("first"),
        device="cpu",
    )
    convert_checkpoint(
        first_output,
        second_output,
        ConfigAppendingConverter("second"),
        device="cpu",
    )

    config_data = json.loads((second_output / "config.json").read_text())
    quantization_config = QuantizationConfig.model_validate(
        config_data["quantization_config"]
    )
    assert quantization_config.config_groups.keys() == {"first", "second"}


def test_config_writer_loads_ct_config_without_quant_method(tmp_path):
    existing_config = ConfigAppendingConverter("first").update_config(None).model_dump()
    existing_config.pop("quant_method")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"quantization_config": existing_config}))

    write_checkpoint_quantization_config(
        tmp_path,
        [ConfigAppendingConverter("second")],
    )

    written_config = json.loads(config_path.read_text())
    quantization_config = QuantizationConfig.model_validate(
        written_config["quantization_config"]
    )
    assert quantization_config.config_groups.keys() == {"first", "second"}


def test_config_writer_writes_quantization_config_at_root(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"text_config": {"model_type": "gemma4_text"}}))

    write_checkpoint_quantization_config(
        tmp_path,
        [ConfigAppendingConverter("first")],
    )

    written_config = json.loads(config_path.read_text())
    assert "quantization_config" in written_config
    assert "quantization_config" not in written_config["text_config"]


@pytest.mark.parametrize(
    "existing_config",
    ({"quant_method": "awq"}, ["malformed"]),
)
def test_config_writer_ignores_non_ct_config(existing_config, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"quantization_config": existing_config}))
    converter = Mock()
    converter.update_config.return_value = None
    converter.update_model_config.side_effect = lambda model_config: model_config

    write_checkpoint_quantization_config(tmp_path, [converter])

    converter.update_config.assert_called_once_with(None)
    assert "quantization_config" not in json.loads(config_path.read_text())
