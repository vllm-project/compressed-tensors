# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from compressed_tensors.entrypoints.convert import convert_checkpoint
from compressed_tensors.entrypoints.convert.convert_checkpoint import (
    _auto_max_workers,
    _max_threads,
    _resolve_devices,
)
from compressed_tensors.entrypoints.convert.convert_file import (
    convert_file,
    write_checkpoint_quantization_config,
)
from compressed_tensors.entrypoints.convert.memory import estimate_job_memory
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
)
from safetensors.torch import load_file, save_file
from tests.testing_utils import requires_gpu


_CHECKPOINT_MODULE = sys.modules[
    "compressed_tensors.entrypoints.convert.convert_checkpoint"
]
_MEMORY_MODULE = sys.modules["compressed_tensors.entrypoints.convert.memory"]
_CONVERT_FILE_MODULE = "compressed_tensors.entrypoints.convert.convert_file"
_GB = 1024**3


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


@patch.object(_CHECKPOINT_MODULE, "update_safetensors_index")
@patch.object(_CHECKPOINT_MODULE, "write_checkpoint_quantization_config")
@patch.object(_CHECKPOINT_MODULE, "exec_jobs_dynamic")
@patch.object(_CHECKPOINT_MODULE, "exec_jobs")
@patch.object(_CHECKPOINT_MODULE, "build_inverse_weight_maps")
@patch.object(_CHECKPOINT_MODULE, "get_weight_map")
@patch.object(_CHECKPOINT_MODULE, "get_checkpoint_files")
@pytest.mark.parametrize(
    ("device", "resolved_devices"),
    (
        ("cpu", [torch.device("cpu")]),
        # device=None on a host with no accelerator resolves to CPU
        (None, [torch.device("cpu")]),
    ),
)
def test_convert_checkpoint_cpu_uses_dynamic_scheduler(
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
    exec_jobs.return_value = []  # validation phase
    exec_jobs_dynamic.return_value = [(4, {"weight": "model.safetensors"})]
    converter = Mock()
    job_memory_estimator = Mock(return_value=1000)

    with patch.object(
        _CHECKPOINT_MODULE, "_resolve_devices", return_value=resolved_devices
    ):
        convert_checkpoint(
            "source",
            tmp_path,
            converter,
            max_workers=2,
            device=device,
            job_memory_estimator=job_memory_estimator,
        )

    # validation still runs through exec_jobs, with one worker per job (one job
    # here) regardless of max_workers
    exec_jobs.assert_called_once()
    assert exec_jobs.call_args.kwargs == {"desc": "Validating"}
    assert exec_jobs.call_args.args[1] == 1

    # conversion is scheduled through exec_jobs_dynamic on CPU too, with memory
    # estimates widened by the cpu margin for host-memory admission
    dynamic_call = exec_jobs_dynamic.call_args
    assert dynamic_call.kwargs["devices"] == resolved_devices
    assert dynamic_call.kwargs["max_workers"] == 2
    assert dynamic_call.kwargs["memory_estimates"] == [
        int(1000 * _CHECKPOINT_MODULE._CPU_MEMORY_MARGIN)
    ]
    assert dynamic_call.kwargs["desc"] == "Converting"
    job_memory_estimator.assert_called_once_with(inverse_weight_map, [converter])

    write_checkpoint_quantization_config.assert_called_once_with(tmp_path, [converter])
    update_safetensors_index.assert_called_once_with(
        tmp_path, 4, {"weight": "model.safetensors"}
    )


@patch.object(_CHECKPOINT_MODULE, "update_safetensors_index")
@patch.object(_CHECKPOINT_MODULE, "write_checkpoint_quantization_config")
@patch.object(_CHECKPOINT_MODULE, "exec_jobs_dynamic")
@patch.object(_CHECKPOINT_MODULE, "exec_jobs", return_value=[])
@patch.object(_CHECKPOINT_MODULE, "convert_file")
@patch.object(_CHECKPOINT_MODULE, "build_inverse_weight_maps")
@patch.object(_CHECKPOINT_MODULE, "get_weight_map")
@patch.object(_CHECKPOINT_MODULE, "get_checkpoint_files")
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
    job_memory_estimator.assert_called_once_with(inverse_weight_map, [converter])
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


@patch.object(_CHECKPOINT_MODULE, "get_checkpoint_files")
def test_convert_checkpoint_rejects_empty_device_list(get_checkpoint_files, tmp_path):
    with pytest.raises(ValueError, match="device list cannot be empty"):
        convert_checkpoint("source", tmp_path, Mock(), device=[])

    get_checkpoint_files.assert_not_called()


def test_convert_checkpoint_defaults_to_auto_max_workers():
    default = inspect.signature(convert_checkpoint).parameters["max_workers"].default
    assert default == "auto"


def test_convert_checkpoint_defaults_to_meta_estimator():
    default = (
        inspect.signature(convert_checkpoint).parameters["job_memory_estimator"].default
    )
    assert default is estimate_job_memory


def test_resolve_devices_uses_all_accelerators_by_default():
    with (
        patch(
            "torch.accelerator.current_accelerator",
            return_value=torch.device("cuda"),
        ),
        patch("torch.accelerator.device_count", return_value=2),
    ):
        devices = _resolve_devices(None)

    assert devices == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_resolve_devices_falls_back_to_cpu():
    with patch(
        "torch.accelerator.current_accelerator",
        return_value=None,
    ):
        devices = _resolve_devices(None)

    assert devices == [torch.device("cpu")]


@pytest.mark.parametrize("max_workers", (2, "auto"))
def test_convert_checkpoint_local_checkpoint_end_to_end(tmp_path, max_workers):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    tensors = {
        "first.weight": torch.arange(4, dtype=torch.float32),
        "second.weight": torch.ones(2, dtype=torch.float16),
    }
    save_file(tensors, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps({"model_type": "test"}))

    convert_checkpoint(
        source, output, NoOpConverter(), max_workers=max_workers, device="cpu"
    )

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


# ── max_workers="auto" ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("free_memory", "memory_estimates", "max_threads", "expected"),
    (
        # bounded by device memory: 10 // 5 + 25 // 5 copies of the largest job
        ({"cuda:0": 10 * _GB, "cuda:1": 25 * _GB}, [4 * _GB, 5 * _GB] * 5, 32, 7),
        # bounded by the number of jobs
        ({"cuda:0": 80 * _GB}, [1 * _GB] * 3, 32, 3),
        # bounded by the thread limit
        ({"cuda:0": 80 * _GB}, [1 * _GB] * 20, 8, 8),
        # largest job does not fit, so the scheduler is left to raise
        ({"cuda:0": 10 * _GB}, [20 * _GB] * 4, 32, 1),
        # zero memory estimates are not bounded by device memory
        ({"cuda:0": 0}, [0] * 20, 32, 20),
    ),
)
def test_auto_max_workers(free_memory, memory_estimates, max_threads, expected):
    free_memory = {torch.device(dev): free for dev, free in free_memory.items()}
    devices = list(free_memory.keys())

    with (
        patch.object(_CHECKPOINT_MODULE, "_snapshot_free", return_value=free_memory),
        patch.object(_CHECKPOINT_MODULE, "_max_threads", return_value=max_threads),
    ):
        assert _auto_max_workers(devices, memory_estimates) == expected


@pytest.mark.parametrize(
    ("sched_getaffinity", "expected"),
    (
        # process is pinned to 8 of the host's 24 CPUs, e.g. in a container
        ({"return_value": set(range(8))}, 8 + 4),
        # sched_getaffinity is not available on macOS and Windows
        ({"side_effect": AttributeError}, 24 + 4),
        # capped at 32, like ThreadPoolExecutor
        ({"return_value": set(range(64))}, 32),
    ),
)
def test_max_threads_counts_available_cpus(sched_getaffinity, expected):
    with (
        patch("os.cpu_count", return_value=24),
        patch("os.sched_getaffinity", create=True, **sched_getaffinity),
    ):
        assert _max_threads() == expected


@pytest.mark.parametrize("memory_estimates", ([0] * 8, [0], []))
def test_auto_max_workers_cpu_runs_sequentially(memory_estimates):
    with patch.object(_CHECKPOINT_MODULE, "_snapshot_free") as snapshot_free:
        assert _auto_max_workers([torch.device("cpu")], memory_estimates) == 1

    snapshot_free.assert_not_called()


@patch.object(_CHECKPOINT_MODULE, "update_safetensors_index")
@patch.object(_CHECKPOINT_MODULE, "write_checkpoint_quantization_config")
@patch.object(_CHECKPOINT_MODULE, "exec_jobs_dynamic")
@patch.object(_CHECKPOINT_MODULE, "exec_jobs", return_value=[])
@patch.object(_CHECKPOINT_MODULE, "_auto_max_workers", return_value=3)
@patch.object(_CHECKPOINT_MODULE, "build_inverse_weight_maps")
@patch.object(_CHECKPOINT_MODULE, "get_weight_map")
@patch.object(_CHECKPOINT_MODULE, "get_checkpoint_files")
def test_convert_checkpoint_resolves_auto_max_workers(
    get_checkpoint_files,
    get_weight_map,
    build_inverse_weight_maps,
    auto_max_workers,
    exec_jobs,
    exec_jobs_dynamic,
    write_checkpoint_quantization_config,
    update_safetensors_index,
    tmp_path,
):
    shard_names = ["model-0.safetensors", "model-1.safetensors"]
    get_checkpoint_files.return_value = {
        name: f"/source/{name}" for name in shard_names
    }
    get_weight_map.return_value = {"weight": shard_names[0]}
    build_inverse_weight_maps.return_value = {
        name: {f"/source/{name}": None} for name in shard_names
    }
    exec_jobs_dynamic.return_value = [(4, {"weight": shard_names[0]}), (0, {})]
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]

    convert_checkpoint(
        "source",
        tmp_path,
        Mock(),
        max_workers="auto",
        device=devices,
        job_memory_estimator=Mock(return_value=123),
    )

    # validation uses one worker per job, up to the thread limit
    assert exec_jobs.call_args.args[1] == min(len(shard_names), _max_threads())

    # conversion uses the value resolved from devices and memory estimates
    auto_max_workers.assert_called_once_with(devices, [123, 123])
    assert exec_jobs_dynamic.call_args.kwargs["max_workers"] == 3


@pytest.mark.parametrize("max_workers", ("fast", 0, -1, 1.5, None))
@patch.object(_CHECKPOINT_MODULE, "get_checkpoint_files")
def test_convert_checkpoint_rejects_invalid_max_workers(
    get_checkpoint_files, max_workers, tmp_path
):
    with pytest.raises(ValueError, match="max_workers must be"):
        convert_checkpoint(
            "source", tmp_path, Mock(), max_workers=max_workers, device="cpu"
        )

    get_checkpoint_files.assert_not_called()


def _save_tiny_shards(source: Path, num_shards: int) -> dict[str, torch.Tensor]:
    """Save one small tensor per shard, plus an index, and return all tensors"""
    source.mkdir()
    tensors, weight_map = {}, {}
    for index in range(num_shards):
        shard_name = f"model-{index}.safetensors"
        shard = {f"layers.{index}.weight": torch.randn(64, 64)}
        save_file(shard, source / shard_name)
        tensors.update(shard)
        weight_map.update({name: shard_name for name in shard})
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return tensors


def _assert_shards_equal(output: Path, tensors: dict[str, torch.Tensor]):
    converted = {}
    for index in range(len(tensors)):
        converted.update(load_file(output / f"model-{index}.safetensors"))
    assert converted.keys() == tensors.keys()
    for name, tensor in tensors.items():
        assert torch.equal(converted[name], tensor)


@requires_gpu
def test_convert_checkpoint_auto_max_workers_on_accelerators(tmp_path):
    """All available accelerators are passed to the scheduler, and workers are
    resolved from their real free memory. Shards are tiny, so the number of jobs
    is the binding limit"""
    num_shards = 4
    tensors = _save_tiny_shards(tmp_path / "source", num_shards)

    exec_jobs_dynamic = _CHECKPOINT_MODULE.exec_jobs_dynamic
    with patch.object(
        _CHECKPOINT_MODULE, "exec_jobs_dynamic", wraps=exec_jobs_dynamic
    ) as spy:
        convert_checkpoint(
            tmp_path / "source",
            tmp_path / "output",
            NoOpConverter(),
            max_workers="auto",
        )

    assert spy.call_args.kwargs["max_workers"] == num_shards
    assert len(spy.call_args.kwargs["devices"]) == torch.accelerator.device_count()
    _assert_shards_equal(tmp_path / "output", tensors)


@requires_gpu(2)
def test_convert_checkpoint_auto_max_workers_runs_on_each_device(tmp_path):
    """When two accelerators each only have room for one job at a time, "auto" uses
    one worker per accelerator and both accelerators run a job. Using two devices
    keeps the test independent of the CPU-based thread cap, which is at least 5"""
    accelerator = torch.accelerator.current_accelerator()
    devices = [torch.device(accelerator.type, index) for index in range(2)]
    free_memory = [torch.accelerator.memory.get_memory_info(dev)[0] for dev in devices]
    job_memory = int(0.9 * min(free_memory))
    if 2 * job_memory <= max(free_memory):
        pytest.skip("Free memory differs too much between accelerators")

    num_shards = len(devices) + 1
    tensors = _save_tiny_shards(tmp_path / "source", num_shards)

    exec_jobs_dynamic = _CHECKPOINT_MODULE.exec_jobs_dynamic
    run_job_on_device = _MEMORY_MODULE._run_job_on_device
    with (
        patch.object(
            _CHECKPOINT_MODULE, "exec_jobs_dynamic", wraps=exec_jobs_dynamic
        ) as spy,
        patch.object(
            _MEMORY_MODULE, "_run_job_on_device", wraps=run_job_on_device
        ) as run_job,
    ):
        convert_checkpoint(
            tmp_path / "source",
            tmp_path / "output",
            NoOpConverter(),
            max_workers="auto",
            device=devices,
            job_memory_estimator=lambda *_: job_memory,
        )

    assert spy.call_args.kwargs["max_workers"] == len(devices)
    assert {call.args[1] for call in run_job.call_args_list} == set(devices)
    _assert_shards_equal(tmp_path / "output", tensors)
