# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Literal

import torch
import tqdm
from compressed_tensors.entrypoints.convert.convert_file import (
    convert_file,
    validate_file,
    write_checkpoint_quantization_config,
)
from compressed_tensors.entrypoints.convert.converters import (
    Converter,
    build_inverse_weight_maps,
)
from compressed_tensors.entrypoints.convert.memory import (
    _snapshot_free,
    estimate_job_memory,
    exec_jobs_dynamic,
)
from compressed_tensors.utils.safetensors_load import (
    InverseWeightMap,
    get_checkpoint_files,
    get_weight_map,
    is_weights_file,
    update_safetensors_index,
)
from loguru import logger


__all__ = ["convert_checkpoint", "exec_jobs"]

# On cpu, measured host memory use of concurrent conversion jobs exceeded the sum of
# their meta-tensor estimates by up to ~1.5x, so cpu jobs reserve this much more
_CPU_MEMORY_MARGIN = 1.5


def convert_checkpoint(
    model_stub: str | os.PathLike,
    save_directory: str | os.PathLike,
    converter: Converter | list[Converter],
    max_workers: int | Literal["auto"] = "auto",
    device: str | torch.device | list[str | torch.device] | None = None,
    job_memory_estimator: Callable[
        [InverseWeightMap, list[Converter]], int
    ] = estimate_job_memory,
):
    """
    Convert a model checkpoint to either:
    - its equivalent quantized format in compressed-tensors
    - the unquantized format
    without loading it up in memory, instead operating directly on the model
    safetensors files. This entrypoint operates on a model stub or folder containing
    weights saved in safetensors files, and updates the corresponding
    quantization_config field in the config.json. All additional files will be
    copied to new checkpoint.

    :param model_stub: huggingface model hub or path to local weights files
    :param save_directory: new checkpoint will be saved in this directory.
    :param max_workers: number of worker threads to process files with. If
        "auto" (default), the number of workers is chosen from the number of
        safetensors files, the estimated memory of each conversion job, the free
        memory of each accelerator, and the number of CPUs available to this
        process (1 on cpu). For accelerators, host memory is not taken into
        account, so pass a smaller number of workers if conversion runs out of
        host memory. On cpu, jobs run concurrently only while their estimated
        memory fits in the host memory available to this process
    :param device: device or devices on which to run conversion. When omitted,
        all available accelerator devices are used, falling back to CPU when no
        accelerator is available.
    :param job_memory_estimator: callable returning the estimated memory in bytes
        for a conversion job, given its inverse weight map and the converters to
        apply. Defaults to a meta-tensor profiler. On cpu, estimates are scaled
        by a safety margin. Conversion raises before any file is converted if a
        job's estimate exceeds the free memory of every device
    :param converter: single converter or list of converters to apply
        in order, e.g. a dequantizer followed by a re-quantizer
    """
    if max_workers != "auto" and not (isinstance(max_workers, int) and max_workers > 0):
        raise ValueError(
            f"max_workers must be a positive integer or 'auto', got {max_workers!r}"
        )

    converters = converter if isinstance(converter, list) else [converter]
    devices = _resolve_devices(device)

    # get all model_files for checkpoint
    model_files = get_checkpoint_files(model_stub)

    weight_map = get_weight_map(model_files)

    # Build inverse_weight_maps, so that each job knows how to load up every necessary
    # weight and its dependencies
    inverse_weight_maps = build_inverse_weight_maps(
        weight_map=weight_map,
        model_files=model_files,
        converters=converters,
    )
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    # Build validation/conversion jobs, copy over any other file
    validate_jobs = []
    convert_jobs = []
    for shard_name, resolved_path in model_files.items():
        save_path = Path(save_directory) / shard_name
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if shard_name.endswith("safetensors"):
            if shard_name not in inverse_weight_maps:
                raise ValueError(
                    f"Could not find inverse_weight_map for shard {shard_name}"
                )
            validate_jobs.append(
                (validate_file, inverse_weight_maps[shard_name], converters)
            )
            convert_jobs.append(
                (convert_file, inverse_weight_maps[shard_name], save_path, converters)
            )

        else:
            if is_weights_file(shard_name):
                logger.warning(f"Skip processing for weights file {shard_name}")
            if str(resolved_path) != str(save_path):
                logger.debug(f"Copying {shard_name} -> {save_path}")
                shutil.copyfile(resolved_path, save_path)

    # Validate before long-running procssing job. Validation runs on meta tensors,
    # so it uses as many workers as possible, regardless of max_workers
    num_validate_workers = max(1, min(len(validate_jobs), _max_threads()))
    exec_jobs(validate_jobs, num_validate_workers, desc="Validating")

    # Process weights, accumulating total bytes used and the new weight_map.
    # The same scheduler drives cpu and accelerator runs: jobs are admitted while
    # their memory estimates fit in free device memory (host memory on cpu)
    total_size = 0
    weight_map = dict()
    callable_jobs = [partial(job[0], *job[1:]) for job in convert_jobs]
    memory_estimates = [
        job_memory_estimator(job[1], converters) for job in convert_jobs
    ]
    if all(dev.type == "cpu" for dev in devices):
        memory_estimates = [
            int(estimate * _CPU_MEMORY_MARGIN) for estimate in memory_estimates
        ]
    if max_workers == "auto":
        max_workers = _auto_max_workers(devices, memory_estimates)
    convert_results = exec_jobs_dynamic(
        jobs=callable_jobs,
        devices=devices,
        max_workers=max_workers,
        memory_estimates=memory_estimates,
        desc="Converting",
    )
    for _total_size, _weight_map in convert_results:
        total_size += _total_size
        weight_map.update(_weight_map)

    # Update config and safetensors index
    write_checkpoint_quantization_config(save_directory, converters)
    update_safetensors_index(save_directory, total_size, weight_map)


def _resolve_devices(
    device: str | torch.device | list[str | torch.device] | None,
) -> list[torch.device]:
    """Resolve explicit devices or auto-detect all available accelerators."""
    if device is None:
        accelerator = torch.accelerator.current_accelerator(check_available=True)
        if accelerator is None:
            devices = [torch.device("cpu")]
        else:
            devices = [
                torch.device(accelerator.type, index)
                for index in range(torch.accelerator.device_count())
            ]
    else:
        devices = device if isinstance(device, list) else [device]
        if not devices:
            raise ValueError("The device list cannot be empty.")
        devices = [torch.device(dev) for dev in devices]
    return devices


def _max_threads() -> int:
    """
    Thread limit for conversion, following ThreadPoolExecutor's default of
    min(32, cpus + 4), but only counting the CPUs this process may run on, since
    containers and job schedulers often pin processes to a subset of the host's CPUs
    """
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS and Windows
        cpu_count = os.cpu_count() or 1
    return min(32, cpu_count + 4)


def _auto_max_workers(
    devices: list[torch.device],
    memory_estimates: list[int],
) -> int:
    """
    Choose the number of conversion workers for max_workers="auto".

    Concurrency is bounded by
    - the number of conversion jobs (one per safetensors file)
    - how many copies of the largest job fit in the devices' free memory at once
    - a thread limit based on the CPUs available to this process (see
      `_max_threads`), since jobs also spend time on disk reads and host-side
      serialization

    For accelerators, host memory is not taken into account. On cpu, 1 is
    returned: an explicit max_workers runs cpu jobs concurrently within the host
    memory budget, but a higher default still needs measurements across more
    workloads and machines.

    :param devices: devices on which jobs will be scheduled
    :param memory_estimates: estimated peak device memory of each job, in bytes
    :returns: number of workers to pass to `exec_jobs_dynamic`
    """
    num_jobs = len(memory_estimates)
    if num_jobs <= 1 or all(dev.type == "cpu" for dev in devices):
        return 1

    # Assuming every job needs as much memory as the largest one, how many jobs fit
    # in the free memory of all accelerators at once
    largest_job = max(memory_estimates)
    if largest_job > 0:
        free_memory = _snapshot_free(devices)
        device_capacity = sum(free // largest_job for free in free_memory.values())
    else:
        device_capacity = num_jobs

    max_threads = _max_threads()
    max_workers = max(1, min(num_jobs, device_capacity, max_threads))
    logger.info(
        f"Using max_workers={max_workers} (jobs={num_jobs}, "
        f"device_capacity={device_capacity}, max_threads={max_threads})"
    )
    return max_workers


def exec_jobs(
    jobs: list[tuple[Callable, ...]], max_workers: int = 1, desc: str = "Executing Jobs"
) -> list:
    """
    Execute jobs in parallel, using ThreadPoolExecutor

    :param jobs: list of tuples, the first entry of which is the callable,
        and the remaining elements are the inputs args to the callable
    :param max_workers: number of workers to use
    :param desc: tqdm description
    """
    results = []

    # For easier debugging, don't run single-threaded jobs via ThreadPoolExecutor
    if max_workers == 1:
        for job in tqdm.tqdm(jobs, desc=desc):
            results.append(job[0](*job[1:]))
        return results

    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(*job) for job in jobs]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc=desc):
            results.append(future.result())

    return results
