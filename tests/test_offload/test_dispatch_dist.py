# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import copy
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.distributed import is_source_process, set_source_process
from compressed_tensors.offload import disable_onloading
from compressed_tensors.offload.cache import dist_batch
from compressed_tensors.offload.cache.dist_batch import batch_offload_sync
from compressed_tensors.offload.dispatch import dispatch_with_map
from compressed_tensors.offload.module import offload_module
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


CPU = torch.device("cpu")


def _linears(num: int, meta: bool = False) -> torch.nn.ModuleList:
    """Same seeded weights on every rank; `meta` mimics non-source ranks at load"""
    torch.manual_seed(0)
    with torch.device("meta" if meta else "cpu"):
        return torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(num))


def _assert_matches(model: torch.nn.Module, expected: torch.nn.Module):
    with torch.no_grad():
        for name, tensor in expected.state_dict().items():
            actual = model.get_submodule(name.rpartition(".")[0])
            actual = getattr(actual, name.rpartition(".")[2])
            assert actual.dtype == tensor.dtype, name
            assert torch.equal(actual.cpu(), tensor), name


def _assert_offloads_match_across_ranks(model: torch.nn.Module):
    """Offloaded tensors have the same gradient flags and strides on every rank"""
    with disable_onloading():
        local = {
            name: (tensor.requires_grad, tensor.stride())
            for name, tensor in model.state_dict(keep_vars=True).items()
        }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    assert all(flags == gathered[0] for flags in gathered)


@contextlib.contextmanager
def _count_collectives():
    with (
        patch(
            "torch.distributed.broadcast_object_list",
            wraps=dist.broadcast_object_list,
        ) as broadcast_object_list,
        patch("torch.distributed.barrier", wraps=dist.barrier) as barrier,
    ):
        yield broadcast_object_list, barrier


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_exchanges_metadata_once(accel_device, offload_folder):
    for offload_device in (CPU, "disk"):
        model = _linears(6, meta=not is_source_process())
        device_map = {str(i): (accel_device, offload_device) for i in range(6)}

        with _count_collectives() as (broadcast_object_list, barrier):
            dispatch_with_map(
                model, device_map, offload_dir=offload_folder, show_progress=False
            )

        # previously one of each per tensor (12 tensors)
        assert broadcast_object_list.call_count == 1
        assert barrier.call_count == 1
        _assert_matches(model, _linears(6))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_mixed_offload_devices(accel_device, offload_folder):
    model = _linears(4, meta=not is_source_process())
    device_map = {
        "0": (accel_device, CPU),
        "1": (accel_device, "disk"),
        "2": (accel_device, accel_device),  # tensor data broadcast, not batched
        "3": (accel_device, CPU),
    }

    with _count_collectives() as (broadcast_object_list, _):
        dispatch_with_map(
            model, device_map, offload_dir=offload_folder, show_progress=False
        )

    assert broadcast_object_list.call_count == 1
    _assert_matches(model, _linears(4))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_tied_weights(accel_device):
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.head = torch.nn.Linear(4, 8, bias=False)
            self.head.weight = self.embed.weight

    torch.manual_seed(0)
    with torch.device("meta" if not is_source_process() else "cpu"):
        model = Tied()
    torch.manual_seed(0)
    expected = Tied()

    device_map = {"embed": (accel_device, CPU), "head": (accel_device, CPU)}
    dispatch_with_map(model, device_map, show_progress=False)

    # the tie is kept on every rank: one shared storage, not two
    with disable_onloading():
        embed, head = model.embed.weight, model.head.weight
        assert embed.untyped_storage().data_ptr() == head.untyped_storage().data_ptr()

    _assert_matches(model, expected)

    # every rank finishes checking the original values before the source mutates
    # the shared storage
    dist.barrier()

    # an in-place update on the source reaches both aliases on every rank
    if is_source_process():
        with disable_onloading(), torch.no_grad():
            model.embed.weight.fill_(1.0)
    dist.barrier()

    with torch.no_grad():
        expected.embed.weight.fill_(1.0)
    _assert_matches(model, expected)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_rebuilds_dtype_and_shape(accel_device, offload_folder):
    class Rotary(torch.nn.Module):
        def __init__(self, size: int, dtype: torch.dtype):
            super().__init__()
            self.register_buffer("inv_freq", torch.arange(size, dtype=dtype))

    for offload_device in (CPU, "disk"):
        # non-source ranks may init buffers differently than the checkpoint
        if is_source_process():
            model = Rotary(8, torch.float32)
        else:
            with torch.device("meta"):
                model = Rotary(4, torch.float64)

        device_map = {"": (accel_device, offload_device)}
        dispatch_with_map(
            model, device_map, offload_dir=offload_folder, show_progress=False
        )

        _assert_matches(model, Rotary(8, torch.float32))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_modules_missing_on_some_ranks(accel_device, offload_folder):
    for offload_device in (CPU, "disk"):
        model = _linears(3, meta=not is_source_process())
        if not is_source_process():
            model[0] = None  # e.g. a routed expert this rank does not own

        device_map = {str(i): (accel_device, offload_device) for i in range(3)}
        with _count_collectives() as (broadcast_object_list, _):
            dispatch_with_map(
                model, device_map, offload_dir=offload_folder, show_progress=False
            )

        assert broadcast_object_list.call_count == 1
        expected = _linears(3)
        if not is_source_process():
            expected[0] = None
        _assert_matches(model, expected)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_mixed_map_with_missing_modules(accel_device, offload_folder):
    model = _linears(4, meta=not is_source_process())
    if not is_source_process():
        model[0] = None
        model[1] = None

    # cpu and disk modules are missing on other ranks; accelerator modules are not,
    # since their per-tensor data broadcasts need every rank
    device_map = {
        "0": (accel_device, CPU),
        "1": (accel_device, "disk"),
        "2": (accel_device, accel_device),
        "3": (accel_device, CPU),
    }
    with _count_collectives() as (broadcast_object_list, barrier):
        dispatch_with_map(
            model, device_map, offload_dir=offload_folder, show_progress=False
        )

    assert broadcast_object_list.call_count == 1
    assert barrier.call_count == 1
    expected = _linears(4)
    if not is_source_process():
        expected[0] = None
        expected[1] = None
    _assert_matches(model, expected)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_offload_after_dispatch_syncs_per_tensor(accel_device):
    model = _linears(2, meta=not is_source_process())
    device_map = {str(i): (accel_device, CPU) for i in range(2)}
    dispatch_with_map(model, device_map, show_progress=False)

    # outside of dispatch, new offloads are still exchanged immediately
    with _count_collectives() as (broadcast_object_list, barrier):
        model[0]._parameters["extra"] = torch.full((3,), 7.0)

    assert broadcast_object_list.call_count == 1
    assert barrier.call_count == 1
    assert torch.equal(model[0].extra.cpu(), torch.full((3,), 7.0))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_replica_without_local_modules(accel_device, offload_folder):
    for offload_device in (CPU, "disk"):
        model = _linears(2, meta=not is_source_process())
        if not is_source_process():
            model[0] = None
            model[1] = None

        device_map = {str(i): (accel_device, offload_device) for i in range(2)}
        with _count_collectives() as (broadcast_object_list, barrier):
            dispatch_with_map(
                model, device_map, offload_dir=offload_folder, show_progress=False
            )

        # a rank with nothing to rebuild still joins the single exchange
        assert broadcast_object_list.call_count == 1
        assert barrier.call_count == 1
        if is_source_process():
            _assert_matches(model, _linears(2))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_rejects_nested_batch(accel_device):
    model = _linears(2, meta=not is_source_process())
    device_map = {str(i): (accel_device, CPU) for i in range(2)}

    with batch_offload_sync():
        with pytest.raises(RuntimeError, match="cannot be nested"):
            dispatch_with_map(model, device_map, show_progress=False)

    assert dist_batch._active_batch is None


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_failed_batch_resets_state(accel_device):
    model = _linears(2, meta=not is_source_process())

    # without module names, both modules record `weight` under the same key
    with pytest.raises(ValueError, match="already recorded"):
        with batch_offload_sync():
            for module in model:
                offload_module(module, accel_device, CPU)

    assert dist_batch._active_batch is None

    # a later, independent dispatch is unaffected
    model = _linears(2, meta=not is_source_process())
    device_map = {str(i): (accel_device, CPU) for i in range(2)}
    dispatch_with_map(model, device_map, show_progress=False)
    _assert_matches(model, _linears(2))


def _test_non_default_source(accel_device, offload_folder):
    with set_source_process(1):
        for offload_device in (CPU, "disk"):
            model = _linears(4, meta=not is_source_process())
            device_map = {str(i): (accel_device, offload_device) for i in range(4)}

            with _count_collectives() as (broadcast_object_list, barrier):
                dispatch_with_map(
                    model, device_map, offload_dir=offload_folder, show_progress=False
                )

            assert broadcast_object_list.call_count == 1
            assert barrier.call_count == 1
            _assert_matches(model, _linears(4))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_non_default_source(accel_device, offload_folder):
    _test_non_default_source(accel_device, offload_folder)


@pytest.mark.unit
@requires_gpu(3)
@torchrun(world_size=3, init_dist=True)
def test_dispatch_three_ranks_non_default_source(accel_device, offload_folder):
    _test_non_default_source(accel_device, offload_folder)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_shared_empty_buffer(accel_device):
    class SharedEmpty(torch.nn.Module):
        def __init__(self):
            super().__init__()
            empty = torch.empty(0)
            self.register_buffer("a", empty)
            self.register_buffer("b", empty)

    with torch.device("cpu" if is_source_process() else "meta"):
        model = SharedEmpty()

    # the second offload of the shared zero-byte storage must not invalidate the
    # first one's handle before other ranks open it
    dispatch_with_map(model, {"": (accel_device, CPU)}, show_progress=False)

    assert model.a.shape == model.b.shape == (0,)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_keeps_identity_of_cpu_tensors(accel_device):
    # tensors already on cpu are rebuilt in place on every rank, so references to
    # them stay valid (e.g. across a to_accelerate / from_accelerate round trip)
    model = _linears(2)
    before = dict(model.named_parameters())
    device_map = {str(i): (accel_device, CPU) for i in range(2)}
    dispatch_with_map(model, device_map, show_progress=False)

    with disable_onloading():
        after = dict(model.named_parameters())
    assert all(after[name] is tensor for name, tensor in before.items())
    _assert_matches(model, _linears(2))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_replica_offloads_match_source(accel_device, offload_folder):
    for offload_device in (CPU, "disk"):
        # every rank holds real cpu tensors: parameters which require grad, and a
        # non-contiguous buffer
        model = _linears(2)
        model[0].register_buffer("transposed", torch.arange(6.0).reshape(2, 3).t())
        device_map = {str(i): (accel_device, offload_device) for i in range(2)}
        dispatch_with_map(
            model, device_map, offload_dir=offload_folder, show_progress=False
        )

        _assert_offloads_match_across_ranks(model)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_frees_replica_tensors_before_onloading(accel_device):
    torch.manual_seed(0)
    expected = torch.nn.ModuleList(
        torch.nn.Linear(1024, 2048, bias=False) for _ in range(4)
    )
    module_bytes = expected[0].weight.nbytes

    # every rank starts with modules 0-1 on the accelerator and 2-3 on cpu. 0-1 are
    # offloaded to cpu, then 2-3 are moved onto the accelerator
    model = copy.deepcopy(expected)
    model[0].to(accel_device)
    model[1].to(accel_device)
    device_map = {
        "0": (accel_device, CPU),
        "1": (accel_device, CPU),
        "2": (accel_device, accel_device),
        "3": (accel_device, accel_device),
    }

    torch.accelerator.synchronize()
    start = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    dispatch_with_map(model, device_map, show_progress=False)
    peak = torch.accelerator.max_memory_allocated()

    # the accelerator copies of 0-1 are released as they are offloaded, so moving
    # 2-3 onto the accelerator does not add to the starting footprint
    assert peak - start < module_bytes
    _assert_matches(model, expected)
    _assert_offloads_match_across_ranks(model)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_records_empty_views_as_meta(accel_device, offload_folder):
    class EmptyView(torch.nn.Module):
        def __init__(self, device: torch.device):
            super().__init__()
            self.register_buffer("view", torch.empty(1024, device=device)[:0])

    recorded = []
    original_complete = dist_batch.OffloadBatch.complete

    def complete(batch: dist_batch.OffloadBatch):
        recorded.extend(tensor for *_, tensor in batch.pending)
        original_complete(batch)

    # an empty view on a device other than the offload device is recorded as a meta
    # copy on non-source ranks, so the batch doesn't keep its allocation alive
    for offload_device, view_device in (("disk", CPU), (CPU, accel_device)):
        recorded.clear()
        model = EmptyView(view_device)
        with patch.object(dist_batch.OffloadBatch, "complete", complete):
            dispatch_with_map(
                model,
                {"": (accel_device, offload_device)},
                offload_dir=offload_folder,
                show_progress=False,
            )

        assert len(recorded) == (0 if is_source_process() else 1)
        assert all(tensor.is_meta for tensor in recorded)
        with disable_onloading():
            offloaded = model.view
        assert offloaded.shape == (0,)
        assert offloaded.device == (
            torch.device("meta") if offload_device == "disk" else CPU
        )


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_rebuilds_accelerator_tensors_with_other_shape(accel_device):
    # a non-source rank's accelerator weight with a different shape than the
    # source's is rebuilt from the source. Like any tensor moved off the
    # accelerator, its offload doesn't require grad, so every rank matches
    model = _linears(1).to(accel_device)
    if not is_source_process():
        model[0].weight = torch.nn.Parameter(torch.zeros(2, 4, device=accel_device))

    dispatch_with_map(model, {"0": (accel_device, CPU)}, show_progress=False)

    _assert_matches(model, _linears(1))
    _assert_offloads_match_across_ranks(model)
    with disable_onloading():
        assert not model[0].weight.requires_grad
