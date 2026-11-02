# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import math

import pytest
import torch
from compressed_tensors.offload.cache import disk_utils, utils
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.cache.disk_utils import _opened, disk_load_context
from compressed_tensors.offload.cache.utils import load_disk_tensor_from_offload
from safetensors import SafetensorError
from safetensors.torch import save_file
from tests.testing_utils import requires_gpu


@pytest.fixture(autouse=True)
def isolated_disk_index(monkeypatch):
    monkeypatch.setattr(DiskCache, "index", {})


@pytest.fixture
def shard(tmp_path):
    path = tmp_path / "shard.safetensors"
    save_file(
        {"a": torch.ones(3), "weight": torch.arange(16).float().reshape(4, 4)}, path
    )
    return path


def _load(path, name="weight", dtype=torch.float32):
    return load_disk_tensor_from_offload(
        {
            "safetensors_file": str(path),
            "weight_name": name,
            "dtype": str(dtype).removeprefix("torch."),
        },
        device="cpu",
        pin_memory=True,
    )


def _unexpected_copy(*args, **kwargs):
    pytest.fail("Same-dtype disk reads must not use Tensor.copy_")


@pytest.fixture
def emulate_pinned_allocation(monkeypatch):
    allocations = []

    def unpinned(factory):
        def allocate(*args, **kwargs):
            allocations.append(kwargs.pop("pin_memory", False))
            return factory(*args, **kwargs)

        return allocate

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    for name in ("empty", "empty_like"):
        monkeypatch.setattr(torch, name, unpinned(getattr(torch, name)))
    return allocations


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(), (0, 3), (11,), (4, 7)])
@pytest.mark.parametrize("dtype", utils.str_to_torch_dtype.values())
def test_raw_read_preserves_bytes(
    tmp_path, monkeypatch, emulate_pinned_allocation, shape, dtype
):
    tensor = torch.arange(math.prod(shape)).reshape(shape).to(dtype)
    shard = tmp_path / "shard.safetensors"
    save_file({"a": torch.ones(9), "weight": tensor, "z": torch.zeros(5)}, shard)

    monkeypatch.setattr(torch.Tensor, "copy_", _unexpected_copy)
    result = _load(shard, dtype=dtype)
    assert result.shape == tensor.shape
    assert result.dtype == dtype
    assert torch.equal(
        result.reshape(-1).view(torch.uint8), tensor.reshape(-1).view(torch.uint8)
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "target_dtype,mapped",
    [
        (torch.float32, True),
        (torch.bfloat16, True),
        (torch.int8, True),
        (torch.float32, False),
    ],
)
@pytest.mark.parametrize("requires_grad", [False, True])
def test_pinned_stage_preserves_parameter_and_cast(
    tmp_path,
    monkeypatch,
    emulate_pinned_allocation,
    target_dtype,
    mapped,
    requires_grad,
):
    parameter = torch.nn.Parameter(
        torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
        requires_grad=requires_grad,
    )
    parameter.test_metadata = "preserve"
    cache = DiskCache("cuda", offload_dir=tmp_path)
    offloaded = cache.offload(parameter)
    offloaded.requires_grad_(requires_grad)
    entry = DiskCache.index[offloaded]
    save_file({"weight": parameter.data.float()}, entry["safetensors_file"])
    entry["dtype"] = str(target_dtype).removeprefix("torch.")
    if not mapped:
        monkeypatch.delitem(utils.str_to_torch_dtype, "F32")
    emulate_pinned_allocation.clear()

    staged = cache.stage(offloaded, pin_memory=True)

    assert isinstance(staged, torch.nn.Parameter)
    assert staged.test_metadata == parameter.test_metadata
    assert staged.requires_grad == (requires_grad and target_dtype.is_floating_point)
    assert staged.dtype == target_dtype
    assert torch.equal(staged, parameter.detach().to(target_dtype))
    assert emulate_pinned_allocation == [True]


@pytest.mark.unit
@pytest.mark.parametrize("onload_device", ["cpu", torch.device("cpu"), "cuda"])
@pytest.mark.parametrize("accelerator_available", [False, True])
def test_stage_without_pinning_uses_existing_loader(
    tmp_path, monkeypatch, onload_device, accelerator_available
):
    cache = DiskCache(onload_device, offload_dir=tmp_path)
    tensor = torch.arange(10)
    offloaded = cache.offload(tensor)
    monkeypatch.setattr(
        torch.accelerator, "is_available", lambda: accelerator_available
    )

    def unexpected_raw_open(*args, **kwargs):
        pytest.fail("Unpinned staging must use the existing safetensors loader")

    monkeypatch.setattr(disk_utils, "open", unexpected_raw_open, raising=False)
    assert torch.equal(cache.stage(offloaded), tensor)
    assert cache.stage(None, pin_memory=True) is None
    if not accelerator_available or torch.device(onload_device).type == "cpu":
        assert torch.equal(cache.stage(offloaded, pin_memory=True), tensor)


@pytest.mark.unit
def test_raw_reader_reused_and_closed(shard, emulate_pinned_allocation):
    with disk_load_context():
        assert torch.equal(_load(shard, "a"), torch.ones(3))
        with _opened(str(shard), "cpu", raw=True) as first:
            with disk_load_context(), _opened(str(shard), "cpu", raw=True) as second:
                assert first is second
                assert torch.equal(_load(shard), torch.arange(16).float().reshape(4, 4))
        assert not first[0].closed
    assert first[0].closed


@pytest.mark.unit
@pytest.mark.parametrize("truncated", [False, True])
def test_raw_reader_handles_short_reads_and_eof(
    shard, monkeypatch, emulate_pinned_allocation, truncated
):
    data = shard.read_bytes()

    class ShortReads(io.BytesIO):
        def readinto(self, buffer):
            return super().readinto(buffer[:3])

    monkeypatch.setattr(
        disk_utils,
        "open",
        lambda *args, **kwargs: ShortReads(data[:-1] if truncated else data),
        raising=False,
    )
    if truncated:
        with pytest.raises(EOFError, match="weight"):
            _load(shard)
    else:
        assert torch.equal(_load(shard), torch.arange(16).float().reshape(4, 4))


@pytest.mark.unit
def test_raw_reader_validates_safetensors(tmp_path):
    shard = tmp_path / "invalid.safetensors"
    shard.write_bytes(b"invalid file")
    with pytest.raises(SafetensorError):
        with _opened(str(shard), "cpu", raw=True):
            pytest.fail("Invalid safetensors files must be rejected")


@pytest.mark.unit
def test_raw_reader_evicted_before_update(tmp_path, emulate_pinned_allocation):
    cache = DiskCache("cuda", offload_dir=tmp_path)
    offloaded = cache.offload(torch.ones(3))
    with disk_load_context():
        assert torch.equal(cache.stage(offloaded, pin_memory=True), torch.ones(3))
        cache.update_offload(offloaded, torch.full((3,), 5.0))
        assert torch.equal(
            cache.stage(offloaded, pin_memory=True), torch.full((3,), 5.0)
        )


@pytest.mark.unit
@requires_gpu
@pytest.mark.parametrize("target_dtype", [torch.float32, torch.bfloat16, torch.int8])
def test_direct_read_is_pinned_after_dtype_cast(shard, monkeypatch, target_dtype):
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    if target_dtype == tensor.dtype:
        monkeypatch.setattr(torch.Tensor, "copy_", _unexpected_copy)
    result = _load(shard, dtype=target_dtype)
    assert result.is_pinned()
    assert torch.equal(result, tensor.to(target_dtype))
