# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from contextlib import nullcontext

import pytest
import torch
from compressed_tensors.offload import disable_offloading, offload_module
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
    initialize_module_for_quantization,
)


class ControlledLinear(torch.nn.Linear):
    gate = None
    failure = None

    def forward(self, inputs):
        if self.gate is not None:
            entered, release = self.gate
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Forward was not released")
        self.observed_weight = self.weight.detach().clone()
        if self.failure is not None:
            raise self.failure("Forward failed")
        return super().forward(inputs)


def _assert_released():
    assert not OffloadCache.offloading_disabled
    assert not OffloadCache.onloading_disabled
    assert not OffloadCache.keep_onloaded_values
    assert getattr(OffloadCache, "_offloading_context_count", 0) == 0


@pytest.fixture(autouse=True)
def retention_state():
    _assert_released()
    try:
        yield
        _assert_released()
    finally:
        OffloadCache.offloading_disabled = False
        OffloadCache.onloading_disabled = False
        OffloadCache.keep_onloaded_values.clear()
        if hasattr(OffloadCache, "_offloading_context_count"):
            OffloadCache._offloading_context_count = 0


def _make_layer(storage, directory, activations=False, layer_cls=ControlledLinear):
    layer = layer_cls(4, 2, bias=False)
    raw = torch.tensor([[0.14, 0.26, -0.34, -0.46], [0.26, -0.14, 0.46, -0.34]])
    layer.weight.data.copy_(raw)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=8, type="int", strategy="channel"),
        input_activations=QuantizationArgs() if activations else None,
        output_activations=QuantizationArgs() if activations else None,
    )
    initialize_module_for_quantization(layer, scheme)
    for name, param in layer.named_parameters(recurse=False):
        if name.endswith("_scale"):
            param.data.fill_(0.1)
        elif name.endswith("_zero_point"):
            param.data.zero_()
    layer.quantization_status = QuantizationStatus.FROZEN
    if storage is not None:
        directory.mkdir(parents=True, exist_ok=True)
        offload_module(
            layer,
            "cpu",
            storage,
            **({"offload_dir": str(directory)} if storage == "disk" else {}),
        )
    inputs = torch.eye(4)
    quantized = torch.round(raw / 0.1).clamp(-128, 127) * 0.1
    expected = torch.nn.functional.linear(inputs, quantized)
    original = torch.nn.functional.linear(inputs, raw)
    assert not torch.equal(expected, original)
    return layer, inputs, raw, expected, original


def _concurrent_forwards(storage, first, fail_first, tmp_path, outer=False):
    layers = [_make_layer(storage, tmp_path / str(i)) for i in range(2)]
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    done = [threading.Event(), threading.Event()]
    outputs, errors = {}, {}
    for i, (layer, inputs, raw, expected, original) in enumerate(layers):
        with torch.no_grad():
            torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
            layer.quantization_enabled = False
            torch.testing.assert_close(layer(inputs), original, rtol=0, atol=0)
            layer.quantization_enabled = True
        layer.gate = entered[i], release[i]
    if fail_first:
        layers[first][0].failure = RuntimeError

    def worker(i):
        try:
            with torch.no_grad():
                outputs[i] = layers[i][0](layers[i][1])
        except BaseException as exc:
            errors[i] = exc
        finally:
            done[i].set()

    threads = [
        threading.Thread(target=worker, args=(i,), daemon=True) for i in range(2)
    ]
    try:
        with disable_offloading() if outer else nullcontext():
            cached = [layer.weight for layer, *_ in layers] if outer else []
            threads[0].start()
            assert entered[0].wait(5)
            threads[1].start()
            assert entered[1].wait(5)
            release[first].set()
            assert done[first].wait(5)
            retained_while_other_active = OffloadCache.offloading_disabled
            release[1 - first].set()
            assert done[1 - first].wait(5)
            for thread in threads:
                thread.join(5)
                assert not thread.is_alive()
            assert set(errors) == ({first} if fail_first else set())
            if fail_first:
                assert isinstance(errors[first], RuntimeError)
                assert str(errors[first]) == "Forward failed"
            for i, output in outputs.items():
                torch.testing.assert_close(output, layers[i][3], rtol=0, atol=0)
            assert retained_while_other_active
            for i, (layer, _, raw, _, _) in enumerate(layers):
                torch.testing.assert_close(layer.weight, raw, rtol=0, atol=0)
                if outer:
                    assert layer.weight is cached[i]
            assert OffloadCache.offloading_disabled == outer
        _assert_released()
    finally:
        for event in release:
            event.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(5)
                assert not thread.is_alive()


@pytest.mark.parametrize("storage", [None, "cpu", "disk"])
@pytest.mark.parametrize("first", [0, 1])
@pytest.mark.parametrize("fail_first", [False, True])
@pytest.mark.parametrize("repeat", range(3))
def test_concurrent_retention(storage, first, fail_first, repeat, tmp_path):
    _concurrent_forwards(storage, first, fail_first, tmp_path)


@pytest.mark.parametrize("fail_first", [False, True])
def test_concurrent_retention_preserves_outer_context(fail_first, tmp_path):
    _concurrent_forwards("disk", 0, fail_first, tmp_path, outer=True)


@pytest.mark.parametrize("storage", [None, "cpu", "disk"])
@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("failure", ["input", "weight", "output", "forward"])
def test_quantization_failure_releases_owned_retention(
    storage, outer, failure, tmp_path
):
    layer, inputs, raw, expected, _ = _make_layer(storage, tmp_path, activations=True)
    if failure == "forward":
        layer.failure = RuntimeError
    else:
        setattr(
            layer,
            f"{failure}_scale",
            torch.nn.Parameter(torch.full((3,), 0.1), requires_grad=False),
        )
    with disable_offloading() if outer else nullcontext():
        cached_weight = layer.weight if outer else None
        with pytest.raises(RuntimeError):
            layer(inputs)
        torch.testing.assert_close(layer.weight, raw, rtol=0, atol=0)
        assert OffloadCache.offloading_disabled == outer
        if outer:
            assert layer.weight is cached_weight
        else:
            _assert_released()
        if failure in ("forward", "output"):
            torch.testing.assert_close(
                layer.observed_weight,
                torch.round(raw / 0.1).clamp(-128, 127) * 0.1,
                rtol=0,
                atol=0,
            )
        if failure == "forward":
            layer.failure = None
        else:
            shape = (2, 1) if failure == "weight" else (1,)
            setattr(
                layer,
                f"{failure}_scale",
                torch.nn.Parameter(torch.full(shape, 0.1), requires_grad=False),
            )
        with torch.no_grad():
            torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
        torch.testing.assert_close(layer.weight, raw, rtol=0, atol=0)
    _assert_released()


@pytest.mark.parametrize("outer", [False, True])
def test_cancelled_forward_releases_retention(outer, tmp_path):
    layer, inputs, raw, expected, _ = _make_layer("disk", tmp_path)
    layer.failure = KeyboardInterrupt
    with disable_offloading() if outer else nullcontext():
        with pytest.raises(KeyboardInterrupt):
            layer(inputs)
        torch.testing.assert_close(layer.weight, raw, rtol=0, atol=0)
        assert OffloadCache.offloading_disabled == outer
        layer.failure = None
        with torch.no_grad():
            torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
    _assert_released()


@pytest.mark.parametrize("storage", [None, "cpu", "disk"])
@pytest.mark.parametrize(
    "mode",
    [
        "disabled",
        "no_scheme",
        "no_status",
        "compressed",
        "decompressed",
        "activation_only",
    ],
)
def test_retention_preserves_quantization_modes(storage, mode, tmp_path):
    layer, inputs, raw, _, original = _make_layer(storage, tmp_path)
    if mode == "disabled":
        layer.quantization_enabled = False
    elif mode == "no_scheme":
        del layer.quantization_scheme
    elif mode == "no_status":
        del layer.quantization_status
    elif mode in ("compressed", "decompressed"):
        layer.quantization_status = QuantizationStatus(mode)
    else:
        scheme = QuantizationScheme(
            targets=["Linear"], input_activations=QuantizationArgs()
        )
        initialize_module_for_quantization(layer, scheme)
        layer.input_scale = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=False)
        layer.input_zero_point = torch.nn.Parameter(
            torch.tensor([0], dtype=torch.int8), requires_grad=False
        )
        layer.quantization_status = QuantizationStatus.FROZEN
    for _ in range(2):
        with torch.no_grad():
            torch.testing.assert_close(layer(inputs), original, rtol=0, atol=0)
        torch.testing.assert_close(layer.weight, raw, rtol=0, atol=0)
        _assert_released()


@pytest.mark.parametrize("storage", [None, "cpu", "disk"])
def test_same_module_reentry_preserves_weight_and_context(storage, tmp_path):
    class ReentrantLinear(torch.nn.Linear):
        active = False

        def forward(self, inputs):
            if self.active:
                return super().forward(inputs)
            self.active = True
            try:
                nested = self(inputs)
                return super().forward(inputs) + nested
            finally:
                self.active = False

    layer, inputs, raw, expected, _ = _make_layer(
        storage, tmp_path, layer_cls=ReentrantLinear
    )
    with torch.no_grad():
        torch.testing.assert_close(layer(inputs), expected * 2, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight, raw, rtol=0, atol=0)
    _assert_released()


def test_shared_weight_sequential_forwards_preserve_parameter(tmp_path):
    first, inputs, raw, expected, _ = _make_layer(None, tmp_path / "first")
    second, *_ = _make_layer(None, tmp_path / "second")
    second.weight = first.weight
    shared = first.weight
    with disable_offloading():
        for layer in (first, second, first):
            with torch.no_grad():
                torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
            assert layer.weight is shared
            torch.testing.assert_close(shared, raw, rtol=0, atol=0)
    _assert_released()


@pytest.mark.parametrize("storage", [None, "cpu", "disk"])
def test_nested_child_forward_retains_parent_weight(storage, tmp_path):
    class ParentLinear(torch.nn.Linear):
        def forward(self, inputs):
            self.child_output = self.child(inputs)
            assert OffloadCache.offloading_disabled
            return super().forward(inputs)

    parent, inputs, raw, expected, _ = _make_layer(
        storage, tmp_path / "parent", layer_cls=ParentLinear
    )
    child, _, child_raw, child_expected, _ = _make_layer(storage, tmp_path / "child")
    parent.child = child
    with torch.no_grad():
        torch.testing.assert_close(parent(inputs), expected, rtol=0, atol=0)
    torch.testing.assert_close(parent.child_output, child_expected, rtol=0, atol=0)
    torch.testing.assert_close(parent.weight, raw, rtol=0, atol=0)
    torch.testing.assert_close(child.weight, child_raw, rtol=0, atol=0)
    _assert_released()


@pytest.mark.parametrize("storage", [None, "cpu", "disk"])
@pytest.mark.parametrize("kind", ["empty", "singleton", "noncontiguous", "nan", "inf"])
def test_retention_preserves_input_boundaries(storage, kind, tmp_path):
    layer, _, raw, _, _ = _make_layer(storage, tmp_path)
    if kind == "empty":
        inputs = torch.empty(0, 4)
    elif kind == "singleton":
        inputs = torch.ones(1, 4)
    elif kind == "noncontiguous":
        inputs = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
        assert not inputs.is_contiguous()
    else:
        inputs = torch.ones(1, 4)
        inputs[0, 0] = float(kind)
    quantized = torch.round(raw / 0.1).clamp(-128, 127) * 0.1
    expected = torch.nn.functional.linear(inputs, quantized)
    with torch.no_grad():
        torch.testing.assert_close(
            layer(inputs), expected, rtol=0, atol=0, equal_nan=True
        )
    torch.testing.assert_close(layer.weight, raw, rtol=0, atol=0)
    _assert_released()


def test_preexisting_retention_flag_and_cache_remain_owned_by_caller():
    key = torch.tensor([1.0])
    value = torch.tensor([2.0])
    OffloadCache.offloading_disabled = True
    OffloadCache.keep_onloaded_values[key] = value
    try:
        with disable_offloading():
            with disable_offloading():
                assert OffloadCache.keep_onloaded_values[key] is value
        assert OffloadCache.offloading_disabled
        assert OffloadCache.keep_onloaded_values[key] is value
        assert getattr(OffloadCache, "_offloading_context_count", 0) == 0
    finally:
        OffloadCache.offloading_disabled = False
        OffloadCache.keep_onloaded_values.clear()
