# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import compressed_tensors.offload as offload
import compressed_tensors.offload.module as module_offload


class _FakeOffloadCache(dict):
    pass


class _IndividualExpert(torch.nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset

    def forward(self, inputs):
        return inputs + self.offset


class _IndividualExpertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList(
            [_IndividualExpert(1.0), _IndividualExpert(2.0)]
        )


def test_subgraph_onload_skips_unoffloaded_modules(monkeypatch):
    root = torch.nn.Module()
    root.child = torch.nn.Linear(4, 4)
    root._parameters = _FakeOffloadCache()

    calls = []

    def fake_get_cache_init_kwargs(module):
        calls.append(("init", module))
        return {"onload_device": "cpu", "offload_device": "cpu"}

    def fake_remove_module_offload(module, onload_tensors=False):
        calls.append(("onload", module, onload_tensors))

    monkeypatch.setattr(module_offload, "OffloadCache", _FakeOffloadCache)
    monkeypatch.setattr(offload, "get_cache_init_kwargs", fake_get_cache_init_kwargs)
    monkeypatch.setattr(
        module_offload, "remove_module_offload", fake_remove_module_offload
    )

    result = module_offload.subgraph_onload_modules(
        {"root": root, "root.child": root.child}
    )

    assert result == {
        "root": {"onload_device": "cpu", "offload_device": "cpu"}
    }
    assert calls == [("init", root), ("onload", root, True)]


def test_individual_expert_sequential_target_can_be_onloaded(monkeypatch):
    model = _IndividualExpertModel()
    target = model.experts[1]
    target._parameters = _FakeOffloadCache()
    subgraph_modules = {"experts.1": target}
    calls = []

    def fake_get_cache_init_kwargs(module):
        calls.append(("init", module))
        return {"onload_device": "cpu", "offload_device": "cpu"}

    def fake_remove_module_offload(module, onload_tensors=False):
        calls.append(("onload", module, onload_tensors))

    def fake_offload_module(module, **kwargs):
        calls.append(("offload", module, kwargs))

    monkeypatch.setattr(module_offload, "OffloadCache", _FakeOffloadCache)
    monkeypatch.setattr(offload, "get_cache_init_kwargs", fake_get_cache_init_kwargs)
    monkeypatch.setattr(
        module_offload, "remove_module_offload", fake_remove_module_offload
    )
    monkeypatch.setattr(module_offload, "offload_module", fake_offload_module)

    offload_kwargs = module_offload.subgraph_onload_modules(subgraph_modules)
    assert target(torch.tensor([3.0])) == torch.tensor([5.0])
    module_offload.subgraph_offload_modules(subgraph_modules, offload_kwargs)

    assert calls == [
        ("init", target),
        ("onload", target, True),
        ("offload", target, {"onload_device": "cpu", "offload_device": "cpu"}),
    ]


def test_subgraph_offload_uses_defaults_for_untracked_modules(monkeypatch):
    root = torch.nn.Module()
    root.child = torch.nn.Linear(4, 4)
    calls = []

    def fake_get_cache_init_kwargs(module):
        calls.append(("init", module))
        return {"onload_device": "cpu", "offload_device": "cpu"}

    def fake_offload_module(module, **kwargs):
        calls.append(("offload", module, kwargs))

    monkeypatch.setattr(offload, "get_cache_init_kwargs", fake_get_cache_init_kwargs)
    monkeypatch.setattr(module_offload, "offload_module", fake_offload_module)

    module_offload.subgraph_offload_modules(
        {"root": root, "root.child": root.child},
        {"root": {"onload_device": "meta", "offload_device": "cpu"}},
    )

    assert calls == [
        ("offload", root, {"onload_device": "meta", "offload_device": "cpu"}),
        ("init", root.child),
        ("offload", root.child, {"onload_device": "cpu", "offload_device": "cpu"}),
    ]


class _StagingCache(_FakeOffloadCache):
    def __init__(self, tensor):
        super().__init__()
        self.offloaded_values = {"weight": tensor}
        self.is_staged = False
        self.stage_calls = []

    def stage(self, tensor, pin_memory=False):
        self.stage_calls.append((tensor.clone(), pin_memory))
        return tensor + (1 if pin_memory else 0)


def test_subgraph_stage_values_are_consumed_by_onload(monkeypatch):
    root = torch.nn.Module()
    root._original_forward_func = root.forward.__func__
    root._parameters = _StagingCache(torch.tensor([2.0]))
    root._buffers = _StagingCache(torch.tensor([3.0]))
    parameter_cache = root._parameters
    buffer_cache = root._buffers

    def fake_get_cache_init_kwargs(module):
        return {"onload_device": "cpu", "offload_device": "cpu"}

    def fake_remove_module_offload(module, onload_tensors=False):
        assert onload_tensors is True
        module._parameters = {
            name: tensor + 10
            for name, tensor in module._parameters.offloaded_values.items()
        }
        module._buffers = {
            name: tensor + 20
            for name, tensor in module._buffers.offloaded_values.items()
        }

    monkeypatch.setattr(module_offload, "OffloadCache", _FakeOffloadCache)
    monkeypatch.setattr(offload, "get_cache_init_kwargs", fake_get_cache_init_kwargs)
    monkeypatch.setattr(
        module_offload, "remove_module_offload", fake_remove_module_offload
    )

    modules = {"root": root}
    assert module_offload.subgraph_stage_modules(modules, pin_memory=True) is None
    module_offload.subgraph_onload_modules(modules)

    assert root._parameters["weight"].item() == 13.0
    assert root._buffers["weight"].item() == 24.0
    assert len(parameter_cache.stage_calls) == 1
    assert len(buffer_cache.stage_calls) == 1
    assert torch.equal(parameter_cache.stage_calls[0][0], torch.tensor([2.0]))
    assert parameter_cache.stage_calls[0][1] is True
    assert torch.equal(buffer_cache.stage_calls[0][0], torch.tensor([3.0]))
    assert buffer_cache.stage_calls[0][1] is True
