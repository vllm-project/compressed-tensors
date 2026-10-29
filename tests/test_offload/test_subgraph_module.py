# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import compressed_tensors.offload as offload
import compressed_tensors.offload.module as module_offload
import pytest
import torch
from compressed_tensors.offload.cache import OffloadCache


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

    assert result == {"root": {"onload_device": "cpu", "offload_device": "cpu"}}
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
        {"root": root, "root_other.child": root.child},
        {"root": {"onload_device": "meta", "offload_device": "cpu"}},
    )

    assert calls == [
        ("offload", root, {"onload_device": "meta", "offload_device": "cpu"}),
        ("init", root.child),
        ("offload", root.child, {"onload_device": "cpu", "offload_device": "cpu"}),
    ]


@pytest.mark.parametrize("prefix", ["", "layer"])
def test_subgraph_offload_inherits_saved_ancestor_settings(tmp_path, prefix):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    module_offload.offload_module(model, "cpu", "disk", offload_dir=tmp_path)
    module_offload.offload_module(model[0], "cpu", "cpu")
    # Children can precede parents in the original scope.
    modules = dict(reversed(list(model.named_modules(prefix=prefix))))
    original_scope = modules.copy()
    saved = module_offload.subgraph_onload_modules(modules)

    # New nested children inherit the parent settings even though their tensors
    # are currently on CPU. Explicit child settings override the root's settings.
    model.added = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model[0].added = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model.alias = model.added[0]  # shared descendants must only be offloaded once
    model.added[0].register_buffer("extra", torch.randn(4))
    expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    module_offload.subgraph_offload_modules(modules, saved)
    assert modules == original_scope

    for module in (model, model.added, model.added[0]):
        for cache in (module._parameters, module._buffers):
            assert isinstance(cache, OffloadCache)
            assert cache.offload_device == "disk"
            assert cache.offload_dir == tmp_path
            assert cache.onload_device == "cpu"
    for module in (model[0], model[0].added, model[0].added[0]):
        assert isinstance(module._parameters, OffloadCache)
        assert module._parameters.offload_device == torch.device("cpu")

    module_offload.subgraph_onload_modules(dict(model.named_modules(prefix=prefix)))
    torch.testing.assert_close(model.state_dict(), expected)


def test_subgraph_offload_inherits_without_offloading_ancestor_or_siblings(tmp_path):
    parent = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    module_offload.offload_module(parent, "cpu", "disk", offload_dir=tmp_path)
    module_offload.offload_module(parent[1], "cpu", "disk", offload_dir=tmp_path)
    sibling_modules = {"parent.1": parent[1]}
    module_offload.subgraph_stage_modules(sibling_modules)
    sibling_cache = parent[1]._parameters
    sibling_weight = sibling_cache.offloaded_values["weight"]
    parent_cache = parent._parameters

    module_offload.subgraph_offload_modules(
        {"parent.0": parent[0]}, {"parent": offload.get_cache_init_kwargs(parent)}
    )

    assert parent[0]._parameters.offload_device == "disk"
    assert parent[0]._parameters.offload_dir == tmp_path
    assert parent._parameters is parent_cache
    assert parent[1]._parameters is sibling_cache
    assert sibling_cache.is_staged
    assert sibling_cache.offloaded_values["weight"] is sibling_weight


@pytest.mark.parametrize("staged", [False, True])
def test_subgraph_offload_stops_at_other_offloaded_subtrees(tmp_path, staged):
    parent = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.Sequential(torch.nn.Linear(4, 4))
    )
    for module in parent.modules():
        module_offload.offload_module(module, "cpu", "disk", offload_dir=tmp_path)
    modules = {"parent": parent, "parent.0": parent[0]}
    saved = module_offload.subgraph_onload_modules(modules)
    other_scope = {"parent.1": parent[1], "parent.1.0": parent[1][0]}
    if staged:
        module_offload.subgraph_stage_modules(other_scope)
    other_caches = [module._parameters for module in other_scope.values()]
    other_weight = parent[1][0]._parameters.offloaded_values["weight"]

    parent.added = torch.nn.Sequential(torch.nn.Linear(4, 4))
    parent[1].added = torch.nn.Linear(4, 4)
    module_offload.subgraph_offload_modules(modules, saved)

    assert set(modules) == {"parent", "parent.0"}
    for module in parent.added.modules():
        assert isinstance(module._parameters, OffloadCache)
        assert module._parameters.offload_device == "disk"
    for module, cache in zip(other_scope.values(), other_caches):
        assert module._parameters is cache
        assert cache.is_staged == staged
    assert parent[1][0]._parameters.offloaded_values["weight"] is other_weight
    assert not isinstance(parent[1].added._parameters, OffloadCache)


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
