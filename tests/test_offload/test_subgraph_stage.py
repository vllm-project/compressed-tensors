# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import compressed_tensors.offload.module as module_offload
import torch


class _DeviceOffloadCache(dict):
    offload_device = torch.device("cuda")


def test_subgraph_stage_handles_accelerator_offload(monkeypatch):
    module = torch.nn.Module()
    module._parameters = _DeviceOffloadCache()
    module._buffers = _DeviceOffloadCache()
    calls = []

    monkeypatch.setattr(module_offload, "OffloadCache", _DeviceOffloadCache)
    monkeypatch.setattr(
        module_offload,
        "stage_module_offload",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    module_offload.subgraph_stage_modules({"module": module})

    assert len(calls) == 1
    assert calls[0][0] == (module,)
    assert calls[0][1] == {"pin_memory": False}
