# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import os
import subprocess
import sys

import pytest
import torch
from compressed_tensors.utils import impl_backend
from compressed_tensors.utils.impl_backend import ImplBackend


@pytest.fixture(autouse=True)
def isolate_registry():
    """
    Restore the class level registries after each test.

    `ImplBackend` keeps `_backends` and `_fn_registry` as mutable class
    attributes, so a test that registers a backend would otherwise leak into
    every later test and trip the unique name check.
    """
    backends = {name: list(entries) for name, entries in ImplBackend._backends.items()}
    fn_registry = dict(ImplBackend._fn_registry)
    yield
    ImplBackend._backends = backends
    ImplBackend._fn_registry = fn_registry


def _probe_devices():
    """Devices to evaluate requirements against, including any accelerator."""
    devices = [torch.device("cpu"), torch.device("meta")]
    if torch.accelerator.is_available():
        devices.append(torch.device(torch.accelerator.current_accelerator().type, 0))
    return devices


def _probes():
    """Representative inputs spanning the properties requirements read."""
    dtypes = [torch.float32, torch.bfloat16, torch.float8_e4m3fn]
    return [
        torch.empty(0, device=device, dtype=dtype)
        for device, dtype in itertools.product(_probe_devices(), dtypes)
    ]


def _backend(name):
    """Build a uniquely named backend that reports which backend ran."""

    def impl(x):
        return name

    impl.__name__ = name
    return impl


def _overlapping_requirements(name, probes):
    """
    Find probes for which more than one backend requirement holds.

    Two backends registered under the same name are expected to partition the
    input space between them. When they do, at most one requirement can hold
    for any input, dispatch does not depend on the order the backends were
    registered, and the `priority` field is not load bearing.

    :param name: operation name to check
    :param probes: inputs to evaluate every requirement against
    :return: list of (probe, [backend names]) for probes with multiple matches
    """
    entries = ImplBackend._backends.get(name, [])
    conflicts = []
    for probe in probes:
        matched = [fn.__name__ for fn, req, _ in entries if req(probe)]
        if len(matched) > 1:
            conflicts.append((probe, matched))
    return conflicts


def test_entrypoint_dispatches_to_backend_when_requirement_holds():
    @ImplBackend.register("op", lambda x: x.device.type == "cpu", 0)
    def op_cpu(x):
        return "backend"

    @ImplBackend.entrypoint("op")
    def op(x):
        return "fallback"

    assert op(torch.empty(0)) == "backend"


def test_entrypoint_falls_back_when_no_requirement_holds():
    @ImplBackend.register("op", lambda x: x.device.type == "meta", 0)
    def op_meta(x):
        return "backend"

    @ImplBackend.entrypoint("op")
    def op(x):
        return "fallback"

    assert op(torch.empty(0)) == "fallback"


def test_disabled_backend_is_neither_dispatched_nor_resolvable():
    """Characterizes current behavior: `priority="disable"` returns before the
    name reaches `_fn_registry`, so a disabled backend is skipped by dispatch
    (intended) but is *also* unreachable through `ImplBackend.call`, which the
    class docstring documents as the way to exercise a backend in isolation.

    No backend in this build registers as `"disable"`, so nothing is currently
    affected. The asymmetry is still worth pinning down, because `"disable"` is
    documented as a supported value and the next thing to use it would silently
    lose its isolated test path.
    """

    @ImplBackend.register("op", lambda x: True, "disable")
    def op_disabled(x):
        return "backend"

    @ImplBackend.entrypoint("op")
    def op(x):
        return "fallback"

    assert op(torch.empty(0)) == "fallback"
    assert "op" not in ImplBackend._backends
    with pytest.raises(KeyError, match="op_disabled"):
        ImplBackend.call("op_disabled", torch.empty(0))


def test_call_invokes_a_backend_regardless_of_its_requirement():
    @ImplBackend.register("op", lambda x: False, 0)
    def op_never(x):
        return "backend"

    @ImplBackend.entrypoint("op")
    def op(x):
        return "fallback"

    assert op(torch.empty(0)) == "fallback"
    assert ImplBackend.call("op_never", torch.empty(0)) == "backend"


def test_call_raises_for_unknown_name():
    """The error names the missing backend and lists what is available."""
    with pytest.raises(KeyError, match="No registered backend named"):
        ImplBackend.call("not_registered")


def test_duplicate_backend_name_raises():
    @ImplBackend.entrypoint("op")
    def duplicated(x):
        return "fallback"

    with pytest.raises(ValueError, match="already"):

        @ImplBackend.register("other_op", lambda x: True, 0)
        def duplicated(x):  # noqa: F811
            return "backend"


def test_enforce_eager_skips_registered_backends(monkeypatch):
    @ImplBackend.register("op", lambda x: True, 0)
    def op_always(x):
        return "backend"

    @ImplBackend.entrypoint("op")
    def op(x):
        return "fallback"

    assert op(torch.empty(0)) == "backend"
    monkeypatch.setattr(impl_backend, "ENFORCE_EAGER", True)
    assert op(torch.empty(0)) == "fallback"


@pytest.mark.parametrize(
    "value,expected", [("0", False), ("false", False), ("1", True), ("true", True)]
)
def test_enforce_eager_env_parses_boolean_strings(value, expected):
    # Run in a fresh interpreter: ENFORCE_EAGER is read once at import time.
    code = (
        "from compressed_tensors.utils import impl_backend; "
        "print(impl_backend.ENFORCE_EAGER)"
    )
    env = {**os.environ, "CT_ENFORCE_EAGER": value}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == str(expected)


@pytest.mark.parametrize("reverse", [False, True], ids=["in_order", "reversed"])
def test_disjoint_requirements_dispatch_independent_of_registration_order(reverse):
    """
    Registration order does not change dispatch when requirements are disjoint.

    This is the property that makes `priority` redundant: priority only breaks
    ties between backends that can both accept the same input.
    """
    registrations = [
        ("op_cpu", lambda x: x.device.type == "cpu"),
        ("op_meta", lambda x: x.device.type == "meta"),
    ]
    if reverse:
        registrations.reverse()

    for name, req in registrations:
        ImplBackend.register("op", req, 0)(_backend(name))

    @ImplBackend.entrypoint("op")
    def op(x):
        return "fallback"

    assert not _overlapping_requirements("op", _probes())
    assert op(torch.empty(0, device="cpu")) == "op_cpu"
    assert op(torch.empty(0, device="meta")) == "op_meta"


def test_overlap_detection_reports_backends_that_both_accept_an_input():
    """The disjointness check below is only meaningful if it can fail."""
    ImplBackend.register("op", lambda x: True, 0)(_backend("op_any"))
    ImplBackend.register("op", lambda x: not x.is_cuda, 0)(_backend("op_cpu_too"))

    conflicts = _overlapping_requirements("op", _probes())
    assert conflicts, "expected overlapping requirements to be reported"
    _, matched = conflicts[0]
    assert sorted(matched) == ["op_any", "op_cpu_too"]


def test_each_registered_op_has_exactly_one_backend():
    """
    Priority cannot influence dispatch while every op has a single backend.

    This is the form of the non-overlap property that is actually true of this
    build: with one backend per name there is nothing for `priority` to order,
    so removing the field would be a no-op. A whole-registry disjointness scan
    would pass vacuously here, which is why it is not what this asserts.

    When a second backend appears for some op, this fails. That is the point at
    which its requirements have to be shown disjoint for real, against probes
    matching that op's own signature.
    """
    # imported for their registration side effects
    import compressed_tensors.compressors.nvfp4.helpers  # noqa: F401
    import compressed_tensors.quantization.lifecycle.forward_helpers  # noqa: F401
    import compressed_tensors.quantization.utils.fp4_utils  # noqa: F401

    assert ImplBackend._backends, "expected at least one registered backend"
    multiple = {
        name: [fn.__name__ for fn, _, _ in entries]
        for name, entries in ImplBackend._backends.items()
        if len(entries) > 1
    }
    assert not multiple, (
        "more than one backend is registered for: "
        + "; ".join(f"{name} -> {names}" for name, names in sorted(multiple.items()))
        + ". Dispatch now depends on requirement disjointness, so add probes for "
        "these ops and assert the requirements cannot both hold."
    )
