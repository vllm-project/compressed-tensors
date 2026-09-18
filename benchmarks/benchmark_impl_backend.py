# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for ImplBackend dispatch overhead.

Measures the host side cost of the `ImplBackend.entrypoint` wrapper as a
function of how many backends are registered under an op, against a direct call
to the same function. The payload is a no-op, so the number reported is the
dispatch machinery and nothing else.

Dispatch overhead is host side by construction, so this runs on CPU and does not
require a GPU. Backends are registered with requirements that do not match, which
forces the full requirement walk and then the fallback: the worst case for the
current implementation and the case the cache would remove.

Based on https://github.com/vllm-project/compressed-tensors/blob/aa91ea5/benchmarks/benchmark_quantize_triton.py
"""

import time
import torch

from compressed_tensors.utils.impl_backend import ImplBackend

N_BACKENDS = [0, 1, 2, 4, 8]
N_CALLS = 200_000
N_REPS = 7

# fake_quantize calls in one GPTQ W4A16 pass over Qwen2.5-0.5B-Instruct,
# = sum(in_features over quantized Linears) * layers = (896 * 6 + 4864) * 24.
# Used only to turn ns/call into a per-run figure.
CALLS_PER_RUN = 245_760


def payload(x):
    """No-op stand-in for the wrapped op, so only dispatch is measured."""
    return x


def make_requirement(depth):
    """
    Build a requirement that always fails, doing `depth` attribute reads.

    depth=1 approximates `triton_req`, which early exits on a CPU tensor.
    depth=3 approximates `_quantize_triton_req`, which also inspects dtypes.
    """

    def requirement(x):
        if depth == 1:
            return x.is_cuda
        if depth == 2:
            return x.is_cuda and x.dtype == torch.float64
        return x.is_cuda and x.dtype == torch.float64 and x.device.type == "xpu"

    return requirement


def build_entrypoint(op_name, n_backends, depth):
    """Register `n_backends` non-matching backends and return the entrypoint."""
    for index in range(n_backends):
        ImplBackend.register(op_name, make_requirement(depth), index)(
            _named_payload(f"{op_name}_backend_{index}")
        )

    return ImplBackend.entrypoint(op_name)(_named_payload(op_name))


def _named_payload(name):
    """`ImplBackend` requires unique function names across all registrations."""

    def impl(x):
        return x

    impl.__name__ = name
    return impl


def benchmark(func, x, n_calls=N_CALLS, n_reps=N_REPS):
    """
    Time `func(x)` and return the floor in nanoseconds per call.

    Reports the minimum across repetitions rather than the mean: the payload is
    deterministic, so run to run variation here is scheduler noise, and the
    minimum is the least contaminated estimate of the true cost.
    """
    for _ in range(1000):  # warm up the interpreter's call caches
        func(x)

    samples = []
    for _ in range(n_reps):
        start = time.perf_counter_ns()
        for _ in range(n_calls):
            func(x)
        samples.append((time.perf_counter_ns() - start) / n_calls)

    return min(samples)


def run_config(depth, n_backends, x):
    """Benchmark one (requirement cost, backend count) pair."""
    op_name = f"bench_d{depth}_n{n_backends}"
    entrypoint = build_entrypoint(op_name, n_backends, depth)

    # every requirement is built to fail, so dispatch must reach the fallback
    assert entrypoint(x) is x, f"{op_name} dispatched away from the fallback"

    direct_ns = benchmark(payload, x)
    dispatch_ns = benchmark(entrypoint, x)
    overhead_ns = dispatch_ns - direct_ns

    return {
        "depth": depth,
        "n_backends": n_backends,
        "direct_ns": direct_ns,
        "dispatch_ns": dispatch_ns,
        "overhead_ns": overhead_ns,
        "per_run_ms": overhead_ns * CALLS_PER_RUN / 1e6,
    }


def main():
    x = torch.empty(0)
    print(f"torch {torch.__version__}, device cpu, {N_CALLS} calls x {N_REPS} reps")
    print(f"per run column assumes {CALLS_PER_RUN:,} calls")

    results = []
    for depth in (1, 3):
        label = "cheap, early exit" if depth == 1 else "inspects dtype and device"
        print(f"\n{'='*80}")
        print(f"Requirement depth {depth} ({label})")
        print("=" * 80)
        for n_backends in N_BACKENDS:
            result = run_config(depth, n_backends, x)
            results.append(result)
            print(
                f"  {n_backends} backends: dispatch {result['dispatch_ns']:7.1f} ns, "
                f"direct {result['direct_ns']:6.1f} ns, "
                f"overhead {result['overhead_ns']:7.1f} ns"
            )

    print(f"\n{'='*100}")
    print("SUMMARY")
    print("=" * 100)
    print(
        f"{'Depth':<8} {'Backends':<10} {'Direct (ns)':<14} {'Dispatch (ns)':<16} "
        f"{'Overhead (ns)':<16} {'Per run (ms)':<14}"
    )
    print("-" * 100)
    for result in results:
        print(
            f"{result['depth']:<8} {result['n_backends']:<10} "
            f"{result['direct_ns']:<14.1f} {result['dispatch_ns']:<16.1f} "
            f"{result['overhead_ns']:<16.1f} {result['per_run_ms']:<14.2f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()


"""
Example output:

torch X.XX.X, device cpu, 200000 calls x 7 reps
per run column assumes 245,760 calls

====================================================================================================
SUMMARY
====================================================================================================
Depth    Backends   Direct (ns)    Dispatch (ns)    Overhead (ns)    Per run (ms)
----------------------------------------------------------------------------------------------------
1        0          XX.X           XX.X             XX.X             X.XX
1        1          XX.X           XXX.X            XXX.X            XX.XX
1        8          XX.X           XXX.X            XXX.X            XXX.XX
3        0          XX.X           XX.X             XX.X             X.XX
3        8          XX.X           XXX.X            XXX.X            XXX.XX
====================================================================================================
"""
