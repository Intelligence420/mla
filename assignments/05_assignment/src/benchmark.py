"""
Pipeline und Benchmark fuer Task 4d.

Faedelt die Bausteine aus config.py / optimizer.py / kernel.py
zur kompletten Aufgabe zusammen:

  1. Basis-Config bauen und ausgeben (Task 4a-Output).
  2. Optimizer-Pipeline ausfuehren -> L2-Config (Task 4b-Output).
  3. Verifikation beider Kernels gegen torch.einsum.
  4. Benchmark via triton.testing.do_bench, TFLOPS-Vergleich,
     Bar-Chart als PNG.
"""

import os

import matplotlib.pyplot as plt
import torch
import triton

from config import pretty
from kernel import (
    DIMS, GROUP_M, GROUP_N, K_PRIM, M_PRIM, N_PRIM,
    build_basic_config,
    build_l2_config,
    run_baseline,
    run_l2,
    verify_kernel,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# FLOPs / Hilfen
# ===========================================================================

def flops_count(dims: dict) -> int:
    """cmk,ckn->cmn: 2 * c * m * n * k FLOPs."""
    return 2 * dims["C"] * dims["M"] * dims["N"] * dims["K"]


def tflops(dims: dict, time_ms: float) -> float:
    return flops_count(dims) / (time_ms * 1e-3) / 1e12


def bench(runner, A, B, dims: dict,
          n_warmup: int = 10, n_runs: int = 50, **kwargs) -> float:
    return triton.testing.do_bench(
        lambda: runner(A, B, dims=dims, **kwargs),
        warmup=n_warmup, rep=n_runs,
    )


# ===========================================================================
# Task 4d — Benchmark
# ===========================================================================

def benchmark() -> dict[str, tuple[float, float]]:
    """Misst Laufzeit und TFLOPS fuer Baseline und L2-optimiert.

    Returns
    -------
    dict[name -> (ms, tflops)]
    """
    Cd, M, N, K = DIMS["C"], DIMS["M"], DIMS["N"], DIMS["K"]
    print(f"\n  dims  = {DIMS}")
    print(f"  FLOPs = {flops_count(DIMS):.3e}")

    torch.manual_seed(0)
    A = torch.randn(Cd, M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(Cd, K, N, device="cuda", dtype=torch.float16)

    runners = [
        ("baseline", run_baseline, {}),
        ("l2_swizzle", run_l2, dict(group=(GROUP_M, GROUP_N))),
    ]

    results: dict[str, tuple[float, float]] = {}
    for name, fn, kwargs in runners:
        ms = bench(fn, A, B, DIMS, **kwargs)
        tf = tflops(DIMS, ms)
        results[name] = (ms, tf)

    baseline_ms = results["baseline"][0]
    print(f"\n  {'kernel':<14} {'ms':>10} {'TFLOPS':>10} {'vs baseline':>14}")
    print("  " + "-" * 50)
    for name, (ms, tf) in results.items():
        speedup = baseline_ms / ms
        print(f"  {name:<14} {ms:>10.4f} {tf:>10.3f} {speedup:>13.2f}x")
    return results


def plot_results(results: dict[str, tuple[float, float]],
                 path: str | None = None) -> str:
    """Bar-Chart: Laufzeit + TFLOPS nebeneinander."""
    if path is None:
        path = os.path.join(SCRIPT_DIR, "task04_l2_vs_baseline.png")

    names = list(results.keys())
    ms_vals = [results[n][0] for n in names]
    tf_vals = [results[n][1] for n in names]

    fig, (ax_ms, ax_tf) = plt.subplots(1, 2, figsize=(10, 4))

    ax_ms.bar(names, ms_vals, color=["tab:gray", "tab:orange"])
    ax_ms.set_ylabel("Laufzeit [ms]")
    ax_ms.set_title("Laufzeit (kleiner = besser)")
    for i, v in enumerate(ms_vals):
        ax_ms.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    ax_tf.bar(names, tf_vals, color=["tab:gray", "tab:orange"])
    ax_tf.set_ylabel("TFLOPS")
    ax_tf.set_title("Durchsatz (groesser = besser)")
    for i, v in enumerate(tf_vals):
        ax_tf.text(i, v, f"{v:.1f}", ha="center", va="bottom")

    fig.suptitle(f"Task 4d: cmk,ckn->cmn  ({DIMS['C']} x {DIMS['M']}^3)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ===========================================================================
# Pipeline
# ===========================================================================

def main() -> None:
    print("=" * 70)
    print("Task 4a — Basis-Config")
    print("=" * 70)
    cfg_basic = build_basic_config()
    print(pretty(cfg_basic, list("cmkn")))

    print()
    print("=" * 70)
    print("Task 4b — L2-optimierte Config")
    print("=" * 70)
    cfg_l2 = build_l2_config()
    print(pretty(cfg_l2, ["c", "m_l2", "n_l2", "m_prim", "n_prim", "k"]))
    print(f"\n  m_prim={M_PRIM}, n_prim={N_PRIM}, k_prim={K_PRIM}, "
          f"GROUP_M={GROUP_M}, GROUP_N={GROUP_N}")

    print()
    print("=" * 70)
    print("Task 4c — Verifikation gegen torch.einsum")
    print("=" * 70)
    verify_kernel(run_baseline, "baseline")
    verify_kernel(run_l2, "l2", group=(GROUP_M, GROUP_N))

    print()
    print("=" * 70)
    print("Task 4d — Benchmark")
    print("=" * 70)
    results = benchmark()
    path = plot_results(results)
    print(f"\n  Plot saved to {path}")


if __name__ == "__main__":
    main()
