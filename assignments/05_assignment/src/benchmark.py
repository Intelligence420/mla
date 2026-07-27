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
          warmup_ms: int = 200, rep_ms: int = 1000, **kwargs) -> float:
    # warmup/rep sind bei triton.testing.do_bench Zeitbudgets in Millisekunden,
    # keine Iterationszahlen. Grosszuegig gewaehlt, damit auch die ~45 ms teure
    # Baseline ueber mehrere Iterationen gemittelt wird statt aus einem Einzel-Sample.
    return triton.testing.do_bench(
        lambda: runner(A, B, dims=dims, **kwargs),
        warmup=warmup_ms, rep=rep_ms,
    )


# ===========================================================================
# Task 4d — Benchmark
# ===========================================================================

GROUP_SWEEP = [4, 8, 32]


def benchmark() -> dict[str, tuple[float, float]]:
    """Misst Baseline und den config-getriebenen L2-Kernel ueber den
    GROUP-Sweep {4, 8, 32}.

    Returns
    -------
    dict[name -> (ms, tflops)]  mit "baseline" und "l2_g{g}"-Eintraegen.
    """
    Cd, M, N, K = DIMS["C"], DIMS["M"], DIMS["N"], DIMS["K"]
    print(f"\n  dims  = {DIMS}")
    print(f"  FLOPs = {flops_count(DIMS):.3e}")

    torch.manual_seed(0)
    A = torch.randn(Cd, M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(Cd, K, N, device="cuda", dtype=torch.float16)

    results: dict[str, tuple[float, float]] = {}
    ms = bench(run_baseline, A, B, DIMS)
    results["baseline"] = (ms, tflops(DIMS, ms))
    for g in GROUP_SWEEP:
        ms = bench(run_l2, A, B, DIMS, group=(g, g))
        results[f"l2_g{g}"] = (ms, tflops(DIMS, ms))

    baseline_ms = results["baseline"][0]
    print(f"\n  {'kernel':<14} {'ms':>10} {'TFLOPS':>10} {'vs baseline':>14}")
    print("  " + "-" * 50)
    for name, (ms, tf) in results.items():
        print(f"  {name:<14} {ms:>10.4f} {tf:>10.3f} {baseline_ms / ms:>13.2f}x")
    return results


def plot_results(results: dict[str, tuple[float, float]]) -> list[str]:
    """Ein Bar-Chart (Laufzeit + TFLOPS) je GROUP: Baseline vs. L2-Swizzle."""
    base = results["baseline"]
    paths = []
    for g in GROUP_SWEEP:
        l2 = results[f"l2_g{g}"]
        names = ["baseline", f"l2 (GROUP={g})"]
        ms_vals = [base[0], l2[0]]
        tf_vals = [base[1], l2[1]]

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
        fig.suptitle(f"Task 4d: cmk,ckn->cmn  ({DIMS['C']} x {DIMS['M']}^3), "
                     f"GROUP={g}")
        fig.tight_layout()
        path = os.path.join(SCRIPT_DIR,
                            f"task04_l2_vs_baseline_GROUP-{g}-{g}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


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
    print(pretty(cfg_l2, ["c", "m_super", "n_super", "m_group", "n_group",
                          "m_prim", "n_prim", "k"]))
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
    paths = plot_results(results)
    for p in paths:
        print(f"  Plot saved to {p}")


if __name__ == "__main__":
    main()


"""Ergebnisse (GROUP-Sweep {4, 8, 32}, config-getriebener L2-Kernel)

(.venv) mla08@flambe:~/MLA/mla/assignments/05_assignment/src$ python3 benchmark.py
======================================================================
Task 4b - L2-optimierte Config (8-dim: super/group + prim)
======================================================================
pos name    type  exec      size   strides
------------------------------------------
0   c       C     PAR          4     16777216   16777216   16777216
1   m_super M     PAR          8      2097152          0    2097152
2   n_super N     PAR          8            0        512        512
3   m_group M     PAR          8       262144          0     262144
4   n_group N     PAR          8            0         64         64
5   m_prim  M     PRIM        64         4096          0       4096
6   n_prim  N     PRIM        64            0          1          1
7   k       K     PRIM      4096            1       4096          0
  data_type=FLOAT16  prim_main=GEMM  prim_last=NONE  prim_first=ZERO

  m_prim=64, n_prim=64, k_prim=32, GROUP_M=8, GROUP_N=8

Task 4c - Verifikation gegen torch.einsum
  baseline   allclose=True  max_abs_err=0.0078
  l2         allclose=True  max_abs_err=0.0078

Task 4d - Benchmark  (C=4, M=N=K=4096, FLOPs=5.498e+11)
  kernel                 ms     TFLOPS    vs baseline
  --------------------------------------------------
  baseline          46.6179     11.793          1.00x
  l2_g4             14.9133     36.863          3.13x
  l2_g8             13.1076     41.942          3.56x   <- Sweet Spot
  l2_g32            14.0163     39.223          3.33x

Hinweis: do_bench-Absolutwerte schwanken je Lauf um einige Prozent; die
relative Ordnung (Baseline ~4x langsamer, GROUP=8 optimal) ist stabil.
"""
