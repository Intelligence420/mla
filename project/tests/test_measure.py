"""Mess-Schicht (TZ 4): Verteilungs-Statistik headless + L2-Flush/Verteilung real.

Standalone-Runner (kein pytest im venv): `python tests/test_measure.py` aus
`project/` mit dem venv-Python. Die reine Statistik (`_summarize_times`) läuft
**ohne GPU** (deterministisch prüfbar); die bench-/run-Tests brauchen GPU +
cuTile und überspringen sich sauber, wenn keine CUDA-GPU da ist.
"""

from __future__ import annotations

import math
import os
import sys

# project/ auf den Pfad, damit `tool_pipeline` importierbar ist (standalone-Lauf).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_pipeline.measure.bench import _summarize_times  # noqa: E402


# ---------------------------------------------------------------------------
# Verteilungs-Statistik — headless, exakt vorhersagbar (kein GPU nötig).
# ---------------------------------------------------------------------------
def test_summarize_times_known_values():
    """Bekannte Liste 1..10 → exakte Kennzahlen.

    Median=5.5, min=1, p90 (nearest-rank: ceil(0.9*10)-1 = 8 → s[8]) = 9,
    σ (Population) = sqrt(82.5/10) = sqrt(8.25) ≈ 2.87228.
    """
    d = _summarize_times([float(x) for x in range(1, 11)])
    assert d["run_ms"] == 5.5, d
    assert d["min_ms"] == 1.0, d
    assert d["p90_ms"] == 9.0, d
    assert abs(d["sigma_ms"] - math.sqrt(8.25)) < 1e-9, d


def test_summarize_times_unsorted_and_ordered():
    """Unsortierte Eingabe: intern sortiert → min ≤ median ≤ p90."""
    d = _summarize_times([7.0, 1.0, 3.0, 9.0, 5.0])
    assert d["min_ms"] == 1.0 and d["p90_ms"] == 9.0, d
    assert d["min_ms"] <= d["run_ms"] <= d["p90_ms"], d


def test_summarize_times_single_value():
    """n=1: alle Lagemaße = der Wert, σ=0 (kein Stichproben-Crash)."""
    d = _summarize_times([4.2])
    assert d["run_ms"] == d["min_ms"] == d["p90_ms"] == 4.2, d
    assert d["sigma_ms"] == 0.0, d


# ---------------------------------------------------------------------------
# Reale Messung — braucht GPU + cuTile (überspringt sich sonst).
# ---------------------------------------------------------------------------
def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def test_benchmark_returns_distribution_keys():
    """Echter bench-Lauf: liefert die Verteilungs-Keys, min ≤ median ≤ p90, σ≥0."""
    if not _has_cuda():
        print("  (übersprungen: keine CUDA-GPU)")
        return
    import torch
    from tool_pipeline.codegen.compile import load_kernel
    from tool_pipeline.codegen.emit import emit
    from tool_pipeline.measure.bench import benchmark
    from tool_pipeline.schema import RunConfig

    cfg = RunConfig(dim_sizes={"i": 256, "k": 256, "j": 256})
    comp = load_kernel(cfg, emit(cfg))
    torch.manual_seed(0)
    A = torch.randn(256, 256, dtype=torch.float16, device="cuda")
    B = torch.randn(256, 256, dtype=torch.float16, device="cuda")
    C = torch.empty(256, 256, dtype=torch.float32, device="cuda")
    comp.launch(A, B, C)             # Kalt-Lauf (cuTile-JIT)
    torch.cuda.synchronize()

    b = benchmark(comp.launch, A, B, C, warmup=3, iters=10)
    for k in ("run_ms", "min_ms", "p90_ms", "sigma_ms", "iters", "warmup"):
        assert k in b, f"Key {k} fehlt im bench-dict: {b}"
    assert b["min_ms"] <= b["run_ms"] <= b["p90_ms"], b
    assert b["sigma_ms"] >= 0.0 and b["iters"] == 10, b


def test_benchmark_flush_toggle_runs():
    """flush_l2=False läuft ebenfalls durch (Vergleichs-Pfad warm vs. cold-L2)."""
    if not _has_cuda():
        print("  (übersprungen: keine CUDA-GPU)")
        return
    import torch
    from tool_pipeline.codegen.compile import load_kernel
    from tool_pipeline.codegen.emit import emit
    from tool_pipeline.measure.bench import benchmark
    from tool_pipeline.schema import RunConfig

    cfg = RunConfig(dim_sizes={"i": 128, "k": 128, "j": 128})
    comp = load_kernel(cfg, emit(cfg))
    torch.manual_seed(0)
    A = torch.randn(128, 128, dtype=torch.float16, device="cuda")
    B = torch.randn(128, 128, dtype=torch.float16, device="cuda")
    C = torch.empty(128, 128, dtype=torch.float32, device="cuda")
    comp.launch(A, B, C)
    torch.cuda.synchronize()
    b = benchmark(comp.launch, A, B, C, warmup=2, iters=5, flush_l2=False)
    assert b["run_ms"] > 0 and b["iters"] == 5, b


def test_run_timing_has_distribution():
    """run() reicht die Verteilungs-Keys in RunResult.timing durch (+ compile getrennt)."""
    if not _has_cuda():
        print("  (übersprungen: keine CUDA-GPU)")
        return
    import tool_pipeline.run as R
    from tool_pipeline.schema import RunConfig
    from tool_pipeline.store import store as st

    orig = st.append_result
    st.append_result = lambda r, path=None: None   # Store isolieren
    try:
        res = R.run(RunConfig(dim_sizes={"i": 256, "k": 256, "j": 256}))
    finally:
        st.append_result = orig
    assert res.status == "ok", f"status={res.status} error={res.error}"
    for k in ("compile_ms", "run_ms", "min_ms", "p90_ms", "sigma_ms", "bench_iters"):
        assert k in res.timing, f"timing fehlt {k}: {res.timing}"
    assert res.timing["min_ms"] <= res.timing["run_ms"] <= res.timing["p90_ms"], res.timing


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} Tests bestanden")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
