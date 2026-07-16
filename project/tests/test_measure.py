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

from tool_pipeline.hardware import peak_tflops  # noqa: E402
from tool_pipeline.measure.bench import _summarize_times  # noqa: E402
from tool_pipeline.measure.metrics import (  # noqa: E402
    compute_metrics,
    compute_metrics_elementwise,
    compute_metrics_reduction,
    elementwise_bytes,
    elementwise_flops,
    gemm_bytes,
    gemm_flops,
    reduction_bytes,
    reduction_flops,
)


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
# Abgeleitete Metriken — headless, exakt vorhersagbar (kein GPU nötig).
# ---------------------------------------------------------------------------
def test_gemm_bytes_known_values():
    """512³, fp16(2 B)→fp32(4 B): bytes = 2·(M·K+K·N) + 4·(M·N) = 2 097 152."""
    assert gemm_bytes(512, 512, 512, "fp16", "fp32") == 2_097_152
    # tf32 liegt als float32 (4 B) im Speicher (Cast passiert im Kernel): nur die
    # Inputs wachsen 2→4 B, der fp32-Output ist in beiden Fällen 4 B.
    assert gemm_bytes(512, 512, 512, "tf32", "fp32") == 4 * (512 * 512 + 512 * 512) + 4 * (512 * 512)
    # fp8 (1 B) Input, fp16 (2 B) Output.
    assert gemm_bytes(512, 512, 512, "fp8e4m3", "fp16") == (
        1 * (512 * 512 + 512 * 512) + 2 * (512 * 512))


def test_compute_metrics_known_values():
    """512³ fp16→fp32 @ run_ms=1.0 → exakt nachrechenbare Kennzahlen.

    flops=2·512³=268 435 456; bytes=2 097 152 ⇒ arithm. Intensität = 128 FLOP/Byte
    (deterministisch, GPU-unabhängig). Bei 1 ms: TFLOP/s=0.2684…, GB/s=2.097…,
    %-Peak-flops=0.2684/213·100≈0.1, %-Peak-bw=2.097/273·100≈0.8.
    """
    m = compute_metrics(512, 512, 512, 1.0, "fp16", "fp32")
    assert m["arithmetic_intensity"] == 128.0, m
    assert m["gbps"] == 2.1, m                       # round(2.097152, 2)
    assert abs(m["tflops"] - 0.268435456) < 1e-9, m
    assert m["percent_peak_flops"] == 0.1, m
    assert m["percent_peak_bw"] == 0.8, m


def test_batched_metrics_scale_with_B():
    """B (Batch) skaliert FLOPs UND Bytes gemeinsam linear: tflops/gbps ×B, die
    arithmetische Intensität (FLOP/Byte) bleibt batch-unabhängig. Default B=1
    lässt TZ 1–5 unverändert (die anderen Metrik-Tests belegen das)."""
    assert gemm_flops(512, 512, 512, B=4) == 4 * gemm_flops(512, 512, 512)
    assert (gemm_bytes(512, 512, 512, "fp16", "fp32", B=4)
            == 4 * gemm_bytes(512, 512, 512, "fp16", "fp32"))
    m1 = compute_metrics(512, 512, 512, 1.0, "fp16", "fp32")
    m4 = compute_metrics(512, 512, 512, 1.0, "fp16", "fp32", B=4)
    assert m4["arithmetic_intensity"] == m1["arithmetic_intensity"] == 128.0, (m1, m4)
    assert abs(m4["tflops"] - 4 * m1["tflops"]) < 1e-9, (m1["tflops"], m4["tflops"])
    assert m4["gbps"] == round(4 * 2.097152, 2), m4      # gbps ×B, danach gerundet (8.39)


def test_compute_metrics_fp32_peak_none():
    """fp32-plain hat keinen Tensor-Core-Peak → %-Peak-flops ist None (kein
    irreführender Nenner); GB/s/Intensität bleiben aber definiert."""
    m = compute_metrics(256, 256, 256, 0.5, "fp32", "fp32")
    assert peak_tflops("fp32") is None
    assert m["percent_peak_flops"] is None, m
    assert m["gbps"] is not None and m["arithmetic_intensity"] is not None, m


def test_compute_metrics_run_ms_zero_graceful():
    """run_ms=0 → tflops/gbps NaN statt Division durch Null (kein Crash)."""
    m = compute_metrics(128, 128, 128, 0.0, "fp16", "fp32")
    assert m["tflops"] != m["tflops"], m             # NaN != NaN
    assert m["gbps"] != m["gbps"], m


# ---------------------------------------------------------------------------
# Memory-bound-Metriken (TZ 7) — headless, exakt vorhersagbar (kein GPU nötig).
# ---------------------------------------------------------------------------
def test_elementwise_flops_and_bytes():
    """add/mul = 1 FLOP/Element, copy = 0 (reine Bandbreite). Bytes = arity·in + out."""
    E = 1_000_000
    assert elementwise_flops(E, "add") == E
    assert elementwise_flops(E, "mul") == E
    assert elementwise_flops(E, "copy") == 0
    # binär fp16(2)->fp32(4): 2·2 + 4 = 8 B/Element; unär (copy): 1·2 + 4 = 6 B/Element.
    assert elementwise_bytes(E, 2, "fp16", "fp32") == 8 * E
    assert elementwise_bytes(E, 1, "fp16", "fp32") == 6 * E


def test_reduction_flops_and_bytes():
    """Summe ~ kept·reduced Additionen; Traffic = ganze Eingabe + kleiner Output."""
    assert reduction_flops(1000, 2000) == 2_000_000
    # fp16(2) Eingabe (kept·reduced), fp32(4) Output (kept).
    assert reduction_bytes(1000, 2000, "fp16", "fp32") == 1000 * 2000 * 2 + 1000 * 4


def test_compute_metrics_elementwise_add_known_values():
    """E=1e6, add, fp16->fp32 @ 1 ms: flops=1e6, bytes=8e6 ⇒ AI=0.12 (round 0.125);
    tflops=0.001, gbps=8.0, %-Peak-bw=8/273·100≈2.9, %-Peak-flops≈0.0 (memory-bound)."""
    m = compute_metrics_elementwise(1_000_000, 2, "add", 1.0, "fp16", "fp32")
    assert m["arithmetic_intensity"] == 0.12, m
    assert abs(m["tflops"] - 0.001) < 1e-12, m
    assert m["gbps"] == 8.0, m
    assert m["percent_peak_bw"] == 2.9, m
    assert m["percent_peak_flops"] == 0.0, m


def test_compute_metrics_elementwise_copy_zero_flops():
    """copy: flops=0 ⇒ tflops=0, AI=0 (kein Roofline-Punkt — reine Bandbreite);
    bytes=6e6 ⇒ gbps=6.0, %-Peak-bw definiert."""
    m = compute_metrics_elementwise(1_000_000, 1, "copy", 1.0, "fp16", "fp32")
    assert m["tflops"] == 0.0 and m["arithmetic_intensity"] == 0.0, m
    assert m["gbps"] == 6.0, m
    assert m["percent_peak_bw"] == 2.2, m


def test_compute_metrics_reduction_known_values():
    """kept=1000, reduced=2000, fp16->fp32 @ 1 ms: flops=2e6, bytes=4.004e6 ⇒ AI=0.5;
    tflops=0.002, gbps=4.0, %-Peak-bw≈1.5."""
    m = compute_metrics_reduction(1000, 2000, 1.0, "fp16", "fp32")
    assert m["arithmetic_intensity"] == 0.5, m
    assert abs(m["tflops"] - 0.002) < 1e-12, m
    assert m["gbps"] == 4.0, m
    assert m["percent_peak_bw"] == 1.5, m


def test_memory_bound_ai_far_below_gemm():
    """Kernaussage der Roofline: memory-bound-AI (Elementwise/Reduktion) liegt
    weit UNTER der GEMM-AI (128 @ 512³) — die Punkte sitzen links vom Ridge."""
    gemm_ai = compute_metrics(512, 512, 512, 1.0, "fp16", "fp32")["arithmetic_intensity"]
    el_ai = compute_metrics_elementwise(1_000_000, 2, "add", 1.0, "fp16", "fp32")["arithmetic_intensity"]
    rd_ai = compute_metrics_reduction(1000, 2000, 1.0, "fp16", "fp32")["arithmetic_intensity"]
    assert el_ai < 1.0 and rd_ai < 1.0 < gemm_ai, (el_ai, rd_ai, gemm_ai)


# ---------------------------------------------------------------------------
# Baselines — headless prüfbare Anteile (kein GPU nötig).
# ---------------------------------------------------------------------------
def test_measure_baselines_unknown_name_graceful():
    """Unbekannte Baseline → available=False + Grund, ohne GPU/Crash."""
    import torch
    from tool_pipeline.measure.baselines import measure_baselines
    from tool_pipeline.schema import RunConfig

    A = torch.zeros(4, 4); B = torch.zeros(4, 4); C = torch.zeros(4, 4)
    out = measure_baselines(["bogus"], A, B, C, RunConfig())
    assert out["bogus"]["available"] is False and "note" in out["bogus"], out


def test_baselines_not_in_slug():
    """Baselines ändern den Kernel-Quelltext NICHT → gleicher Config-Slug
    (kein Cache-Split, keine doppelte Kernel-Datei)."""
    from tool_pipeline.schema import RunConfig
    from tool_pipeline.store.store import config_slug

    a = config_slug(RunConfig(baselines=[]))
    b = config_slug(RunConfig(baselines=["cublas", "naive"]))
    assert a == b, (a, b)


def test_group_m_in_slug_conditional():
    """GROUP_M (TZ 7.5) geht NUR bei swizzle=True UND group_m!=8 in den Slug —
    sonst byte-identisch zu TZ 1-6. Schützt den Compile-Cache vor stillem Fehltreffer
    (zwei GROUP_M-Werte dürfen nicht dieselbe kernels/<slug>.py teilen)."""
    from tool_pipeline.schema import RunConfig
    from tool_pipeline.store.store import config_slug

    base = "ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64"
    # Default 8 (implizit + explizit) → bares __sw (byte-identisch)
    assert config_slug(RunConfig(swizzle=True)) == base + "__sw"
    assert config_slug(RunConfig(swizzle=True, group_m=8)) == base + "__sw"
    # abweichender GROUP_M → eigener Slug
    assert config_slug(RunConfig(swizzle=True, group_m=16)) == base + "__sw_g16"
    assert config_slug(RunConfig(swizzle=True, group_m=1)) == base + "__sw_g1"
    # GROUP_M ohne Swizzle wirkungslos → kein Suffix
    assert config_slug(RunConfig(swizzle=False, group_m=16)) == base
    # Altzeile ohne group_m-Key → bares __sw (Rückwärtskompatibilität)
    assert config_slug({"expr": "ik,kj->ij", "dtype": "fp16", "acc_dtype": "fp32",
                        "tile": {"TM": 128, "TN": 128, "TK": 64}, "swizzle": True}) == base + "__sw"


def test_compute_metrics_nary_aggregates():
    """n-är-Metrik (TZ 7.5-3): FLOPs+Bytes über die paarweisen Schritte aggregiert →
    EIN Roofline-Punkt = Summe der Per-Schritt-GEMM-Kosten (inkl. Zwischentensor)."""
    from tool_pipeline.measure.metrics import (compute_metrics_nary, gemm_bytes,
                                               gemm_flops, tflops)
    steps = [(8, 6, 4, 1), (8, 5, 6, 1)]   # (M,N,K,B) zweier Ketten-Schritte
    m = compute_metrics_nary(steps, run_ms=2.0, dtype="fp16", acc_dtype="fp32")
    exp_flops = sum(gemm_flops(M, N, K, B) for (M, N, K, B) in steps)
    exp_bytes = sum(gemm_bytes(M, N, K, "fp16", "fp32", B) for (M, N, K, B) in steps)
    assert abs(m["tflops"] - tflops(exp_flops, 2.0)) < 1e-9
    assert m["arithmetic_intensity"] == round(exp_flops / exp_bytes, 2)
    assert m["gbps"] is not None and m["percent_peak_flops"] is not None


def test_gpu_state_returns_dict():
    """gpu_state() liefert ein dict; wo nvidia-smi da ist, die erwarteten Keys."""
    from tool_pipeline.measure.provenance import gpu_state

    d = gpu_state()
    assert isinstance(d, dict), d
    if d:  # nvidia-smi vorhanden
        for k in ("sm_clock_mhz", "temp_c", "power_w"):
            assert k in d, d


def test_gpu_state_graceful_without_nvidia_smi():
    """Ohne nvidia-smi im PATH → leeres dict (kein Crash)."""
    import shutil
    from tool_pipeline.measure import provenance as P

    orig = shutil.which
    shutil.which = lambda name: None
    try:
        assert P.gpu_state() == {}, "ohne nvidia-smi muss gpu_state() {} liefern"
    finally:
        shutil.which = orig


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
    A = torch.randn(1, 256, 256, dtype=torch.float16, device="cuda")   # kanonisch (B,M,K), B=1
    B = torch.randn(1, 256, 256, dtype=torch.float16, device="cuda")
    C = torch.empty(1, 256, 256, dtype=torch.float32, device="cuda")
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
    A = torch.randn(1, 128, 128, dtype=torch.float16, device="cuda")   # kanonisch (B,M,K), B=1
    B = torch.randn(1, 128, 128, dtype=torch.float16, device="cuda")
    C = torch.empty(1, 128, 128, dtype=torch.float32, device="cuda")
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


def test_run_metrics_has_roofline_keys():
    """run() liefert die Roofline-Metriken in RunResult.metrics (512³ fp16→fp32).

    arithm. Intensität = 128 FLOP/Byte ist deterministisch (GPU-unabhängig) und
    hier hart prüfbar; GB/s/TFLOP/s/%-Peak müssen positiv und plausibel sein.
    """
    if not _has_cuda():
        print("  (übersprungen: keine CUDA-GPU)")
        return
    import tool_pipeline.run as R
    from tool_pipeline.schema import RunConfig
    from tool_pipeline.store import store as st

    orig = st.append_result
    st.append_result = lambda r, path=None: None
    try:
        res = R.run(RunConfig(dim_sizes={"i": 512, "k": 512, "j": 512}))
    finally:
        st.append_result = orig
    assert res.status == "ok", f"status={res.status} error={res.error}"
    m = res.metrics
    for k in ("tflops", "gbps", "arithmetic_intensity", "percent_peak_flops", "percent_peak_bw"):
        assert k in m, f"metrics fehlt {k}: {m}"
    assert m["arithmetic_intensity"] == 128.0, m
    assert m["tflops"] > 0 and m["gbps"] > 0, m
    assert 0 < m["percent_peak_flops"] <= 100.0, m


def test_baselines_cublas_naive_real():
    """cuBLAS + naive-cuTile werden mitgemessen; Obergrenze ≥ Untergrenze.

    cuBLAS ist die hochoptimierte Bibliothek (Obergrenze), der naive 16³-Kernel
    die untunte cuTile-Variante (Untergrenze) — beide positiv, cuBLAS ≥ naive.
    """
    if not _has_cuda():
        print("  (übersprungen: keine CUDA-GPU)")
        return
    import tool_pipeline.run as R
    from tool_pipeline.schema import RunConfig
    from tool_pipeline.store import store as st

    orig = st.append_result
    st.append_result = lambda r, path=None: None
    try:
        res = R.run(RunConfig(dim_sizes={"i": 512, "k": 512, "j": 512},
                              baselines=["cublas", "naive"]))
    finally:
        st.append_result = orig
    assert res.status == "ok", f"status={res.status} error={res.error}"
    bl = res.metrics["baselines"]
    assert bl["cublas"]["available"] and bl["naive"]["available"], bl
    cub, nai = bl["cublas"]["tflops"], bl["naive"]["tflops"]
    assert cub > 0 and nai > 0, bl
    assert cub >= nai, f"cuBLAS (Obergrenze) sollte ≥ naive (Untergrenze) sein: {bl}"


def test_baselines_fp8_graceful():
    """fp8-Lauf mit Baselines kippt nicht: naive läuft, cuBLAS ist entweder
    verfügbar oder sauber als nicht verfügbar markiert (kein Crash)."""
    if not _has_cuda():
        print("  (übersprungen: keine CUDA-GPU)")
        return
    import tool_pipeline.run as R
    from tool_pipeline.schema import RunConfig
    from tool_pipeline.store import store as st

    orig = st.append_result
    st.append_result = lambda r, path=None: None
    try:
        res = R.run(RunConfig(dim_sizes={"i": 256, "k": 256, "j": 256},
                              dtype="fp8e4m3", acc_dtype="fp16",
                              baselines=["cublas", "naive"]))
    finally:
        st.append_result = orig
    assert res.status == "ok", f"status={res.status} error={res.error}"
    bl = res.metrics["baselines"]
    assert "available" in bl["cublas"] and "available" in bl["naive"], bl


def test_run_provenance_has_gpu_state():
    """run() legt den GPU-Zustand pro Lauf in provenance ab (auf diesem Host
    mit nvidia-smi → nicht leer, mit numerischem sm_clock_mhz)."""
    if not _has_cuda():
        print("  (übersprungen: keine CUDA-GPU)")
        return
    import tool_pipeline.run as R
    from tool_pipeline.schema import RunConfig
    from tool_pipeline.store import store as st

    orig = st.append_result
    st.append_result = lambda r, path=None: None
    try:
        res = R.run(RunConfig(dim_sizes={"i": 256, "k": 256, "j": 256}))
    finally:
        st.append_result = orig
    assert res.status == "ok", f"status={res.status} error={res.error}"
    assert "gpu_state" in res.provenance, res.provenance
    gs = res.provenance["gpu_state"]
    assert isinstance(gs, dict) and gs, gs                 # nvidia-smi ist hier da
    assert gs.get("sm_clock_mhz") is not None, gs


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
