"""Headless-Tests der reinen Render-Funktionen (TZ 2 / TODO 4).

Prüft, dass ``kpis``/``code_panel`` aus einem ``RunResult`` **jeden** Status sauber
zu Dash-Komponenten rendern (ok / verify_failed / compile_error / run_error) —
fehlende Werte werden zu „—", nichts wirft. Der Inhalt wird über einen rekursiven
Text-Extraktor aus dem Komponentenbaum geprüft (kein Server, kein GPU nötig).

Standalone (`python tests/test_app_render.py`, aus `project/`) und via pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_pipeline.app.components import code_panel, kpis  # noqa: E402
from tool_pipeline.schema import RunConfig, RunResult  # noqa: E402

# --- ein generierter Quelltext-Ausschnitt (steht für RunResult.kernel_source) ---
SRC = (
    "import cuda.tile as ct\n\n@ct.kernel\ndef gemm(A, B, C, M, N, K):\n"
    "    acc = ct.full((TM, TN), 0, dtype=ct.float32)\n    acc = ct.mma(a, b, acc)\n"
)
_PROV = {"gpu": "NVIDIA GB10", "dtype": "fp16", "acc_dtype": "fp32",
         "sizes": {"M": 512, "N": 512, "K": 512}, "timestamp": "2026-07-04T16:00:00"}


def _ok() -> RunResult:
    return RunResult(
        status="ok", config=RunConfig().to_dict(),
        kernel_path="results/kernels/ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64.py",
        accuracy={"max_abs_err": 1.7e-4, "passed": True, "atol": 0.01, "rtol": 0.001},
        timing={"compile_ms": 312.5, "run_ms": 0.0421, "bench_iters": 200},
        metrics={"tflops": 71.4}, provenance=dict(_PROV), error=None,
    )


_PROV_FULL = {**_PROV, "gpu_state": {"sm_clock_mhz": 2418.0, "mem_clock_mhz": None,
                                     "temp_c": 40.0, "power_w": 12.9, "util_pct": 1.0}}


def _ok_full() -> RunResult:
    """ok-Lauf mit dem VOLLEN TZ-4-Metriksatz (Verteilung, GB/s, %-Peak, arithm.
    Intensität, Baselines) + GPU-Zustand in der Provenienz."""
    return RunResult(
        status="ok", config=RunConfig().to_dict(), kernel_path="results/kernels/x.py",
        accuracy={"max_abs_err": 1.7e-4, "passed": True, "atol": 0.01, "rtol": 0.001},
        timing={"compile_ms": 52.9, "run_ms": 0.0246, "min_ms": 0.0243,
                "p90_ms": 0.0247, "sigma_ms": 0.0020, "bench_iters": 30},
        metrics={"tflops": 10.9, "gbps": 85.3, "arithmetic_intensity": 128.0,
                 "percent_peak_flops": 5.1, "percent_peak_bw": 31.3,
                 "baselines": {"cublas": {"available": True, "tflops": 11.8},
                               "naive": {"available": True, "tflops": 1.2}}},
        provenance=dict(_PROV_FULL), error=None,
    )


def _verify_failed() -> RunResult:
    return RunResult(
        status="verify_failed", config=RunConfig().to_dict(),
        kernel_path="results/kernels/x.py",
        accuracy={"max_abs_err": 999.0, "passed": False, "atol": 0.2, "rtol": 0.02},
        timing={"compile_ms": 300.0}, metrics={}, provenance=dict(_PROV),
        error="max_abs_err=999 überschreitet Toleranz (atol=0.2, rtol=0.02)",
    )


def _compile_error() -> RunResult:
    prov = {**_PROV, "sizes": {}}  # vor dem Parsen abgebrochen → keine Größen
    return RunResult(
        status="compile_error", config=RunConfig().to_dict(), kernel_path=None,
        accuracy={}, timing={}, metrics={}, provenance=prov,
        error="cuTile-JIT: TileError: unsupported something",
    )


def _run_error() -> RunResult:
    return RunResult(
        status="run_error", config=RunConfig().to_dict(), kernel_path="results/kernels/x.py",
        accuracy={"max_abs_err": 1e-4, "passed": True, "atol": 0.01, "rtol": 0.001},
        timing={"compile_ms": 305.0}, metrics={}, provenance=dict(_PROV),
        error="bench: RuntimeError: simulierter Launch-Crash",
    )


def _text(node) -> str:
    """Sammelt alle String-Blätter aus einem Dash-Komponentenbaum (rekursiv)."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, (list, tuple)):
        return " ".join(_text(n) for n in node)
    props = {}
    if hasattr(node, "to_plotly_json"):
        j = node.to_plotly_json()
        if isinstance(j, dict):
            props = j.get("props", {}) or {}
    out = [_text(props.get("children"))]
    for key in ("value", "label", "header", "children"):
        v = props.get(key)
        if isinstance(v, str):
            out.append(v)
    return " ".join(x for x in out if x)


def _renders(fn, result) -> str:
    """Render-Funktion aufrufen (darf nicht werfen) und den Text zurückgeben."""
    comp = fn(result)
    return _text(comp)


def test_all_renderers_survive_every_status():
    """Keine Render-Funktion wirft — über alle vier Stati."""
    for make in (_ok, _verify_failed, _compile_error, _run_error):
        r = make()
        for fn in (kpis.render_status, kpis.render_kpis, kpis.render_verify):
            fn(r)  # darf nicht werfen
        kpis.render_context(r)
        code_panel.render_code_panel(getattr(r, "kernel_source", None), r.kernel_path)


def test_ok_shows_metrics_and_success():
    r = _ok()
    assert "erfolgreich" in _renders(kpis.render_status, r)
    kt = _renders(kpis.render_kpis, r)
    assert "71.40" in kt and "TFLOP/s" in kt and "0.0421" in kt and "312.5" in kt and "200" in kt
    assert "PASS" in _renders(kpis.render_verify, r)


def test_verify_failed_shows_fail_and_error():
    r = _verify_failed()
    assert "FAIL" in _renders(kpis.render_verify, r)
    assert "9.990e+02" in _renders(kpis.render_verify, r)  # max_abs_err formatiert
    assert "überschreitet Toleranz" in _renders(kpis.render_status, r)


def test_compile_error_dashes_and_error_text():
    r = _compile_error()
    kt = _renders(kpis.render_kpis, r)
    assert kt.count("—") >= 3, f'erwartete „—" für fehlende KPIs, bekam: {kt!r}'
    assert "TileError" in _renders(kpis.render_status, r)
    # keine Accuracy → neutrale Verify-Zeile, kein Crash
    assert "keine Verifikation" in _renders(kpis.render_verify, r)


def test_run_error_partial_timing():
    r = _run_error()
    kt = _renders(kpis.render_kpis, r)
    assert "305.0" in kt          # compile_ms vorhanden
    assert "—" in kt              # run_ms/tflops fehlen
    assert "Launch-Crash" in _renders(kpis.render_status, r)


def test_kpis_shows_roofline_cards_and_distribution():
    """Der volle Metriksatz erscheint: GB/s, arithm. Intensität, %-Peak-sub und die
    Verteilung (min/p90/σ) auf der Median-Karte."""
    kt = _renders(kpis.render_kpis, _ok_full())
    assert "GB/s" in kt and "85.3" in kt, kt
    assert "FLOP/Byte" in kt and "128.0" in kt, kt
    assert "% vom Peak" in kt, kt                     # %-Peak als sub
    assert "min 0.0243" in kt and "σ" in kt, kt        # Verteilung
    assert "10.90" in kt, kt                           # Durchsatz weiter da


def test_kpis_shows_baseline_cards():
    """Baseline-Vergleiche: Anteil an cuBLAS (~92 %) + Tuning-Speedup (~9.1×)."""
    kt = _renders(kpis.render_kpis, _ok_full())
    assert "Anteil an cuBLAS" in kt and "92" in kt, kt   # 10.9/11.8*100 ≈ 92
    assert "Tuning-Speedup" in kt and "9.1" in kt, kt     # 10.9/1.2 ≈ 9.1


def test_context_shows_gpu_state():
    """render_context zeigt den GPU-Zustand (Takt/Temp/Power); [N/A]-Felder fehlen still."""
    ct = _renders(kpis.render_context, _ok_full())
    assert "GPU-Zustand" in ct and "2418 MHz" in ct and "40 °C" in ct and "12.9 W" in ct, ct


def test_code_panel_with_and_without_source():
    with_src = _text(code_panel.render_code_panel(SRC, "results/kernels/x.py"))
    assert "ct.mma" in with_src and "results/kernels/x.py" in with_src
    without = _text(code_panel.render_code_panel(None, None))
    assert "Kein generierter Kernel" in without


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
