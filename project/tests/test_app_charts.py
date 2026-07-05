"""Headless-Tests der Chart-Builder (TZ 3 / TODO 6).

Die Charts sind reine Funktionen ``RunResult-Liste → plotly.Figure`` und daher
ohne Dash-Server + ohne GPU prüfbar: nur verifizierte Läufe werden zu Punkten,
Farbe folgt dem Format (nicht dem Rang), das primäre Format ist hervorgehoben,
und der Leerfall bringt einen Platzhalter statt einen Crash.

Lauffähig standalone (``python tests/test_app_charts.py``, aus ``project/``) **und**
via pytest. Braucht ``plotly``/``dash``/``schema`` — KEIN torch/cuTile/GPU.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_pipeline.app.components.charts import (  # noqa: E402
    _FORMAT_COLOR,
    figure_accuracy_throughput,
    figure_throughput,
)
from tool_pipeline.app.components.controls import combo_key, combo_label  # noqa: E402
from tool_pipeline.schema import RunResult  # noqa: E402


def _ok(dtype, acc, tflops, rel, maxabs=1e-4) -> RunResult:
    return RunResult(
        status="ok",
        config={"dtype": dtype, "acc_dtype": acc, "dim_sizes": {"i": 256, "k": 256, "j": 256}},
        metrics={"tflops": tflops},
        accuracy={"rel_err": rel, "max_abs_err": maxabs, "passed": True},
    )


def _failed(dtype, acc) -> RunResult:
    return RunResult(status="verify_failed",
                     config={"dtype": dtype, "acc_dtype": acc},
                     accuracy={"passed": False})


# --- Beispiel-Läufe (drei ok + zwei nicht-ok) --------------------------------
def _mixed():
    return [
        _ok("fp16", "fp32", 18.5, 6.8e-7),
        _ok("tf32", "fp32", 7.0, 2.9e-4),
        _ok("fp8e4m3", "fp16", 19.8, 6.0e-4),
        _failed("bf16", "fp32"),          # verify_failed → NICHT im Chart
        RunResult(status="compile_error", config={"dtype": "fp8e5m2", "acc_dtype": "fp32"}),
    ]


def _ok_bl(dtype, acc, tflops, rel, cublas=None, naive=None, maxabs=1e-4) -> RunResult:
    """ok-Lauf inkl. optionaler Baseline-TFLOP/s (in metrics['baselines'])."""
    met = {"tflops": tflops, "gbps": 85.0, "arithmetic_intensity": 128.0,
           "percent_peak_flops": 5.0, "percent_peak_bw": 31.0}
    bl = {}
    if cublas is not None:
        bl["cublas"] = {"available": True, "tflops": cublas}
    if naive is not None:
        bl["naive"] = {"available": True, "tflops": naive}
    if bl:
        met["baselines"] = bl
    return RunResult(status="ok", config={"dtype": dtype, "acc_dtype": acc},
                     metrics=met, accuracy={"rel_err": rel, "max_abs_err": maxabs, "passed": True})


def test_throughput_single_series_without_baselines():
    """Ohne Baselines bleibt es EINE Balken-Serie (unveränderter TZ-3-Pfad)."""
    fig = figure_throughput(_mixed())
    assert len(fig.data) == 1 and fig.data[0].type == "bar"


def test_throughput_grouped_with_baselines():
    """Mit cuBLAS+naive → drei gruppierte Serien (cuTile/cuBLAS/naive), barmode=group."""
    results = [_ok_bl("fp16", "fp32", 18.5, 7e-7, cublas=20.0, naive=2.0),
               _ok_bl("tf32", "fp32", 7.0, 3e-4, cublas=8.0, naive=1.0)]
    fig = figure_throughput(results)
    assert [t.name for t in fig.data] == \
        ["cuTile (getunt)", "cuBLAS (Obergrenze)", "naive-cuTile (Untergrenze)"], [t.name for t in fig.data]
    assert fig.layout.barmode == "group"
    # cuTile-Serie trägt die Format-Farben; die Baseline-Serien NICHT (neutral).
    assert set(fig.data[0].marker.color) <= set(_FORMAT_COLOR.values())
    assert fig.data[1].marker.color not in _FORMAT_COLOR.values()


def test_throughput_grouped_only_available_baseline():
    """Nur cuBLAS zugeschaltet → zwei Serien (cuTile + cuBLAS), keine naive-Serie."""
    fig = figure_throughput([_ok_bl("fp16", "fp32", 18.5, 7e-7, cublas=20.0)])
    assert [t.name for t in fig.data] == ["cuTile (getunt)", "cuBLAS (Obergrenze)"]


def _ok_sw(dtype, acc, tflops, rel, swizzle, cublas=None, naive=None) -> RunResult:
    """ok-Lauf mit explizitem swizzle-Flag (+ optionale Baselines)."""
    met = {"tflops": tflops}
    bl = {}
    if cublas is not None:
        bl["cublas"] = {"available": True, "tflops": cublas}
    if naive is not None:
        bl["naive"] = {"available": True, "tflops": naive}
    if bl:
        met["baselines"] = bl
    return RunResult(status="ok",
                     config={"dtype": dtype, "acc_dtype": acc, "swizzle": swizzle},
                     metrics=met, accuracy={"rel_err": rel, "max_abs_err": 1e-4, "passed": True})


def test_throughput_swizzle_compare_grouped():
    """Beide Swizzle-Zustände je Format → gruppierte Serien 'ohne/mit Swizzle',
    barmode=group, mit-Swizzle schraffiert."""
    results = [_ok_sw("fp16", "fp32", 10.0, 7e-7, False),
               _ok_sw("fp16", "fp32", 11.5, 7e-7, True),
               _ok_sw("tf32", "fp32", 4.9, 3e-4, False),
               _ok_sw("tf32", "fp32", 5.2, 3e-4, True)]
    fig = figure_throughput(results)
    assert [t.name for t in fig.data][:2] == ["ohne Swizzle", "mit Swizzle"], [t.name for t in fig.data]
    assert fig.layout.barmode == "group"
    assert fig.data[1].marker.pattern.shape == "/"      # mit Swizzle = schraffiert


def test_throughput_swizzle_compare_with_baselines():
    """Swizzle-A/B + Baselines → beide Swizzle-Serien PLUS cuBLAS/naive-Serien."""
    results = [_ok_sw("fp16", "fp32", 10.0, 7e-7, False, cublas=12.0, naive=1.2),
               _ok_sw("fp16", "fp32", 11.5, 7e-7, True)]
    names = [t.name for t in figure_throughput(results).data]
    assert "ohne Swizzle" in names and "mit Swizzle" in names
    assert any("cuBLAS" in n for n in names) and any("naive" in n for n in names)


def test_throughput_all_swizzled_is_single_series():
    """Modus 'an' (alle Punkte swizzle=True) → KEIN A/B, eine Balken-Serie."""
    results = [_ok_sw("fp16", "fp32", 11.5, 7e-7, True),
               _ok_sw("tf32", "fp32", 5.2, 3e-4, True)]
    fig = figure_throughput(results)
    assert len(fig.data) == 1 and fig.data[0].type == "bar"


def test_throughput_one_bar_per_verified_run():
    """Nur die drei verifizierten Läufe werden zu Balken (2 nicht-ok ausgelassen)."""
    fig = figure_throughput(_mixed())
    bar = fig.data[0]
    assert bar.type == "bar" and len(bar.x) == 3, (bar.type, len(bar.x))
    assert set(bar.y) == {combo_label("fp16", "fp32"), combo_label("tf32", "fp32"),
                          combo_label("fp8e4m3", "fp16")}


def test_throughput_sorted_fastest_on_top():
    """Horizontaler Balken aufsteigend sortiert → größtes tflops zuletzt (=oben)."""
    fig = figure_throughput(_mixed())
    assert list(fig.data[0].x) == sorted(fig.data[0].x)


def test_throughput_primary_has_outline_others_none():
    """Nur der primäre Balken bekommt eine Ink-Umrandung (width=2), Rest 0."""
    prim = combo_key("tf32", "fp32")
    fig = figure_throughput(_mixed(), primary_key=prim)
    bar = fig.data[0]
    widths = {lbl: w for lbl, w in zip(bar.y, bar.marker.line.width)}
    assert widths[combo_label("tf32", "fp32")] == 2
    assert all(w == 0 for lbl, w in widths.items() if lbl != combo_label("tf32", "fp32"))


def test_throughput_color_follows_format_not_rank():
    """Jeder Balken trägt die feste Format-Farbe (Entität, nicht Position)."""
    fig = figure_throughput(_mixed())
    bar = fig.data[0]
    for lbl, col in zip(bar.y, bar.marker.color):
        # Label 'd → a' zurück auf key mappen
        d, a = [s.strip() for s in lbl.split("→")]
        assert col == _FORMAT_COLOR[combo_key(d, a)], (lbl, col)


def test_scatter_one_trace_per_verified_format():
    """Eine Scatter-Spur je verifiziertem Format (Legende = Identität)."""
    fig = figure_accuracy_throughput(_mixed())
    assert len(fig.data) == 3
    assert {t.name for t in fig.data} == {combo_label("fp16", "fp32"),
                                          combo_label("tf32", "fp32"),
                                          combo_label("fp8e4m3", "fp16")}


def test_scatter_primary_marker_larger():
    """Das primäre Format hat den größeren, umrandeten Marker."""
    prim = combo_key("fp8e4m3", "fp16")
    fig = figure_accuracy_throughput(_mixed(), primary_key=prim)
    sizes = {t.name: t.marker.size for t in fig.data}
    assert sizes[combo_label("fp8e4m3", "fp16")] == 17
    assert all(s == 11 for n, s in sizes.items() if n != combo_label("fp8e4m3", "fp16"))


def test_scatter_yaxis_is_log():
    """rel_err spannt viele Größenordnungen → log-Y-Achse."""
    fig = figure_accuracy_throughput(_mixed())
    assert fig.layout.yaxis.type == "log"


def test_scatter_single_format_has_legend():
    """Ein-Format-Scatter erzwingt eine Legende (der Punkt wäre sonst unbeschriftet
    — auch im PNG-Export ohne Hover)."""
    fig = figure_accuracy_throughput([_ok("bf16", "fp32", 18.6, 4.8e-7)])
    assert fig.layout.showlegend is True
    assert len(fig.data) == 1 and fig.data[0].name == "bf16 → fp32"


def test_scatter_x_autoranges_but_bar_starts_at_zero():
    """Scatter-x NICHT bei 0 verankert (geclusterte Durchsätze brauchen Trennschärfe);
    der Durchsatz-Balken startet weiterhin bei 0."""
    scat = figure_accuracy_throughput([_ok("fp16", "fp32", 18.5, 7e-7),
                                       _ok("tf32", "fp32", 7.0, 3e-4)])
    assert scat.layout.xaxis.rangemode != "tozero"
    bar = figure_throughput([_ok("fp16", "fp32", 18.5, 7e-7)])
    assert bar.layout.xaxis.rangemode == "tozero"


def test_scatter_rel_err_zero_is_clamped():
    """rel_err == 0 (perfekt) darf die log-Achse nicht sprengen (Clamp > 0)."""
    fig = figure_accuracy_throughput([_ok("fp16", "fp32", 18.5, 0.0)])
    y = fig.data[0].y[0]
    assert y > 0, y


def test_empty_when_no_verified_runs():
    """Nur nicht-ok Läufe → beide Charts zeigen einen Platzhalter, keinen Crash."""
    only_failed = [_failed("fp16", "fp32")]
    for fig in (figure_throughput(only_failed), figure_accuracy_throughput(only_failed)):
        assert fig.data == ()
        assert fig.layout.annotations and "keine verifizierten" in fig.layout.annotations[0].text


def test_color_stable_across_different_selections():
    """Farbe folgt der Entität: fp8e4m3→fp16 hat in verschiedenen Auswahlen
    (und damit an anderer Position) dieselbe Farbe."""
    sel_a = figure_throughput([_ok("fp8e4m3", "fp16", 19.8, 6e-4)])
    sel_b = figure_throughput([_ok("fp16", "fp32", 18.5, 7e-7),
                               _ok("fp8e4m3", "fp16", 19.8, 6e-4)])
    key = combo_key("fp8e4m3", "fp16")
    col_a = dict(zip(sel_a.data[0].y, sel_a.data[0].marker.color))[combo_label("fp8e4m3", "fp16")]
    col_b = dict(zip(sel_b.data[0].y, sel_b.data[0].marker.color))[combo_label("fp8e4m3", "fp16")]
    assert col_a == col_b == _FORMAT_COLOR[key]


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
