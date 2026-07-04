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
