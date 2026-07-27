"""Headless-Tests der Chart-Builder.

Die Charts sind reine Funktionen ``RunResult-Liste → plotly.Figure`` und daher
ohne Dash-Server + ohne GPU prüfbar: nur verifizierte Läufe werden zu Punkten,
Farbe folgt dem Format (nicht dem Rang), das primäre Format ist hervorgehoben,
und der Leerfall bringt einen Platzhalter statt einen Crash.

Lauffähig standalone (``python tests/test_app_charts.py``, aus ``project/``) **und**
via pytest. Braucht ``plotly``/``dash``/``schema`` — KEIN torch/cuTile/GPU.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_pipeline.app.components.charts import (  # noqa: E402
    _BASE_CUBLAS,
    _FORMAT_COLOR,
    _SERIES_CUTILE,
    _bw_slope,
    figure_accuracy_throughput,
    figure_roofline,
    figure_throughput,
)
from tool_pipeline.app.components.controls import combo_key, combo_label  # noqa: E402
from tool_pipeline.hardware import ridge_point  # noqa: E402
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
    # In gruppierten Charts kodiert die Farbe die SERIE (Format = y-Achse):
    # cuTile trägt die eine Serien-Farbe, cuBLAS die neutrale Baseline-Farbe.
    assert fig.data[0].marker.color == _SERIES_CUTILE
    assert fig.data[1].marker.color == _BASE_CUBLAS


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


# --- Roofline (TZ 5) ---------------------------------------------------------
def _roof(dtype, acc, tflops, ai, swz=False, rel=5e-7, pct=5.0) -> RunResult:
    """ok-Lauf mit den Roofline-Zutaten in metrics (tflops + arithmetic_intensity)."""
    return RunResult(
        status="ok", config={"dtype": dtype, "acc_dtype": acc, "swizzle": swz},
        metrics={"tflops": tflops, "arithmetic_intensity": ai,
                 "percent_peak_flops": pct, "gbps": 85.0},
        accuracy={"rel_err": rel, "max_abs_err": 1e-4, "passed": True})


def _roof_set():
    """Realistische GB10-Punkte: fp16 (AI 128, ohne+mit Swizzle), tf32 (AI ~85),
    fp8e4m3 (AI 256) + ein verify_failed (darf NICHT im Chart landen)."""
    return [_roof("fp16", "fp32", 19.0, 128.0),
            _roof("fp16", "fp32", 20.0, 128.0, swz=True),
            _roof("tf32", "fp32", 7.0, 85.33),
            _roof("fp8e4m3", "fp16", 19.9, 256.0),
            _failed("bf16", "fp32")]


def _markers(fig):
    return [t for t in fig.data if t.mode == "markers"]


def _lines(fig):
    return [t for t in fig.data if t.mode == "lines"]


def test_roofline_only_verified_points():
    """Nur ok-Läufe mit AI werden zu Punkten (verify_failed ausgelassen)."""
    assert len(_markers(figure_roofline(_roof_set()))) == 4


def test_roofline_points_at_ai_and_tflops():
    """Ein Punkt liegt an (arithm. Intensität, erreichte TFLOP/s)."""
    m = _markers(figure_roofline([_roof("tf32", "fp32", 7.0, 85.33)]))[0]
    assert m.x[0] == 85.33 and m.y[0] == 7.0


def test_roofline_ceilings_bundled_only_present():
    """Decken nur für vorkommende dtypes; fp16(213)+fp8(214) gebündelt, tf32(53) separat."""
    ceil = [t.name for t in _lines(figure_roofline(_roof_set())) if "Peak" in (t.name or "")]
    assert any("213–214" in n for n in ceil), ceil
    assert any("tf32-Peak ≈ 53" in n for n in ceil), ceil
    assert len(ceil) == 2, ceil          # keine Decke für nicht vorkommende dtypes


def test_roofline_bandwidth_slope_is_0273():
    """Bandbreiten-Schräge y = 0.273·x (273 GB/s) — Steigung an beiden Enden gleich."""
    slope = next(t for t in _lines(figure_roofline([_roof("fp16", "fp32", 19.0, 128.0)]))
                 if "Bandbreite" in (t.name or ""))
    for x, y in zip(slope.x, slope.y):
        assert math.isclose(y / x, _bw_slope(), rel_tol=1e-9)


def test_roofline_real_band_present():
    """Dezentes reales Bandbreiten-Band (70–85 %) als gefüllte Zone."""
    band = [t for t in figure_roofline([_roof("fp16", "fp32", 19.0, 128.0)]).data
            if getattr(t, "fill", None) == "tonexty"]
    assert band and "real erreichbar" in (band[0].name or ""), band


def test_roofline_axes_are_log():
    """AI und TFLOP/s spannen Größenordnungen → log-log-Achsen."""
    fig = figure_roofline([_roof("fp16", "fp32", 19.0, 128.0)])
    assert fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"


def test_roofline_ridge_visible_and_marked():
    """x-Bereich reicht bis zum Ridge (memory-bound-Aussage sichtbar); Ridge-Marker
    + Annotation für das Primärformat."""
    fig = figure_roofline([_roof("fp16", "fp32", 19.0, 128.0)],
                          primary_key=combo_key("fp16", "fp32"))
    assert 10 ** fig.layout.xaxis.range[1] >= ridge_point("fp16")   # Ridge im Sichtfeld
    ridge = next(t for t in _lines(fig) if t.name == "Ridge")
    assert math.isclose(ridge.x[0], ridge_point("fp16"), abs_tol=0.5)
    assert any("Ridge fp16" in (a.text or "") for a in fig.layout.annotations)


def test_roofline_swizzle_encoded_as_symbol():
    """Swizzle über Marker-Form (Kreis = ohne, Raute = mit) — keine neue Farbe."""
    fig = figure_roofline([_roof("fp16", "fp32", 19.0, 128.0, swz=False),
                           _roof("fp16", "fp32", 20.0, 128.0, swz=True)])
    syms = {t.name: t.marker.symbol for t in _markers(fig)}
    assert syms["fp16 → fp32"] == "circle" and syms["fp16 → fp32 · sw"] == "diamond", syms


def test_roofline_primary_marker_larger():
    """Das primäre Format bekommt den größeren, umrandeten Marker (wie im Scatter)."""
    sizes = {t.name: t.marker.size for t in
             _markers(figure_roofline(_roof_set(), primary_key=combo_key("fp16", "fp32")))}
    assert sizes["fp16 → fp32"] == 17
    assert all(s == 11 for n, s in sizes.items() if not n.startswith("fp16 → fp32"))


def test_roofline_color_follows_format():
    """Punktfarbe folgt dem Format (geteilte Palette mit den anderen Charts)."""
    fig = figure_roofline([_roof("fp8e4m3", "fp16", 19.9, 256.0)])
    assert _markers(fig)[0].marker.color == _FORMAT_COLOR[combo_key("fp8e4m3", "fp16")]


def test_roofline_empty_without_ai():
    """ok-Läufe OHNE arithmetische Intensität → Platzhalter statt irreführender Punkt."""
    fig = figure_roofline([_ok("fp16", "fp32", 18.5, 7e-7)])   # _ok setzt keine AI
    assert fig.data == () and "keine verifizierten" in fig.layout.annotations[0].text


def test_roofline_empty_when_no_verified_runs():
    """Nur nicht-ok Läufe → Platzhalter, kein Crash."""
    fig = figure_roofline([_failed("fp16", "fp32")])
    assert fig.data == () and fig.layout.annotations


def test_ridge_point_matches_known_numbers():
    """Ridge-Rechnung gegen bekannte Zahlen (DoD): BF16-Peak 213 → Ridge ≈ 780."""
    assert round(ridge_point("bf16")) == 780
    assert round(ridge_point("fp8e4m3")) == 784
    assert round(ridge_point("tf32")) == 194
    assert ridge_point("fp32") is None and ridge_point("fp64") is None


# --- TZ 7.5-2: Multi-Config (mehrere Tiles/GROUP_M) --------------------------
def _ok_cfg(dtype, acc, tflops, rel, tile=None, swizzle=False, group_m=8, ai=128.0) -> RunResult:
    """ok-Lauf mit explizitem Tile/Swizzle/GROUP_M (+ Roofline-AI) — für die
    Multi-Config-Disambiguierung."""
    cfg = {"dtype": dtype, "acc_dtype": acc, "swizzle": swizzle, "group_m": group_m}
    if tile is not None:
        cfg["tile"] = tile
    return RunResult(status="ok", config=cfg,
                     metrics={"tflops": tflops, "arithmetic_intensity": ai,
                              "percent_peak_flops": 5.0, "gbps": 85.0},
                     accuracy={"rel_err": rel, "max_abs_err": 1e-4, "passed": True})


def test_throughput_multi_tile_one_bar_per_config():
    """Zwei Tiles desselben Formats → ZWEI Balken (kein last-write-wins-Kollaps wie
    im alten _by_format); beide Zeilen tragen die Tile-Signatur."""
    r = [_ok_cfg("fp16", "fp32", 18.0, 7e-7, tile={"TM": 128, "TN": 128, "TK": 64}),
         _ok_cfg("fp16", "fp32", 15.0, 7e-7, tile={"TM": 64, "TN": 64, "TK": 32})]
    fig = figure_throughput(r)
    assert len(fig.data) == 1 and fig.data[0].type == "bar"
    assert len(fig.data[0].y) == 2, "zwei Tiles → zwei Balken"
    ys = list(fig.data[0].y)
    assert any("TM64/TN64/TK32" in lbl for lbl in ys) and any("TM128/TN128/TK64" in lbl for lbl in ys)


def test_scatter_multi_variant_distinct_symbols():
    """Zwei Tiles gleicher Format-Farbe → verschiedene Marker-SYMBOLE (kollisionsfrei),
    gleiche Farbe (Format), voll disambiguierte Namen."""
    r = [_ok_cfg("fp16", "fp32", 18.0, 7e-7, tile={"TM": 128, "TN": 128, "TK": 64}),
         _ok_cfg("fp16", "fp32", 15.0, 7e-7, tile={"TM": 64, "TN": 64, "TK": 32})]
    fig = figure_accuracy_throughput(r)
    assert len({t.marker.symbol for t in fig.data}) == 2, "Symbole müssen je Variante differieren"
    assert len({t.marker.color for t in fig.data}) == 1, "Farbe bleibt das Format"
    assert {t.name for t in fig.data} == {"fp16 → fp32 · TM128/TN128/TK64",
                                          "fp16 → fp32 · TM64/TN64/TK32"}


def test_roofline_multi_group_m_distinct():
    """Zwei GROUP_M (beide swizzle) desselben Formats/Tiles → verschiedene Symbole +
    disambiguierte Namen (der GROUP_M-Kanal, nicht nur Swizzle)."""
    tile = {"TM": 128, "TN": 128, "TK": 64}
    r = [_ok_cfg("fp16", "fp32", 19.0, 7e-7, tile=tile, swizzle=True, group_m=8),
         _ok_cfg("fp16", "fp32", 20.0, 7e-7, tile=tile, swizzle=True, group_m=16)]
    ms = [t for t in figure_roofline(r).data if t.mode == "markers"]
    assert len(ms) == 2 and len({t.marker.symbol for t in ms}) == 2
    assert {t.name for t in ms} == {"fp16 → fp32 · TM128/TN128/TK64 · sw G8",
                                    "fp16 → fp32 · TM128/TN128/TK64 · sw G16"}


def test_charts_single_tile_keeps_plain_names():
    """Regression: EINE Variante (auch mit explizitem tile/group_m) behält die
    schlichten Format-Namen — der Varianten-Kanal schaltet nur bei Mehrdeutigkeit zu."""
    r = [_ok_cfg("fp16", "fp32", 18.0, 7e-7, tile={"TM": 128, "TN": 128, "TK": 64})]
    assert figure_accuracy_throughput(r).data[0].name == "fp16 → fp32"


def test_scatter_multi_run_disambiguates_by_name():
    """History-Vergleich (TZ 7.5-4): dieselbe Config aus ZWEI Läufen → verschiedene
    Serien (Lauf-Name vorangestellt, verschiedene Symbole); Farbe bleibt das Format."""
    tile = {"TM": 128, "TN": 128, "TK": 64}
    a = _ok_cfg("fp16", "fp32", 18.0, 7e-7, tile=tile); a.run_name = "Lauf A"
    b = _ok_cfg("fp16", "fp32", 16.0, 7e-7, tile=tile); b.run_name = "Lauf B"
    fig = figure_accuracy_throughput([a, b])
    # Lauf-Name vorangestellt (+ Tile-Signatur, da gesetzt) → beide Serien eindeutig
    assert {t.name for t in fig.data} == {"Lauf A · fp16 → fp32 · TM128/TN128/TK64",
                                          "Lauf B · fp16 → fp32 · TM128/TN128/TK64"}
    assert len({t.marker.symbol for t in fig.data}) == 2   # je Lauf ein Symbol
    assert len({t.marker.color for t in fig.data}) == 1    # Farbe = Format (unverändert)


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
