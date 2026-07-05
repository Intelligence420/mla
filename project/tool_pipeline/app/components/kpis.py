"""KPI-Karten + Verify-/Status-Anzeige (TZ 2 / TODO 4).

Reine ``RunResult -> Dash-Komponente``-Funktionen, Dash-frei testbar
(``tests/test_app_render.py``). Sie müssen **jeden** Status sauber rendern
(``ok`` / ``verify_failed`` / ``compile_error`` / ``run_error``): fehlende Werte
werden zu „—" statt zu einem Crash. Der Background-Callback (TODO 6) komponiert
Status → Kontext → KPIs → Verify → Code (siehe ``code_panel.py``) in den Main-Bereich.

Naht-Regel (README): importiert nur ``tool_pipeline.schema`` (RunResult) —
kein run/torch/cuda.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from ...schema import RunResult

_MUTED = {"color": "#6b7280"}

_STATUS_LABEL = {
    "ok": "Lauf erfolgreich — verifiziert und gemessen.",
    "verify_failed": "Verifikation fehlgeschlagen — Zahlen weichen von der fp32-Referenz ab.",
    "compile_error": "Compile-Fehler — Kernel/Config nicht baubar.",
    "run_error": "Laufzeit-Fehler — Kernel crasht beim Launch/Messen.",
}
_STATUS_COLOR = {"ok": "success", "verify_failed": "danger",
                 "compile_error": "danger", "run_error": "danger"}


def _fmt(x, spec: str, default: str = "—") -> str:
    """Zahl formatieren, sonst „—" (bool ist kein KPI-Wert → ausgeschlossen)."""
    return format(x, spec) if isinstance(x, (int, float)) and not isinstance(x, bool) else default


def render_status(result: RunResult):
    """Statusbanner (grün bei ok, rot bei Fehler) + ggf. Fehlertext."""
    color = _STATUS_COLOR.get(result.status, "secondary")
    children = [html.Strong(_STATUS_LABEL.get(result.status, result.status))]
    if result.error:
        children += [html.Br(), html.Span(result.error, style={"fontSize": "12.5px"})]
    return dbc.Alert(children, color=color, className="mb-3")


def _gpu_state_str(gs: dict) -> str | None:
    """GPU-Zustand (Takt/Temp/Power/Last) zu einer kompakten Zeichenkette; None wenn
    kein Feld gesetzt ist (nvidia-smi fehlte oder lieferte [N/A])."""
    if not isinstance(gs, dict):
        return None
    bits = []
    if isinstance(gs.get("sm_clock_mhz"), (int, float)):
        bits.append(f"{gs['sm_clock_mhz']:.0f} MHz")
    if isinstance(gs.get("temp_c"), (int, float)):
        bits.append(f"{gs['temp_c']:.0f} °C")
    if isinstance(gs.get("power_w"), (int, float)):
        bits.append(f"{gs['power_w']:.1f} W")
    if isinstance(gs.get("util_pct"), (int, float)):
        bits.append(f"{gs['util_pct']:.0f} % Last")
    return "GPU-Zustand: " + " · ".join(bits) if bits else None


def render_context(result: RunResult):
    """Dezente Provenienz-Zeile (Größen · Format · GPU · GPU-Zustand · Zeitstempel);
    None wenn leer."""
    p = result.provenance or {}
    s = p.get("sizes") or {}
    parts = []
    if s:
        parts.append(f"M={s.get('M', '?')} · N={s.get('N', '?')} · K={s.get('K', '?')}")
    if p.get("dtype"):
        parts.append(f"{p.get('dtype')} → {p.get('acc_dtype')}")
    if p.get("gpu"):
        parts.append(str(p["gpu"]))
    gpu_state = _gpu_state_str(p.get("gpu_state") or {})
    if gpu_state:
        parts.append(gpu_state)
    if p.get("timestamp"):
        parts.append(str(p["timestamp"]))
    if not parts:
        return None
    return html.Div(" · ".join(parts), style={**_MUTED, "fontSize": "12px", "marginBottom": "12px"})


def _kpi_card(label: str, value: str, unit: str | None = None, sub: str | None = None):
    val_line = [html.Span(value, style={"fontSize": "26px", "fontWeight": 700,
                                        "fontVariantNumeric": "tabular-nums"})]
    if unit:
        val_line.append(html.Span(f" {unit}", style={"fontSize": "13px", **_MUTED}))
    body = [html.Div(label, style={"fontSize": "12px", **_MUTED}),
            html.Div(val_line, style={"marginTop": "2px"})]
    if sub:
        body.append(html.Div(sub, style={"fontSize": "12px", **_MUTED, "marginTop": "2px"}))
    return dbc.Card(dbc.CardBody(body), style={"height": "100%"})


def _pct_sub(pct) -> str | None:
    """„X.X % vom Peak" oder None (fp32/fp64 ohne Tensor-Core-Peak → kein Wert)."""
    return (f"{pct:.1f} % vom Peak"
            if isinstance(pct, (int, float)) and not isinstance(pct, bool) else None)


def _dist_sub(tim: dict) -> str | None:
    """Verteilungs-Zeile der Median-Karte: min/p90/σ (falls da) + Iterationszahl."""
    parts = []
    if all(isinstance(tim.get(k), (int, float)) for k in ("min_ms", "p90_ms", "sigma_ms")):
        parts.append(f"min {tim['min_ms']:.4f} · p90 {tim['p90_ms']:.4f} · σ {tim['sigma_ms']:.4f} ms")
    it = tim.get("bench_iters")
    if isinstance(it, int):
        parts.append(f"{it} Iterationen")
    return " · ".join(parts) or None


def _baseline_cards(met: dict) -> list:
    """Optionale Vergleichs-Karten: Anteil an cuBLAS (Obergrenze) + Tuning-Speedup
    vs naive-cuTile (Untergrenze). Nur wenn die jeweilige Baseline verfügbar ist."""
    bl = met.get("baselines")
    tf = met.get("tflops")
    if not isinstance(bl, dict) or not isinstance(tf, (int, float)):
        return []
    cards = []
    cub = bl.get("cublas") or {}
    if cub.get("available") and isinstance(cub.get("tflops"), (int, float)) and cub["tflops"] > 0:
        cards.append(_kpi_card("Anteil an cuBLAS", f"{tf / cub['tflops'] * 100:.0f}", "%",
                               sub=f"cuBLAS {cub['tflops']:.1f} TFLOP/s (Obergrenze)"))
    nai = bl.get("naive") or {}
    if nai.get("available") and isinstance(nai.get("tflops"), (int, float)) and nai["tflops"] > 0:
        cards.append(_kpi_card("Tuning-Speedup", f"{tf / nai['tflops']:.1f}", "×",
                               sub=f"vs naive-cuTile {nai['tflops']:.1f} TFLOP/s"))
    return cards


def render_kpis(result: RunResult):
    """KPI-Karten (wrappen responsiv): Durchsatz (+%-Peak) · Laufzeit-Median
    (+min/p90/σ) · Compile · Bandbreite (+%-Peak-BW) · arithm. Intensität · optional
    Baseline-Vergleiche. Fehlende Werte → „—" (nie ein Crash)."""
    met, tim = result.metrics or {}, result.timing or {}
    cards = [
        _kpi_card("Durchsatz", _fmt(met.get("tflops"), ".2f"), "TFLOP/s",
                  sub=_pct_sub(met.get("percent_peak_flops"))),
        _kpi_card("Laufzeit (Median)", _fmt(tim.get("run_ms"), ".4f"), "ms",
                  sub=_dist_sub(tim)),
        _kpi_card("Compile (Kalt-Lauf)", _fmt(tim.get("compile_ms"), ".1f"), "ms"),
        _kpi_card("Bandbreite", _fmt(met.get("gbps"), ".1f"), "GB/s",
                  sub=_pct_sub(met.get("percent_peak_bw"))),
        _kpi_card("Arithm. Intensität", _fmt(met.get("arithmetic_intensity"), ".1f"), "FLOP/Byte"),
    ]
    cards += _baseline_cards(met)
    return dbc.Row([dbc.Col(c, md=4) for c in cards], className="g-3 mb-3")


def render_verify(result: RunResult):
    """Verify-Badge (PASS/FAIL) + max_abs_err/Toleranzen; neutral wenn keine Verifikation lief."""
    acc = result.accuracy or {}
    if not acc:
        return html.Div(
            [dbc.Badge("Verify —", color="secondary", className="me-2"),
             html.Span("keine Verifikation (Lauf vor dem Verify-Schritt abgebrochen)", style=_MUTED)],
            className="mb-3",
        )
    passed = bool(acc.get("passed"))
    detail = (f"max_abs_err = {_fmt(acc.get('max_abs_err'), '.3e')}   "
              f"(atol={acc.get('atol')}, rtol={acc.get('rtol')})")
    return html.Div(
        [dbc.Badge(f"Verify: {'PASS' if passed else 'FAIL'}",
                   color=("success" if passed else "danger"), className="me-2"),
         html.Span(detail, style={"fontSize": "13px"})],
        className="mb-3",
    )
