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


def render_context(result: RunResult):
    """Dezente Provenienz-Zeile (Größen · Format · GPU · Zeitstempel); None wenn leer."""
    p = result.provenance or {}
    s = p.get("sizes") or {}
    parts = []
    if s:
        parts.append(f"M={s.get('M', '?')} · N={s.get('N', '?')} · K={s.get('K', '?')}")
    if p.get("dtype"):
        parts.append(f"{p.get('dtype')} → {p.get('acc_dtype')}")
    if p.get("gpu"):
        parts.append(str(p["gpu"]))
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


def render_kpis(result: RunResult):
    """Drei KPI-Karten: Durchsatz (TFLOP/s), Laufzeit-Median (ms), Compile (ms)."""
    met, tim = result.metrics or {}, result.timing or {}
    iters = tim.get("bench_iters")
    return dbc.Row(
        [
            dbc.Col(_kpi_card("Durchsatz", _fmt(met.get("tflops"), ".2f"), "TFLOP/s"), md=4),
            dbc.Col(_kpi_card("Laufzeit (Median)", _fmt(tim.get("run_ms"), ".4f"), "ms",
                              sub=(f"{iters} Iterationen" if isinstance(iters, int) else None)), md=4),
            dbc.Col(_kpi_card("Compile (Kalt-Lauf)", _fmt(tim.get("compile_ms"), ".1f"), "ms"), md=4),
        ],
        className="g-3 mb-3",
    )


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
