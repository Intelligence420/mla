"""Plotly-Charts für den Format-Vergleich (TZ 3): **Durchsatz je Format** (Balken)
und **Genauigkeit ↔ Durchsatz** (Scatter). Die log-log-Roofline kommt in TZ 5.

Reine Funktionen ``RunResult-Liste → plotly.Figure`` — Dash-frei, torch-/cuda-frei
(Naht-/Fork-Regel), damit sie headless (``tests/test_app_charts.py``) prüfbar sind
und der Haupt-Prozess CUDA-frei bleibt. Verdrahtet werden sie im Callback (TODO 7).

Nur **verifizierte** Läufe (``status == "ok"``) werden zu Punkten — Zahlen landen
erst im Chart, nachdem sie gegen fp32 geprüft wurden (verify-before-trust).

Farb-System (dataviz): eine feste kategoriale Palette (CVD-validiert), **eine Farbe
je Format** stabil über beide Charts (aus ``COMBOS`` abgeleitet → kein Drift, nie
zyklisch neu vergeben). Das **primäre/aktive Format** wird über die *Form*
hervorgehoben (Balken-Umrandung, größerer Marker), nicht über die Farbe.
"""

from __future__ import annotations

import math
from typing import Optional

import plotly.graph_objects as go

from .controls import COMBOS, combo_key, combo_label

# --- Farb-/Ink-Tokens (dataviz-Referenzpalette, Light-Surface) ---------------
_PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
            "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
# Feste Farbe je (dtype, acc)-Kombi — Reihenfolge = COMBOS (Entität, nicht Rang):
# ein Format hat in BEIDEN Charts dieselbe Farbe, unabhängig von der Auswahl.
_FORMAT_COLOR = {combo_key(d, a): _PALETTE[i % len(_PALETTE)] for i, (d, a) in enumerate(COMBOS)}
_FALLBACK = "#898781"

_INK = "#0b0b0b"          # Primär-Ink (Titel, Wert-Labels)
_INK2 = "#52514e"         # Sekundär-Ink (Achsentitel)
_MUTED = "#898781"        # Achsen-Ticks / Hinweise
_GRID = "#e1e0d9"         # Gitter (Haarlinie)
_AXIS = "#c3c2b7"         # Basislinie / Achse
_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
_REL_FLOOR = 1e-10        # rel_err==0 (perfekt) auf log-Achse darstellbar machen


# ---------------------------------------------------------------------------
# Datenextraktion
# ---------------------------------------------------------------------------
def _points(results) -> list[dict]:
    """Verifizierte Läufe → Punkt-dicts (in Eingabe-/kanonischer Reihenfolge).

    Nur ``status == "ok"`` mit endlichem tflops und vorhandenem rel_err — alles
    andere (compile_error/verify_failed/run_error) gehört NICHT in einen Chart.
    """
    pts = []
    for r in results or []:
        if getattr(r, "status", None) != "ok":
            continue
        cfg = getattr(r, "config", None) or {}
        dt, ac = cfg.get("dtype"), cfg.get("acc_dtype")
        met, acc = (getattr(r, "metrics", None) or {}), (getattr(r, "accuracy", None) or {})
        tf, rel = met.get("tflops"), acc.get("rel_err")
        if tf is None or rel is None or not math.isfinite(tf):
            continue
        key = combo_key(dt, ac)
        pts.append({
            "key": key, "label": combo_label(dt, ac),
            "tflops": float(tf), "rel_err": float(rel),
            "max_abs_err": acc.get("max_abs_err"),
            "color": _FORMAT_COLOR.get(key, _FALLBACK),
        })
    return pts


def _resolve_primary(pts: list[dict], primary_key: Optional[str]) -> Optional[str]:
    """Schlüssel des primären Formats: die Vorgabe, sonst der erste Punkt."""
    keys = [p["key"] for p in pts]
    if primary_key in keys:
        return primary_key
    return keys[0] if keys else None


def _style(fig: go.Figure, title: str, xaxis_title: str = "") -> None:
    """Gemeinsames Light-Theme: transparenter Grund, recessive Achsen/Gitter."""
    fig.update_layout(
        title=dict(text=title, font=dict(color=_INK, size=15), x=0.01, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, color=_INK2, size=12),
        margin=dict(l=12, r=16, t=44, b=40),
        xaxis_title=xaxis_title,
        legend=dict(font=dict(size=11), orientation="v", yanchor="top", y=1, x=1.02),
        hoverlabel=dict(font_size=12, font_family=_FONT),
    )
    for ax in (fig.update_xaxes, fig.update_yaxes):
        ax(gridcolor=_GRID, zerolinecolor=_AXIS, linecolor=_AXIS,
           tickfont=dict(color=_MUTED, size=11))


def _empty(msg: str) -> go.Figure:
    """Platzhalter-Figur, wenn (noch) keine verifizierten Punkte vorliegen."""
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, xref="paper", yref="paper",
                       x=0.5, y=0.5, font=dict(color=_MUTED, size=13))
    _style(fig, title="")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ---------------------------------------------------------------------------
# Chart 1: Durchsatz je Format (Balken)
# ---------------------------------------------------------------------------
def figure_throughput(results, primary_key: Optional[str] = None) -> go.Figure:
    """Horizontaler Balken: TFLOP/s je verifiziertem Format (schnellstes oben).

    Identität über die Achsen-Labels + Farbe; das primäre Format erhält eine
    Ink-Umrandung.
    """
    pts = _points(results)
    if not pts:
        return _empty("Noch keine verifizierten Läufe.")
    prim = _resolve_primary(pts, primary_key)
    pts = sorted(pts, key=lambda p: p["tflops"])  # aufsteigend → größtes oben (h-Balken)

    fig = go.Figure(go.Bar(
        x=[p["tflops"] for p in pts],
        y=[p["label"] for p in pts],
        orientation="h",
        marker=dict(
            color=[p["color"] for p in pts],
            line=dict(color=_INK, width=[2 if p["key"] == prim else 0 for p in pts]),
            cornerradius=4,
        ),
        text=[f"{p['tflops']:.1f}" for p in pts],
        textposition="outside", textfont=dict(color=_INK, size=11),
        cliponaxis=False,
        customdata=[[p["label"]] for p in pts],
        hovertemplate="%{customdata[0]}<br>%{x:.2f} TFLOP/s<extra></extra>",
    ))
    _style(fig, title="Durchsatz je Format", xaxis_title="TFLOP/s")
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(automargin=True)
    return fig


# ---------------------------------------------------------------------------
# Chart 2: Genauigkeit ↔ Durchsatz (Scatter)
# ---------------------------------------------------------------------------
def figure_accuracy_throughput(results, primary_key: Optional[str] = None) -> go.Figure:
    """Scatter: Durchsatz (x) gegen relativen Fehler (y, log). Unten-rechts =
    schnell UND genau. Eine Spur je Format (Legende = Identität); das primäre
    Format bekommt einen größeren, umrandeten Marker.
    """
    pts = _points(results)
    if not pts:
        return _empty("Noch keine verifizierten Läufe.")
    prim = _resolve_primary(pts, primary_key)

    # Identität über Legende + Hover (geteilte Format-Farben mit dem Balken-Chart)
    # statt Direkt-Labels: die würden am rechten Rand clippen und die Legende
    # doppeln. Das primäre Format bekommt einen größeren, umrandeten Marker.
    fig = go.Figure()
    for p in pts:
        is_prim = p["key"] == prim
        fig.add_trace(go.Scatter(
            x=[p["tflops"]], y=[max(p["rel_err"], _REL_FLOOR)],
            mode="markers", name=p["label"],
            marker=dict(
                color=p["color"], size=17 if is_prim else 11,
                line=dict(color=_INK if is_prim else "#ffffff", width=2 if is_prim else 1.5),
            ),
            customdata=[[p["label"], p["rel_err"], p["max_abs_err"]]],
            hovertemplate=("%{customdata[0]}<br>%{x:.2f} TFLOP/s<br>"
                           "rel. Fehler %{customdata[1]:.2e}<br>"
                           "max_abs %{customdata[2]:.2e}<extra></extra>"),
        ))
    _style(fig, title="Genauigkeit ↔ Durchsatz", xaxis_title="TFLOP/s  (→ schneller)")
    fig.update_xaxes(rangemode="tozero")
    # y-Achsentitel steht senkrecht → KEIN Pfeil (würde mitgedreht und zeigte
    # seitlich); stattdessen Worte, die unabhängig von der Drehung stimmen.
    fig.update_yaxes(type="log", title="rel. Fehler vs fp32 · kleiner = genauer")
    return fig


# ---------------------------------------------------------------------------
# Serverseitiger PNG-Export (für Report/Inspektion)
# ---------------------------------------------------------------------------
def save_png(fig: go.Figure, path: str, width: int = 820, height: int = 440,
             scale: int = 2) -> str:
    """Schreibe eine Figur als PNG (via kaleido) und gib den Pfad zurück.

    In der App exportiert der Nutzer clientseitig über den Kamera-Knopf der
    Plotly-Toolbar; diese Funktion ist der serverseitige Weg (z. B. für die
    Sphinx-Doku). Braucht das optionale Paket ``kaleido``.
    """
    fig.write_image(path, width=width, height=height, scale=scale)
    return path
