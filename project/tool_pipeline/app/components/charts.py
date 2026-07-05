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

# Invariante „eine Farbe je Format": die Palette MUSS mindestens so viele Farben
# haben wie es Kombis gibt — sonst würde `i % len` still zwei Formate gleich
# einfärben. Lieber laut beim Import scheitern als leise die Farbidentität brechen.
assert len(COMBOS) <= len(_PALETTE), (
    f"_PALETTE hat nur {len(_PALETTE)} Farben für {len(COMBOS)} Format-Kombis — "
    f"Palette erweitern (sonst teilen sich zwei Formate eine Farbe)."
)

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

        # Baseline-TFLOP/s (falls mitgemessen + verfügbar) — additiv, für die
        # gruppierte Balken-Serie in figure_throughput. Fehlend → None (kein Balken).
        bl = met.get("baselines") or {}

        def _bl_tflops(name):
            e = bl.get(name) or {}
            v = e.get("tflops")
            return float(v) if e.get("available") and isinstance(v, (int, float)) else None

        pts.append({
            "key": key, "label": combo_label(dt, ac),
            "swizzle": bool(cfg.get("swizzle")),
            "tflops": float(tf), "rel_err": float(rel),
            "max_abs_err": acc.get("max_abs_err"),
            "gbps": met.get("gbps"),
            "percent_peak_flops": met.get("percent_peak_flops"),
            "cublas": _bl_tflops("cublas"),
            "naive": _bl_tflops("naive"),
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
# Baseline-Farben: bewusst NEUTRAL (kein Verbrauch der Format-Palette, die bei 8
# Kombis am Limit ist) — cuBLAS dunkelgrau, naive-cuTile hellgrau + Schraffur.
_BASE_CUBLAS = _INK2       # dunkelgrau = Obergrenze
_BASE_NAIVE = _MUTED       # hellgrau (+ Muster) = Untergrenze

# In GRUPPIERTEN Charts kodiert die Farbe die **Serie** (cuTile/ohne/mit Swizzle),
# nicht das Format — das Format steht auf der y-Achse. So ist die Legende ehrlich
# (ein Swatch = eine Serie) statt irreführend einfarbig bei mehrfarbigen Balken.
# Im Standard-Chart (eine Serie) bleibt die Farbe = Format (Kopplung zum Scatter).
_SERIES_CUTILE = _PALETTE[0]   # Blau = unser (getunter) cuTile-Kernel


def _subtitle(fig: go.Figure, text: str) -> None:
    """Dezente Unterzeile unter dem Titel (erklärt die Farb-/Rahmen-Kodierung)."""
    fig.add_annotation(text=text, showarrow=False, xref="paper", yref="paper",
                       x=0.0, y=1.05, xanchor="left", yanchor="bottom",
                       font=dict(color=_MUTED, size=10.5))


def figure_throughput(results, primary_key: Optional[str] = None) -> go.Figure:
    """Horizontaler Balken: TFLOP/s je verifiziertem Format (schnellstes oben).

    Ohne Baselines: eine Serie, Identität über Achsen-Label + Format-Farbe, das
    primäre Format mit Ink-Umrandung. Mit zugeschalteten Baselines: **gruppierte**
    Balken je Format — cuTile (Format-Farbe) neben cuBLAS (dunkelgrau, Obergrenze)
    und naive-cuTile (hellgrau/schraffiert, Untergrenze). Baselines verbrauchen
    KEINE Format-Farben (Palette am Limit) und werden per Ink/Muster abgesetzt.
    """
    pts = _points(results)
    if not pts:
        return _empty("Noch keine verifizierten Läufe.")
    prim = _resolve_primary(pts, primary_key)

    # Swizzle-A/B: liegen für Formate BEIDE Zustände vor → gruppierter Vergleich.
    swz_set = {p["swizzle"] for p in pts}
    if True in swz_set and False in swz_set:
        return _figure_throughput_swizzle(pts, prim)

    pts = sorted(pts, key=lambda p: p["tflops"])  # aufsteigend → größtes oben (h-Balken)

    has_baselines = any(p["cublas"] is not None or p["naive"] is not None for p in pts)
    if has_baselines:
        return _figure_throughput_grouped(pts, prim)

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
    fig.update_layout(bargap=0.4)                        # etwas Luft (nicht bildfüllend)
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(automargin=True)
    return fig


def _base_bar(name: str, labels: list, xs: list, color: str, hover: str,
              pattern: Optional[str] = None) -> go.Bar:
    """Eine Baseline-/Serien-Balkenspur (neutrale Farbe, mit Wert-Labels, damit
    die Zahl NICHT nur im Hover steht — auch winzige naive-Balken bleiben ablesbar)."""
    marker = dict(color=color, cornerradius=3)
    if pattern:
        marker["pattern"] = dict(shape=pattern, fgcolor="#ffffff", size=6)
    return go.Bar(
        name=name, y=labels, x=xs, orientation="h", marker=marker,
        text=[f"{v:.1f}" if v is not None else "" for v in xs],
        textposition="outside", textfont=dict(color=_INK2, size=10), cliponaxis=False,
        hovertemplate=hover + " · %{y}<br>%{x:.2f} TFLOP/s<extra></extra>",
    )


def _figure_throughput_grouped(pts: list[dict], prim: Optional[str]) -> go.Figure:
    """Gruppierte Balken cuTile vs Baselines (aufgerufen, wenn Baselines vorliegen).

    Farbe = Serie (cuTile blau, cuBLAS dunkelgrau, naive hellgrau/schraffiert);
    Format steht auf der y-Achse → ehrliche Legende. Alle Balken beschriftet."""
    labels = [p["label"] for p in pts]
    prim_line = dict(color=_INK, width=[2 if p["key"] == prim else 0 for p in pts])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="cuTile (getunt)", y=labels, x=[p["tflops"] for p in pts], orientation="h",
        marker=dict(color=_SERIES_CUTILE, line=prim_line, cornerradius=3),
        text=[f"{p['tflops']:.1f}" for p in pts], textposition="outside",
        textfont=dict(color=_INK, size=10), cliponaxis=False,
        hovertemplate="cuTile · %{y}<br>%{x:.2f} TFLOP/s<extra></extra>",
    ))
    if any(p["cublas"] is not None for p in pts):
        fig.add_trace(_base_bar("cuBLAS (Obergrenze)", labels, [p["cublas"] for p in pts],
                                _BASE_CUBLAS, "cuBLAS"))
    if any(p["naive"] is not None for p in pts):
        fig.add_trace(_base_bar("naive-cuTile (Untergrenze)", labels, [p["naive"] for p in pts],
                                _BASE_NAIVE, "naiv", pattern="/"))
    _style(fig, title="Durchsatz — cuTile vs Baselines", xaxis_title="TFLOP/s")
    _subtitle(fig, "Farbe = Serie (Legende) · Format = y-Achse · Rahmen = Primärformat")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.12,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
                      margin=dict(l=12, r=16, t=56, b=66))
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(automargin=True)
    return fig


def _by_format(pts: list[dict]) -> list[dict]:
    """Punkte je Format (dtype,acc) bündeln → ohne/mit-Swizzle + Baseline-Werte.

    Baselines sind swizzle-unabhängig → aus dem Punkt übernommen, der sie trägt.
    Reihenfolge: aufsteigend nach dem größeren der beiden Swizzle-Werte (h-Balken
    → schnellstes Format oben)."""
    fmt: dict = {}
    for p in pts:
        e = fmt.setdefault(p["key"], {"key": p["key"], "label": p["label"],
                                      "color": p["color"], "noswz": None, "swz": None,
                                      "cublas": None, "naive": None})
        e["swz" if p["swizzle"] else "noswz"] = p["tflops"]
        if p["cublas"] is not None:
            e["cublas"] = p["cublas"]
        if p["naive"] is not None:
            e["naive"] = p["naive"]
    return sorted(fmt.values(),
                  key=lambda e: max(v for v in (e["noswz"], e["swz"], 0.0) if v is not None))


def _figure_throughput_swizzle(pts: list[dict], prim: Optional[str]) -> go.Figure:
    """A/B-Vergleich ohne↔mit Swizzle je Format (gruppierte h-Balken).

    Serie = Farbe/Muster (ohne Swizzle = blau solid, mit Swizzle = blau schraffiert),
    Format steht auf der y-Achse → ehrliche Legende; optionale Baselines als neutrale
    Serien. Alle Balken beschriftet; primäres Format mit Ink-Umrandung."""
    rows = _by_format(pts)
    labels = [e["label"] for e in rows]
    prim_line = dict(color=_INK, width=[2 if e["key"] == prim else 0 for e in rows])
    noswz = [e["noswz"] for e in rows]
    swz = [e["swz"] for e in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="ohne Swizzle", y=labels, x=noswz, orientation="h",
        marker=dict(color=_SERIES_CUTILE, line=prim_line, cornerradius=3),
        text=[f"{v:.1f}" if v is not None else "" for v in noswz],
        textposition="outside", textfont=dict(color=_INK, size=10), cliponaxis=False,
        hovertemplate="ohne Swizzle · %{y}<br>%{x:.2f} TFLOP/s<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="mit Swizzle", y=labels, x=swz, orientation="h",
        marker=dict(color=_SERIES_CUTILE, line=prim_line, cornerradius=3,
                    pattern=dict(shape="/", fgcolor="#ffffff", size=6)),
        text=[f"{v:.1f}" if v is not None else "" for v in swz],
        textposition="outside", textfont=dict(color=_INK, size=10), cliponaxis=False,
        hovertemplate="mit Swizzle · %{y}<br>%{x:.2f} TFLOP/s<extra></extra>",
    ))
    if any(e["cublas"] is not None for e in rows):
        fig.add_trace(_base_bar("cuBLAS (Obergrenze)", labels, [e["cublas"] for e in rows],
                                _BASE_CUBLAS, "cuBLAS"))
    if any(e["naive"] is not None for e in rows):
        fig.add_trace(_base_bar("naive-cuTile (Untergrenze)", labels, [e["naive"] for e in rows],
                                _BASE_NAIVE, "naiv", pattern="x"))
    _style(fig, title="Durchsatz — L2-Swizzle-Vergleich (ohne ↔ mit)", xaxis_title="TFLOP/s")
    _subtitle(fig, "Serie = ohne/mit Swizzle (Muster) · Format = y-Achse · Rahmen = Primärformat")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.12,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
                      margin=dict(l=12, r=16, t=56, b=66))
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
        name = p["label"] + (" · sw" if p["swizzle"] else "")
        fig.add_trace(go.Scatter(
            x=[p["tflops"]], y=[max(p["rel_err"], _REL_FLOOR)],
            mode="markers", name=name,
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
    # Auch bei genau EINEM Format eine Legende (sonst wäre der einzelne Punkt
    # unbeschriftet — auch im PNG-Export ohne Hover).
    fig.update_layout(showlegend=True)
    # x NICHT bei 0 verankern: die Durchsätze clustern dicht (~18–20) mit einem
    # Ausreißer (tf32 ~7); Auto-Range gibt dem Scatter horizontale Trennschärfe.
    fig.update_xaxes(rangemode="normal")
    # y-Achsentitel steht senkrecht → KEIN Pfeil (würde mitgedreht); Worte statt
    # Pfeil. Feste Dekaden-Ticks (dtick=1) + 10^n-Format → ruhige, konsistente
    # log-Achse (kein 1µ/0.001-SI-Mix, keine nackten Minor-Mantissen).
    fig.update_yaxes(type="log", title="rel. Fehler vs fp32 · kleiner = genauer",
                     dtick=1, exponentformat="power", showexponent="all")
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
