"""Controls-Sidebar (TZ 2 / TODO 3): Größen M/N/K + Run/Cancel + Progress,
dazu die **read-only** Anzeige der in TZ 2 fest verdrahteten Konfiguration.

Enthält die Dash-freie, **headless-testbare** Naht-Logik:

* ``config_from_controls(m, n, k) -> RunConfig`` — Control-Werte → RunConfig
  (setzt NUR die Größen; family/expr/dtype/acc/tile/swizzle bleiben Default).
* ``validate_sizes(m, n, k) -> str | None``      — Eingabe-Prüfung; deutscher
  Fehlertext oder ``None`` (ok).

Diese beiden Funktionen sind bewusst Dash-frei und werden in
``tests/test_app_controls.py`` geprüft. ``build_controls()`` liefert das
Sidebar-Fragment für das Layout.

Naht-Regel (README): importiert NUR ``tool_pipeline.schema`` (RunConfig) —
**kein** run/torch/cuda, damit der Haupt-Prozess CUDA-frei (fork-sicher) bleibt.
Die IDs sind als Konstanten exportiert, damit ``callbacks.py`` (TODO 6) sie
importiert statt Strings zu duplizieren.
"""

from __future__ import annotations

import math
from typing import Optional

import dash_bootstrap_components as dbc
from dash import html

from ...schema import RunConfig

# --- Komponenten-IDs (von callbacks.py importiert) ---------------------------
ID_M, ID_N, ID_K = "in-m", "in-n", "in-k"
ID_RUN, ID_CANCEL = "btn-run", "btn-cancel"
ID_PROGRESS, ID_STATUS = "run-progress", "run-status"

# TZ-2-Fixwerte = die RunConfig-Defaults selbst (Single Source of Truth: ändert
# sich ein Default, ändert sich die Anzeige automatisch mit).
_DEFAULT = RunConfig()
_DEFAULT_SIZE = 512  # Startwert je Größe (= cli.py-Default; klein, deterministisch)

_H2 = {"fontSize": "11px", "letterSpacing": "0.08em", "textTransform": "uppercase",
       "color": "#6b7280", "margin": "18px 0 8px"}
_LABEL = {"display": "block", "fontSize": "12.5px", "color": "#6b7280", "margin": "10px 0 4px"}


# ---------------------------------------------------------------------------
# Reine, testbare Naht-Logik (Dash-frei)
# ---------------------------------------------------------------------------
def validate_sizes(m, n, k) -> Optional[str]:
    """Prüfe M/N/K; gib einen deutschen Fehlertext zurück oder ``None`` (ok).

    Akzeptiert nur **positive ganze Zahlen**. Robust gegen das, was ein
    Dash-Number-Input liefern kann: ``None``/"" (leer), Float (512.0) und
    Zahlen-Strings ("512"). Ganzzahligkeit wird echt geprüft (512.5 → Fehler).
    """
    for name, v in (("M", m), ("N", n), ("K", k)):
        if v is None or v == "":
            return f"{name} fehlt — bitte eine positive ganze Zahl eingeben."
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return f"{name} ist keine Zahl: {v!r}."
        # inf/nan bestehen float(), würden aber int() zum Werfen bringen (N1) →
        # früh abfangen, damit validate_sizes NIE eine Exception wirft.
        if not math.isfinite(fv):
            return f"{name} muss eine endliche Zahl sein (bekommen: {v!r})."
        if fv != int(fv):
            return f"{name} muss ganzzahlig sein (bekommen: {v!r})."
        if int(fv) < 1:
            return f"{name} muss ≥ 1 sein (bekommen: {int(fv)})."
    return None


def config_from_controls(m, n, k) -> RunConfig:
    """M/N/K → ``RunConfig``. Nur die Größen werden gesetzt; alles andere bleibt
    auf den TZ-2-Defaults (``ik,kj->ij``, fp16→fp32, Tile 128/128/64, kein Swizzle).

    Achsen-Zuordnung wie ``cli.build_config``: ``ik,kj->ij`` ⇒ i=M (Zeilen),
    k=K (Kontraktion), j=N (Spalten). Erwartet gültige Eingaben (vorher
    ``validate_sizes``); coerct tolerant über ``float`` (nimmt 512.0 / "512").
    """
    return RunConfig(dim_sizes={"i": int(float(m)), "k": int(float(k)), "j": int(float(n))})


# ---------------------------------------------------------------------------
# Dash-Komponentenbaum
# ---------------------------------------------------------------------------
def _fixed_config() -> html.Div:
    """Read-only Anzeige der festen TZ-2-Konfiguration (aus den Defaults)."""
    c, t = _DEFAULT, _DEFAULT.tile
    rows = [
        ("Ausdruck", c.expr),
        ("Format", f"{c.dtype} → {c.acc_dtype}"),
        ("Tile", f"TM={t['TM']} · TN={t['TN']} · TK={t['TK']}"),
        ("Swizzle", "an" if c.swizzle else "aus"),
    ]
    line = {"display": "flex", "justifyContent": "space-between",
            "fontSize": "12.5px", "padding": "3px 0"}
    return html.Div(
        style={"background": "#f6f4fe", "border": "1px dashed #c9b8f2",
               "borderRadius": "7px", "padding": "8px 10px"},
        children=[
            html.Div([html.Span(key, style={"color": "#6b7280"}),
                      html.Span(val, style={"fontWeight": 600, "fontFamily": "ui-monospace, monospace"})],
                     style=line)
            for key, val in rows
        ],
    )


def _size_input(id_: str, label: str) -> html.Div:
    return html.Div([
        html.Label(label, style=_LABEL),
        dbc.Input(id=id_, type="number", value=_DEFAULT_SIZE, min=1, step=1, debounce=True),
    ])


def build_controls() -> html.Div:
    """Sidebar-Inhalt: feste Config (read-only) + Größen + Run/Cancel + Progress."""
    return html.Div([
        html.H2("Operation (fest)", style={**_H2, "marginTop": 0}),
        _fixed_config(),

        html.H2("Dimensionen", style=_H2),
        _size_input(ID_M, "M  (Zeilen, Index i)"),
        _size_input(ID_N, "N  (Spalten, Index j)"),
        _size_input(ID_K, "K  (Kontraktion, Index k)"),

        html.Div(
            style={"display": "flex", "gap": "8px", "marginTop": "18px"},
            children=[
                dbc.Button("▶  Run", id=ID_RUN, color="primary", n_clicks=0,
                           style={"flex": 1}),
                dbc.Button("Abbrechen", id=ID_CANCEL, color="secondary", outline=True,
                           n_clicks=0, disabled=True),
            ],
        ),

        # Indeterminater Balken + Statustext; Sichtbarkeit/Text steuert der
        # Background-Callback (TODO 6) über running=/progress=. Startzustand: verborgen.
        dbc.Progress(id=ID_PROGRESS, value=100, striped=True, animated=True,
                     style={"display": "none", "marginTop": "12px", "height": "8px"}),
        html.Div(id=ID_STATUS, children="", style={"marginTop": "6px", "fontSize": "12px",
                                                    "color": "#6b7280", "minHeight": "16px"}),
    ])
