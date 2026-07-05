"""Controls-Sidebar: Größen M/N/K, die zu vergleichenden **Zahlenformate**,
Run/Cancel + Progress, dazu die **read-only** Anzeige der festen Konfiguration.

Enthält die Dash-freie, **headless-testbare** Naht-Logik:

* ``validate_sizes(m, n, k) -> str | None``               — Größen-Prüfung.
* ``config_from_controls(m, n, k) -> RunConfig``          — Größen → eine
  RunConfig (fp16→fp32-Default; von der Einzellauf-Naht weiterbenutzt).
* ``configs_from_selection(m, n, k, sel) -> [RunConfig]`` — Größen + Format-
  Auswahl → eine RunConfig je (dtype, acc)-Kombi (Batch-Vergleich, TZ 3).
* ``validate_selection(sel) -> str | None``               — Auswahl-Prüfung.

Die (dtype→acc)-Kombis (``COMBOS``) werden aus ``schema.ALLOWED_ACC`` abgeleitet
(Single Source of Truth → kein Drift): unzulässige Acc-Kombis existieren dadurch
gar nicht in der Auswahl — die **Acc-Regeln sind durch Konstruktion erzwungen**.

Naht-Regel (README): importiert NUR ``tool_pipeline.schema`` — **kein**
run/torch/cuda, damit der Haupt-Prozess CUDA-frei (fork-sicher) bleibt. Die IDs
sind als Konstanten exportiert, damit ``callbacks.py`` sie importiert statt
Strings zu duplizieren.
"""

from __future__ import annotations

import math
from typing import Optional

import dash_bootstrap_components as dbc
from dash import html

from ...schema import ALLOWED_ACC, RunConfig

# --- Komponenten-IDs (von callbacks.py importiert) ---------------------------
ID_M, ID_N, ID_K = "in-m", "in-n", "in-k"
ID_DTYPES = "sel-dtypes"          # Multi-Select der zu vergleichenden Formate
ID_DTYPE_INFO = "dtypes-info"     # Info-Marker (Tooltip: erklärt 'links → rechts')
ID_TILE_TM, ID_TILE_TN, ID_TILE_TK = "sel-tm", "sel-tn", "sel-tk"  # Tile-Dropdowns
ID_SWIZZLE = "chk-swizzle"        # L2-Swizzle-Toggle
ID_TILE_INFO = "tile-info"        # Info-Marker (Tooltip: Tile/Swizzle erklärt)
ID_BASELINES = "sel-baselines"    # Multi-Select der Vergleichs-Baselines
ID_BASELINE_INFO = "baselines-info"
ID_RUN, ID_CANCEL = "btn-run", "btn-cancel"
ID_PROGRESS, ID_STATUS = "run-progress", "run-status"

# Feste (nicht wählbare) Werte = die RunConfig-Defaults selbst (Single Source of
# Truth). dtype/acc sind ab TZ 3 wählbar (siehe COMBOS) und daher NICHT mehr hier.
_DEFAULT = RunConfig()
_DEFAULT_SIZE = 512  # Startwert je Größe (= cli.py-Default; klein, deterministisch)

# Anzeige-Reihenfolge der wählbaren Compute-dtypes. fp32-plain (Anker ohne
# Tensor-Cores) ist baubar/verifizierbar, aber bewusst NICHT in der GUI-Auswahl
# (Diagnose-Format, nur programmatisch via RunConfig) → hier ausgelassen.
_DTYPE_ORDER = ("fp16", "bf16", "tf32", "fp8e4m3", "fp8e5m2")

# Wählbare Tile-Werte (Zweierpotenzen). TM/TN = Kantenlänge der Output-Kachel,
# TK = K-Schrittweite. Aus dem RunConfig-Default (128/128/64) + kleineren/größeren
# Zweierpotenzen — jede Kombi kompiliert + verifiziert (Tile-Matrix in test_codegen).
_TILE_M_OPTIONS = (32, 64, 128, 256)
_TILE_N_OPTIONS = (32, 64, 128, 256)
_TILE_K_OPTIONS = (16, 32, 64, 128)

# Vergleichs-Baselines (kanonische Namen = measure.baselines.KNOWN_BASELINES).
_BASELINE_OPTIONS = [
    {"label": "cuBLAS (Obergrenze)", "value": "cublas"},
    {"label": "naive-cuTile (Untergrenze)", "value": "naive"},
]
_BASELINE_KEYS = {"cublas", "naive"}

# L2-Swizzle-Modus: aus / an / beide (Vergleich ohne↔mit Swizzle nebeneinander).
_SWIZZLE_OPTIONS = [
    {"label": "aus", "value": "off"},
    {"label": "an", "value": "on"},
    {"label": "beide (Vergleich)", "value": "both"},
]
_SWIZZLE_KEYS = {"off", "on", "both"}


def combo_key(dtype: str, acc: str) -> str:
    """Kanonischer Checklist-Wert einer (dtype, acc)-Kombi."""
    return f"{dtype}:{acc}"


def parse_combo(key: str) -> tuple[str, str]:
    """Checklist-Wert → (dtype, acc) — Umkehrung von ``combo_key``."""
    dtype, _, acc = key.partition(":")
    return dtype, acc


def combo_label(dtype: str, acc: str) -> str:
    """Menschlich lesbares Label (z. B. ``'fp8e4m3 → fp16'``)."""
    return f"{dtype} → {acc}"


# Wählbare (dtype, acc)-Kombis: aus ALLOWED_ACC abgeleitet (kein Drift), pro
# dtype fp32 (genau/Anker) vor fp16 (schneller). Das ist die vollständige, durch
# Konstruktion regel-konforme Vergleichs-Auswahl.
COMBOS = [
    (d, a)
    for d in _DTYPE_ORDER
    for a in sorted(ALLOWED_ACC[d], key=lambda x: x != "fp32")
]
_VALID_KEYS = {combo_key(d, a) for (d, a) in COMBOS}

# Default-Auswahl: ein Spektrum über den Tradeoff (genau → schnell/ungenau).
_DEFAULT_SELECTION = [combo_key("fp16", "fp32"),
                      combo_key("tf32", "fp32"),
                      combo_key("fp8e4m3", "fp16")]

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


def validate_selection(selection) -> Optional[str]:
    """Prüfe die Format-Auswahl (Liste von ``combo_key``-Strings).

    :returns: deutscher Fehlertext, oder ``None`` (ok). Leere Auswahl und
              unbekannte Schlüssel werden abgelehnt.
    """
    if not selection:
        return "Bitte mindestens ein Zahlenformat für den Vergleich auswählen."
    unknown = [s for s in selection if s not in _VALID_KEYS]
    if unknown:
        return f"Unbekannte Format-Auswahl: {unknown}."
    return None


def validate_tile(tm, tn, tk) -> Optional[str]:
    """Prüfe die Tile-Auswahl (TM/TN/TK) gegen die zulässigen Zweierpotenzen.

    Zweite Verteidigungslinie: die Dropdowns bieten nur gültige Werte an, aber ein
    unzulässiger (z. B. programmatischer) Wert soll einen sauberen Fehler geben
    statt einen still nicht-baubaren Kernel. :returns: Fehlertext oder ``None``.
    """
    for name, v, allowed in (("TM", tm, _TILE_M_OPTIONS), ("TN", tn, _TILE_N_OPTIONS),
                             ("TK", tk, _TILE_K_OPTIONS)):
        if v is None or v == "":
            return f"{name} fehlt — bitte einen Kachelwert wählen."
        try:
            iv = int(float(v))
        except (TypeError, ValueError):
            return f"{name} ist keine Zahl: {v!r}."
        if iv not in allowed:
            return f"{name}={iv} ist kein zulässiger Kachelwert (erlaubt: {list(allowed)})."
    return None


def validate_baselines(baselines) -> Optional[str]:
    """Prüfe die Baseline-Auswahl (Teilmenge von ``cublas``/``naive``).

    Keine Baseline ist zulässig (optional). :returns: Fehlertext oder ``None``.
    """
    if not baselines:
        return None
    unknown = [b for b in baselines if b not in _BASELINE_KEYS]
    if unknown:
        return f"Unbekannte Baseline-Auswahl: {unknown}."
    return None


def swizzles_from_value(v) -> list:
    """Swizzle-Steuerwert → Liste der zu messenden Swizzle-Zustände.

    Nimmt den Modus-String der GUI (``"off"``/``"on"``/``"both"``), einen ``bool``
    (Rückwärtskompatibilität) oder eine Liste. ``"both"`` ⇒ ``[False, True]``
    (jedes Format zweimal: ohne UND mit Swizzle → A/B-Vergleich).
    """
    if isinstance(v, str):
        return {"off": [False], "on": [True], "both": [False, True]}.get(v, [False])
    if isinstance(v, (list, tuple)):
        return [bool(x) for x in v] or [False]
    return [bool(v)]


def validate_swizzle(v) -> Optional[str]:
    """Prüfe den Swizzle-Steuerwert; unbekannter Modus-String → Fehlertext."""
    if isinstance(v, str) and v not in _SWIZZLE_KEYS:
        return f"Unbekannter Swizzle-Modus {v!r} (erlaubt: {sorted(_SWIZZLE_KEYS)})."
    return None


def tile_from_controls(tm, tn, tk) -> dict:
    """TM/TN/TK (evtl. als Strings aus den Dropdowns) → Tile-dict für ``RunConfig``.

    Tolerant über ``float`` (nimmt "128"/128.0). Erwartet vorher ``validate_tile``.
    """
    return {"TM": int(float(tm)), "TN": int(float(tn)), "TK": int(float(tk))}


def configs_from_selection(m, n, k, selection,
                           tile=None, swizzle=False, baselines=None) -> list[RunConfig]:
    """M/N/K + Format-Auswahl (+ Tile/Swizzle/Baselines) → eine ``RunConfig`` je
    gewählter (dtype, acc)-Kombi.

    Die Liste ist in kanonischer ``COMBOS``-Reihenfolge (unabhängig von der
    Klick-Reihenfolge) — deterministisch, das erste Element ist das **primäre**
    Format für die KPI-Karten. Achsen-Zuordnung wie ``config_from_controls``
    (i=M, k=K, j=N). **Ein festes Tile** gilt für die ganze Auswahl (Design-
    Entscheidung TZ 4); ``tile=None`` ⇒ RunConfig-Default (128/128/64). ``swizzle``
    ist ein Modus (``"off"``/``"on"``/``"both"``, bool erlaubt): ``"both"`` erzeugt
    je Format **zwei** Configs (ohne + mit Swizzle) für den A/B-Vergleich.
    Erwartet vorher validierte Eingaben.
    """
    sizes = {"i": int(float(m)), "k": int(float(k)), "j": int(float(n))}
    bl = list(baselines) if baselines else []
    sw_list = swizzles_from_value(swizzle)   # bool|str|Liste → [False]/[True]/[False,True]
    chosen = set(selection)
    out: list[RunConfig] = []
    for (d, a) in COMBOS:
        if combo_key(d, a) not in chosen:
            continue
        for si, s in enumerate(sw_list):
            # Baselines sind swizzle-unabhängig (cuBLAS/naive kennen unseren Swizzle
            # nicht) → nur am ERSTEN Swizzle-Variant messen, kein Doppel-Aufwand.
            kwargs = dict(dim_sizes=dict(sizes), dtype=d, acc_dtype=a, swizzle=s,
                          baselines=list(bl) if si == 0 else [])
            if tile is not None:
                kwargs["tile"] = dict(tile)
            out.append(RunConfig(**kwargs))
    return out


# ---------------------------------------------------------------------------
# Dash-Komponentenbaum
# ---------------------------------------------------------------------------
def _fixed_config() -> html.Div:
    """Read-only Anzeige der (weiterhin) festen Konfiguration. dtype/acc (TZ 3)
    sowie Tile/Swizzle (TZ 4) sind jetzt wählbar und stehen daher NICHT mehr hier
    — fest bleibt nur der Ausdruck (allgemeine Kontraktion ist TZ 6)."""
    c = _DEFAULT
    rows = [
        ("Ausdruck", c.expr),
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


def _dtype_header() -> list:
    """Abschnitts-Überschrift der Format-Auswahl + Hover-Info-Tooltip, der die
    Schreibweise ``Compute-dtype → Akkumulator/Output`` erklärt (das ``→`` ist
    sonst nicht selbsterklärend)."""
    info = html.Span(
        " ⓘ", id=ID_DTYPE_INFO,
        style={"cursor": "help", "color": "#8b5cf6", "fontWeight": 700,
               "textTransform": "none"},
    )
    tip = dbc.Tooltip(
        [
            html.Div("Schreibweise:  Compute-dtype  →  Akkumulator/Output",
                     style={"fontWeight": 600, "marginBottom": "5px"}),
            html.Div("Links: Zahlenformat der Eingaben A, B, in dem gerechnet wird "
                     "(z. B. fp8e4m3 — 8-Bit)."),
            html.Div("Rechts: Format des Akkumulators (interne Zwischensumme) und "
                     "des Ergebnisses (z. B. fp16 — schneller, oder fp32 — genauer)."),
            html.Div("Regel: bf16/tf32 summieren immer in fp32; fp16/fp8 dürfen fp16 "
                     "oder fp32.", style={"marginTop": "5px", "opacity": 0.8}),
        ],
        target=ID_DTYPE_INFO, placement="right",
        style={"maxWidth": "330px", "textAlign": "left", "fontSize": "12px"},
    )
    return [html.H2(["Zahlenformate (Vergleich)", info], style=_H2), tip]


def _dtype_select() -> html.Div:
    """Multi-Select der zu vergleichenden (dtype→acc)-Formate. Die Acc-Regeln
    sind durch die Kombi-Liste erzwungen — unzulässige Kombis existieren nicht."""
    options = [{"label": combo_label(d, a), "value": combo_key(d, a)} for (d, a) in COMBOS]
    return dbc.Checklist(
        id=ID_DTYPES, options=options, value=list(_DEFAULT_SELECTION),
        style={"fontSize": "13px"}, inputStyle={"marginRight": "6px"},
    )


def _tile_dropdown(id_: str, label: str, options: tuple, default: int) -> html.Div:
    """Ein Tile-Dropdown (feste Zweierpotenzen; Wert als String)."""
    return html.Div(
        style={"flex": 1},
        children=[
            html.Label(label, style=_LABEL),
            dbc.Select(id=id_, value=str(default),
                       options=[{"label": str(o), "value": str(o)} for o in options]),
        ],
    )


def _tile_header() -> list:
    """Überschrift der Kachelung + Hover-Info-Tooltip (TM/TN/TK + Swizzle erklärt)."""
    info = html.Span(" ⓘ", id=ID_TILE_INFO,
                     style={"cursor": "help", "color": "#8b5cf6", "fontWeight": 700,
                            "textTransform": "none"})
    tip = dbc.Tooltip(
        [
            html.Div("Kachelung (Tiling)", style={"fontWeight": 600, "marginBottom": "5px"}),
            html.Div("TM×TN: Kantenlängen der Ausgabe-Kachel, die ein GPU-Block berechnet. "
                     "TK: Schrittweite entlang der Kontraktions-Dimension K."),
            html.Div("Ein festes Tile gilt für alle gewählten Formate.",
                     style={"marginTop": "5px", "opacity": 0.8}),
            html.Div("L2-Swizzle: ordnet die Block→Kachel-Zuordnung L2-freundlicher um "
                     "(gleiches Ergebnis, oft weniger Speicherverkehr). „beide“ misst "
                     "jedes Format ohne UND mit Swizzle → direkter A/B-Vergleich im Chart.",
                     style={"marginTop": "5px", "opacity": 0.8}),
        ],
        target=ID_TILE_INFO, placement="right",
        style={"maxWidth": "330px", "textAlign": "left", "fontSize": "12px"},
    )
    return [html.H2(["Kachelung (Tile)", info], style=_H2), tip]


def _tile_select() -> html.Div:
    """TM/TN/TK-Dropdowns nebeneinander + Swizzle-Toggle darunter."""
    t = _DEFAULT.tile
    return html.Div([
        html.Div(
            style={"display": "flex", "gap": "8px"},
            children=[
                _tile_dropdown(ID_TILE_TM, "TM", _TILE_M_OPTIONS, t["TM"]),
                _tile_dropdown(ID_TILE_TN, "TN", _TILE_N_OPTIONS, t["TN"]),
                _tile_dropdown(ID_TILE_TK, "TK", _TILE_K_OPTIONS, t["TK"]),
            ],
        ),
        html.Div([
            html.Label("L2-Swizzle (grouped-M)", style=_LABEL),
            dbc.RadioItems(id=ID_SWIZZLE, options=_SWIZZLE_OPTIONS, value="off",
                           inline=True, style={"fontSize": "13px"},
                           inputStyle={"marginRight": "5px"}, labelStyle={"marginRight": "14px"}),
        ], style={"marginTop": "10px"}),
    ])


def _baseline_header() -> list:
    """Überschrift der Baselines + Hover-Info-Tooltip (Ober-/Untergrenze erklärt)."""
    info = html.Span(" ⓘ", id=ID_BASELINE_INFO,
                     style={"cursor": "help", "color": "#8b5cf6", "fontWeight": 700,
                            "textTransform": "none"})
    tip = dbc.Tooltip(
        [
            html.Div("Vergleichs-Baselines", style={"fontWeight": 600, "marginBottom": "5px"}),
            html.Div("cuBLAS: hochoptimierte NVIDIA-Bibliothek (torch.matmul) = praktische "
                     "Obergrenze — „wie nah sind wir dran?“."),
            html.Div("naive-cuTile: derselbe Kernel mit winzigem Tile (ohne Tuning) = "
                     "Untergrenze — „was bringt das Tuning?“.", style={"marginTop": "5px"}),
            html.Div("Optional — jede zugeschaltete Baseline kostet je Format eine "
                     "zusätzliche Messung.", style={"marginTop": "5px", "opacity": 0.8}),
        ],
        target=ID_BASELINE_INFO, placement="right",
        style={"maxWidth": "330px", "textAlign": "left", "fontSize": "12px"},
    )
    return [html.H2(["Baselines (Vergleich)", info], style=_H2), tip]


def _baseline_select() -> html.Div:
    """Multi-Select der Vergleichs-Baselines (Default: keine → schneller Lauf)."""
    return dbc.Checklist(
        id=ID_BASELINES, options=_BASELINE_OPTIONS, value=[],
        style={"fontSize": "13px"}, inputStyle={"marginRight": "6px"},
    )


def build_controls() -> html.Div:
    """Sidebar-Inhalt: feste Config (read-only) + Größen + Kachelung/Swizzle +
    Format-Auswahl + Baselines + Run/Cancel + Progress."""
    return html.Div([
        html.H2("Operation (fest)", style={**_H2, "marginTop": 0}),
        _fixed_config(),

        html.H2("Dimensionen", style=_H2),
        _size_input(ID_M, "M  (Zeilen, Index i)"),
        _size_input(ID_N, "N  (Spalten, Index j)"),
        _size_input(ID_K, "K  (Kontraktion, Index k)"),

        *_tile_header(),
        _tile_select(),

        *_dtype_header(),
        _dtype_select(),

        *_baseline_header(),
        _baseline_select(),

        html.Div(
            style={"display": "flex", "gap": "8px", "marginTop": "18px"},
            children=[
                dbc.Button("▶  Vergleichen", id=ID_RUN, color="primary", n_clicks=0,
                           style={"flex": 1}),
                dbc.Button("Abbrechen", id=ID_CANCEL, color="secondary", outline=True,
                           n_clicks=0, disabled=True),
            ],
        ),

        # Determinater Fortschritt (Format i/N) + Statustext; Sichtbarkeit steuert
        # der Background-Callback über running=, Wert/Text über progress=. Start: verborgen.
        dbc.Progress(id=ID_PROGRESS, value=0, striped=True, animated=True,
                     style={"display": "none", "marginTop": "12px", "height": "8px"}),
        html.Div(id=ID_STATUS, children="", style={"marginTop": "6px", "fontSize": "12px",
                                                    "color": "#6b7280", "minHeight": "16px"}),
    ])
