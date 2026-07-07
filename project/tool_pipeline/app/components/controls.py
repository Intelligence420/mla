"""Controls-Sidebar: **einsum-Ausdruck** (Presets + Freitext + Größen je Index),
die zu vergleichenden **Zahlenformate**, Kachelung/Swizzle, Baselines, Run/Cancel.

Enthält die Dash-freie, **headless-testbare** Naht-Logik:

* ``expr_indices(expr) -> [str]``                          — eindeutige Indizes.
* ``resolve_expr(expr) -> str``                            — expliziter Ausdruck (impliziter Output ergänzt).
* ``validate_expr(expr) -> str | None``                    — Ausdruck strukturell prüfen.
* ``index_size_inputs(expr, values=None) -> [Component]``  — Größenfeld je Index (dynamisch).
* ``dim_sizes_from_state(ids, values) -> dict``            — Pattern-Matching-State → Roh-dict.
* ``validate_dim_sizes(expr, dim_sizes) -> str | None``    — Größen + Speicher-Obergrenze prüfen.
* ``config_from_controls(expr, dim_sizes) -> RunConfig``   — eine RunConfig.
* ``configs_from_selection(expr, dim_sizes, sel, ...) -> [RunConfig]`` — je (dtype, acc) eine.
* ``validate_selection/tile/baselines/swizzle``            — die übrigen Achsen.

Die (dtype→acc)-Kombis (``COMBOS``) werden aus ``schema.ALLOWED_ACC`` abgeleitet
(Single Source of Truth → kein Drift): unzulässige Acc-Kombis existieren dadurch
gar nicht in der Auswahl.

Naht-Regel: importiert nur ``schema`` und den **torch-freien** IR-Helfer
``intermediate_representation.parse`` (für Klassifikation/Validierung des
Ausdrucks — **kein** run/torch/cuda), damit der Haupt-Prozess CUDA-frei
(fork-sicher) bleibt. Die IDs sind als Konstanten exportiert, damit
``callbacks.py`` sie importiert statt Strings zu duplizieren.
"""

from __future__ import annotations

import math
from typing import Optional

import dash_bootstrap_components as dbc
from dash import html

from ...intermediate_representation.parse import parse
from ...schema import ALLOWED_ACC, RunConfig

# --- Komponenten-IDs (von callbacks.py importiert) ---------------------------
ID_PRESET = "sel-preset"          # Preset-Dropdown (füllt den Ausdruck)
ID_EXPR = "in-expr"               # einsum-Ausdruck (Freitext, Source of Truth)
ID_EXPR_INFO = "expr-info"        # aufgelöster Output / Klassifikation / Fehler
ID_INDEX_SIZES = "index-sizes"    # Container der dynamischen Größenfelder je Index
INDEX_SIZE_TYPE = "index-size"    # Pattern-Matching-Typ der Größenfelder
ID_DTYPES = "sel-dtypes"          # Multi-Select der zu vergleichenden Formate
ID_DTYPE_INFO = "dtypes-info"     # Info-Marker (Tooltip: erklärt 'links → rechts')
ID_TILE_TM, ID_TILE_TN, ID_TILE_TK = "sel-tm", "sel-tn", "sel-tk"  # Tile-Dropdowns
ID_SWIZZLE = "chk-swizzle"        # L2-Swizzle-Toggle
ID_TILE_INFO = "tile-info"        # Info-Marker (Tooltip: Tile/Swizzle erklärt)
ID_BASELINES = "sel-baselines"    # Multi-Select der Vergleichs-Baselines
ID_BASELINE_INFO = "baselines-info"
ID_RUN, ID_CANCEL = "btn-run", "btn-cancel"
ID_PROGRESS_WRAP = "run-progress-wrap"   # Track (äußere Hülle) — Sichtbarkeit via running=
ID_PROGRESS, ID_STATUS = "run-progress", "run-status"   # Füllbalken (innen) · Statuszeile

# Progress-Balken als eigener Div-Balken (bewusst NICHT dbc.Progress): die Füllbreite
# wird direkt per ``style`` gesetzt — genau wie der Statustext (ein Div, der zuverlässig
# aktualisiert). Das rendert unabhängig von dbc/Bootstrap und vermeidet, dieselbe
# Komponente gleichzeitig in ``running=`` (style) UND ``progress=`` (value) zu haben,
# was die Live-Updates des Balkens verschluckte.
_PROG_TRACK_BASE = {"marginTop": "12px", "height": "8px", "background": "#e4e7ec",
                    "borderRadius": "4px", "overflow": "hidden"}
PROG_TRACK_HIDE = {**_PROG_TRACK_BASE, "display": "none"}    # vor dem ersten Lauf
PROG_TRACK_SHOW = {**_PROG_TRACK_BASE, "display": "block"}   # während + nach dem Lauf


def prog_fill_style(pct) -> dict:
    """Style des Füllbalkens für einen Fortschritt in Prozent (auf 0..100 geklemmt).
    Sanfter Breiten-Übergang → der Balken *wächst* sichtbar statt zu springen."""
    p = max(0, min(100, int(pct)))
    return {"width": f"{p}%", "height": "100%", "background": "#5b21b6",
            "borderRadius": "4px", "transition": "width .3s ease"}

# --- Ausdruck: Presets + Defaults --------------------------------------------
_DEFAULT_EXPR = "ik,kj->ij"        # Plain-GEMM (= RunConfig-Default)
_DEFAULT_INDEX_SIZE = 64           # Startwert je Index (klein/deterministisch; geteilte Maschine)
# Speicher-Obergrenze (geschätzter Peak-Traffic) — schützt die geteilte Maschine vor OOM.
_MAX_TENSOR_BYTES = 8 * 2**30      # 8 GiB

# Kuratierte Presets (Label → Ausdruck). Deckt die Kontraktions-Familie ab:
# Plain-GEMM, Batched, transponiert, mehrdim. M, allgemeine Tensor-Kontraktion.
PRESETS = [
    ("GEMM   ik,kj->ij", "ik,kj->ij"),
    ("Batched GEMM   bik,bkj->bij", "bik,bkj->bij"),
    ("A transponiert   ki,kj->ij", "ki,kj->ij"),
    ("Mehrdim. M   ijk,kl->ijl", "ijk,kl->ijl"),
    ("Tensor-Kontraktion   acspx,bspy->abcyx", "acspx,bspy->abcyx"),
]

# Anzeige-Reihenfolge der wählbaren Compute-dtypes. fp32-plain (Anker ohne
# Tensor-Cores) ist baubar/verifizierbar, aber bewusst NICHT in der GUI-Auswahl.
_DTYPE_ORDER = ("fp16", "bf16", "tf32", "fp8e4m3", "fp8e5m2")

# Wählbare Tile-Werte (Zweierpotenzen).
_TILE_M_OPTIONS = (32, 64, 128, 256)
_TILE_N_OPTIONS = (32, 64, 128, 256)
_TILE_K_OPTIONS = (16, 32, 64, 128)

# Vergleichs-Baselines (kanonische Namen = measure.baselines.KNOWN_BASELINES).
_BASELINE_OPTIONS = [
    {"label": "cuBLAS (Obergrenze)", "value": "cublas"},
    {"label": "naive-cuTile (Untergrenze)", "value": "naive"},
]
_BASELINE_KEYS = {"cublas", "naive"}

# L2-Swizzle-Modus: aus / an / beide.
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


# Wählbare (dtype, acc)-Kombis: aus ALLOWED_ACC abgeleitet (kein Drift).
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
# Reine, testbare Naht-Logik (Dash-frei) — Ausdruck
# ---------------------------------------------------------------------------
def expr_indices(expr: str) -> list[str]:
    """Eindeutige Indizes eines einsum-Ausdrucks (in Auftritts-Reihenfolge über
    die Operanden). Reine String-Logik (keine Validierung)."""
    e = (expr or "").replace(" ", "")
    lhs = e.split("->", 1)[0] if "->" in e else e
    seen: list[str] = []
    for ch in lhs.replace(",", ""):
        if ch not in seen:
            seen.append(ch)
    return seen


def resolve_expr(expr: str) -> str:
    """Ausdruck in **explizite** Form bringen (impliziten Output nach einsum-
    Konvention ergänzen). Erwartet einen strukturell gültigen Ausdruck."""
    e = (expr or "").replace(" ", "")
    if "->" in e:
        return e
    ir = parse(e, {d: 2 for d in expr_indices(e)})   # liefert den impliziten Output
    return f"{','.join(ir.inputs)}->{ir.output}"


def validate_expr(expr: str) -> Optional[str]:
    """Prüfe den Ausdruck **strukturell** (genau 2 Operanden, keine Diagonalen,
    gültiger Output) via `parse` mit Dummy-Größen. :returns: Fehlertext oder None."""
    if not expr or not expr.strip():
        return "Bitte einen einsum-Ausdruck eingeben (z. B. ik,kj->ij)."
    idx = expr_indices(expr)
    if not idx:
        return "Der Ausdruck enthält keine Indizes."
    try:
        parse(expr, {d: 2 for d in idx})
    except (ValueError, NotImplementedError) as e:
        return str(e)
    return None


def index_categories(expr: str) -> dict[str, str]:
    """Index → Kategorie-Label (M/N/K/Batch) via `parse` (Dummy-Größen). Leeres
    dict, wenn der Ausdruck (noch) nicht gültig ist."""
    try:
        ir = parse(expr, {d: 2 for d in expr_indices(expr)})
    except (ValueError, NotImplementedError):
        return {}
    cat: dict[str, str] = {}
    for d in ir.batch_dims:
        cat[d] = "Batch"
    for d in ir.m_dims:
        cat[d] = "M"
    for d in ir.n_dims:
        cat[d] = "N"
    for d in ir.k_dims:
        cat[d] = "K"
    return cat


def dim_sizes_from_state(ids, values) -> dict:
    """Pattern-Matching-State (Liste von ``{'index': d}``-dicts + Werte) → Roh-dict
    ``{d: value}`` (unkonvertiert — die Validierung/Coercion macht der Aufrufer)."""
    out: dict = {}
    for i, v in zip(ids or [], values or []):
        d = i.get("index") if isinstance(i, dict) else None
        if d is not None:
            out[d] = v
    return out


def validate_dim_sizes(expr: str, dim_sizes: dict) -> Optional[str]:
    """Prüfe die Größen je Index (positive ganze Zahl, alle vorhanden) und eine
    **Speicher-Obergrenze** (OOM-Schutz auf der geteilten Maschine).

    :returns: deutscher Fehlertext oder ``None`` (ok). Erwartet einen bereits
              strukturell gültigen Ausdruck (sonst zuerst `validate_expr`).
    """
    idx = expr_indices(expr)
    sizes: dict[str, int] = {}
    for d in idx:
        v = (dim_sizes or {}).get(d)
        if v is None or v == "":
            return f"Größe für Index '{d}' fehlt — bitte eine positive ganze Zahl eingeben."
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return f"Größe für Index '{d}' ist keine Zahl: {v!r}."
        if not math.isfinite(fv):
            return f"Größe für Index '{d}' muss endlich sein (bekommen: {v!r})."
        if fv != int(fv):
            return f"Größe für Index '{d}' muss ganzzahlig sein (bekommen: {v!r})."
        if int(fv) < 1:
            return f"Größe für Index '{d}' muss ≥ 1 sein (bekommen: {int(fv)})."
        sizes[d] = int(fv)
    # Struktur + fusionierte Größen (parse validiert erneut streng).
    try:
        ir = parse(expr, sizes)
    except (ValueError, NotImplementedError) as e:
        return str(e)
    # Speicher-Obergrenze: grober Peak-Traffic (fp16-Inputs 2 B, fp32-Output 4 B).
    est = 2 * ir.B * (ir.M * ir.K + ir.K * ir.N) + 4 * ir.B * ir.M * ir.N
    if est > _MAX_TENSOR_BYTES:
        return (f"Zu groß: ~{est / 2**30:.1f} GiB geschätzt (Grenze "
                f"{_MAX_TENSOR_BYTES // 2**30} GiB) — OOM-Risiko auf der geteilten "
                f"Maschine. Bitte kleinere Größen wählen.")
    return None


def config_from_controls(expr: str, dim_sizes: dict) -> RunConfig:
    """Ausdruck + Größen → eine ``RunConfig`` (fp16→fp32-Default; von der
    Einzellauf-Naht weiterbenutzt). Erwartet validierte Eingaben; coerct tolerant
    über ``float`` und normalisiert den Ausdruck auf die explizite Form."""
    idx = expr_indices(expr)
    ds = {d: int(float(dim_sizes[d])) for d in idx}
    return RunConfig(expr=resolve_expr(expr), dim_sizes=ds)


def validate_selection(selection) -> Optional[str]:
    """Prüfe die Format-Auswahl (Liste von ``combo_key``-Strings)."""
    if not selection:
        return "Bitte mindestens ein Zahlenformat für den Vergleich auswählen."
    unknown = [s for s in selection if s not in _VALID_KEYS]
    if unknown:
        return f"Unbekannte Format-Auswahl: {unknown}."
    return None


def validate_tile(tm, tn, tk) -> Optional[str]:
    """Prüfe die Tile-Auswahl (TM/TN/TK) gegen die zulässigen Zweierpotenzen."""
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
    """Prüfe die Baseline-Auswahl (Teilmenge von ``cublas``/``naive``)."""
    if not baselines:
        return None
    unknown = [b for b in baselines if b not in _BASELINE_KEYS]
    if unknown:
        return f"Unbekannte Baseline-Auswahl: {unknown}."
    return None


def swizzles_from_value(v) -> list:
    """Swizzle-Steuerwert → Liste der zu messenden Swizzle-Zustände."""
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
    """TM/TN/TK (evtl. als Strings aus den Dropdowns) → Tile-dict für ``RunConfig``."""
    return {"TM": int(float(tm)), "TN": int(float(tn)), "TK": int(float(tk))}


def configs_from_selection(expr, dim_sizes, selection,
                           tile=None, swizzle=False, baselines=None) -> list[RunConfig]:
    """Ausdruck + Größen + Format-Auswahl (+ Tile/Swizzle/Baselines) → eine
    ``RunConfig`` je gewählter (dtype, acc)-Kombi.

    Die Liste ist in kanonischer ``COMBOS``-Reihenfolge (deterministisch, erstes
    Element = primäres Format). **Ein festes Tile** gilt für die ganze Auswahl;
    ``swizzle='both'`` erzeugt je Format zwei Configs (ohne + mit Swizzle).
    Erwartet vorher validierte Eingaben.
    """
    norm_expr = resolve_expr(expr)
    idx = expr_indices(expr)
    sizes = {d: int(float(dim_sizes[d])) for d in idx}
    bl = list(baselines) if baselines else []
    sw_list = swizzles_from_value(swizzle)
    chosen = set(selection)
    out: list[RunConfig] = []
    for (d, a) in COMBOS:
        if combo_key(d, a) not in chosen:
            continue
        for si, s in enumerate(sw_list):
            kwargs = dict(expr=norm_expr, dim_sizes=dict(sizes), dtype=d, acc_dtype=a,
                          swizzle=s, baselines=list(bl) if si == 0 else [])
            if tile is not None:
                kwargs["tile"] = dict(tile)
            out.append(RunConfig(**kwargs))
    return out


# ---------------------------------------------------------------------------
# Dash-Komponentenbaum
# ---------------------------------------------------------------------------
def _preset_select() -> html.Div:
    """Preset-Dropdown: setzt den Ausdruck (Komfort; der Freitext bleibt maßgeblich)."""
    return html.Div([
        html.Label("Preset", style=_LABEL),
        dbc.Select(id=ID_PRESET, value=_DEFAULT_EXPR,
                   options=[{"label": lbl, "value": e} for lbl, e in PRESETS]),
    ])


def _expr_input() -> html.Div:
    """Freitext-einsum-Ausdruck (Source of Truth). '->' optional (impliziter Output)."""
    return html.Div([
        html.Label("einsum-Ausdruck", style=_LABEL),
        dbc.Input(id=ID_EXPR, type="text", value=_DEFAULT_EXPR, debounce=True,
                  placeholder="z. B. bik,bkj->bij",
                  style={"fontFamily": "ui-monospace, monospace"}),
    ])


def index_size_inputs(expr: str, values: Optional[dict] = None) -> list:
    """Ein Größen-Eingabefeld je Index des Ausdrucks (Pattern-Matching-ID), mit
    Kategorie-Label (M/N/K/Batch). ``values`` erhält bereits eingegebene Größen."""
    values = values or {}
    cats = index_categories(expr)
    fields = []
    for d in expr_indices(expr):
        label = d + (f"  ({cats[d]})" if d in cats else "")
        fields.append(html.Div(
            style={"flex": "1 1 64px", "minWidth": "64px"},
            children=[
                html.Label(label, style={**_LABEL, "fontFamily": "ui-monospace, monospace"}),
                dbc.Input(id={"type": INDEX_SIZE_TYPE, "index": d}, type="number",
                          value=values.get(d, _DEFAULT_INDEX_SIZE), min=1, step=1, debounce=True),
            ],
        ))
    return fields


def _dtype_header() -> list:
    info = html.Span(
        " ⓘ", id=ID_DTYPE_INFO,
        style={"cursor": "help", "color": "#8b5cf6", "fontWeight": 700, "textTransform": "none"},
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
    options = [{"label": combo_label(d, a), "value": combo_key(d, a)} for (d, a) in COMBOS]
    return dbc.Checklist(
        id=ID_DTYPES, options=options, value=list(_DEFAULT_SELECTION),
        style={"fontSize": "13px"}, inputStyle={"marginRight": "6px"},
    )


def _tile_dropdown(id_: str, label: str, options: tuple, default: int) -> html.Div:
    return html.Div(
        style={"flex": 1},
        children=[
            html.Label(label, style=_LABEL),
            dbc.Select(id=id_, value=str(default),
                       options=[{"label": str(o), "value": str(o)} for o in options]),
        ],
    )


def _tile_header() -> list:
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
    t = RunConfig().tile
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
            html.Label("L2-Swizzle", style=_LABEL),
            dbc.RadioItems(id=ID_SWIZZLE, options=_SWIZZLE_OPTIONS, value="off",
                           inline=True, style={"fontSize": "13px"},
                           inputStyle={"marginRight": "5px"}, labelStyle={"marginRight": "14px"}),
        ], style={"marginTop": "10px"}),
    ])


def _baseline_header() -> list:
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
    return dbc.Checklist(
        id=ID_BASELINES, options=_BASELINE_OPTIONS, value=[],
        style={"fontSize": "13px"}, inputStyle={"marginRight": "6px"},
    )


def build_controls() -> html.Div:
    """Sidebar-Inhalt: Ausdruck (Preset + Freitext + Größen je Index) + Format-
    Auswahl + Kachelung/Swizzle + Baselines + Run/Cancel + Progress."""
    return html.Div([
        html.H2("Operation", style={**_H2, "marginTop": 0}),
        _preset_select(),
        _expr_input(),
        # Aufgelöster Output / Klassifikation / Fehler (vom Callback gefüllt).
        html.Div(id=ID_EXPR_INFO, style={"fontSize": "12px", "margin": "6px 0 0",
                                         "minHeight": "16px"}),

        html.H2("Größen je Index", style=_H2),
        html.Div(id=ID_INDEX_SIZES,
                 style={"display": "flex", "flexWrap": "wrap", "gap": "8px"},
                 children=index_size_inputs(_DEFAULT_EXPR)),

        *_dtype_header(),
        _dtype_select(),

        *_tile_header(),
        _tile_select(),

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

        # Track (Hülle) + Füllbalken. running= blendet den Track ein und lässt ihn
        # stehen; progress= setzt die Füllbreite (prog_fill_style) live je Schritt.
        html.Div(id=ID_PROGRESS_WRAP, style=PROG_TRACK_HIDE,
                 children=html.Div(id=ID_PROGRESS, style=prog_fill_style(0))),
        html.Div(id=ID_STATUS, children="", style={"marginTop": "6px", "fontSize": "12px",
                                                    "color": "#6b7280", "minHeight": "16px"}),
    ])
