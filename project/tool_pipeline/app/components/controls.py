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

from ...intermediate_representation.parse import NAryContractionIR, parse
from ...schema import ALLOWED_ACC, RunConfig

# --- Komponenten-IDs (von callbacks.py importiert) ---------------------------
ID_FAMILY = "sel-family"          # Operations-Familie (contraction/elementwise/reduction)
ID_OP = "sel-op"                  # Elementwise-Op (add/mul/copy) — nur für Elementwise
ID_OP_WRAP = "op-wrap"            # Hülle der Op-Auswahl (Sichtbarkeit je Familie)
ID_EPILOG = "sel-epilog"          # Epilog-Fusion (bias/relu) — nur für Kontraktion (TZ 9)
ID_EPILOG_WRAP = "epilog-wrap"    # Hülle der Epilog-Auswahl (Sichtbarkeit je Familie)
ID_PRESET = "sel-preset"          # Preset-Dropdown (füllt den Ausdruck)
ID_EXPR = "in-expr"               # einsum-Ausdruck (Freitext, Source of Truth)
ID_EXPR_INFO = "expr-info"        # aufgelöster Output / Klassifikation / Fehler
ID_INDEX_SIZES = "index-sizes"    # Container der dynamischen Größenfelder je Index
INDEX_SIZE_TYPE = "index-size"    # Pattern-Matching-Typ der Größenfelder
ID_DTYPES = "sel-dtypes"          # Multi-Select der zu vergleichenden Formate
ID_DTYPE_INFO = "dtypes-info"     # Info-Marker (Tooltip: erklärt 'links → rechts')
ID_TILE_TM, ID_TILE_TN, ID_TILE_TK = "sel-tm", "sel-tn", "sel-tk"  # Tile-Dropdowns
ID_SWIZZLE = "chk-swizzle"        # (Rückfall) Einzel-Swizzle-Toggle — execute_run-Skalarpfad/Tests
ID_SWIZZLE_GROUP_M = "sel-group-m"  # (Rückfall) Einzel-GROUP_M — execute_run-Skalarpfad/Tests
# TZ 7.5-2: mehrere Tile-Zeilen (+/-) und mehrere Swizzle-Konfigurationen gegeneinander.
TILE_TM_TYPE, TILE_TN_TYPE, TILE_TK_TYPE = "tile-tm", "tile-tn", "tile-tk"  # Pattern-Matching je Tile-Zeile
TILE_RM_TYPE = "tile-rm"          # Entfernen-Button je Tile-Zeile (Pattern-Matching)
ID_TILE_ADD = "btn-tile-add"      # „+ Tile"-Button (Zeile hinzufügen)
ID_TILE_ROWS = "tile-rows"        # Container der dynamischen Tile-Zeilen
ID_SWIZZLE_CONFIGS = "chk-swizzle-configs"  # Mehrfachauswahl der Swizzle-Konfigurationen (aus / G…)
ID_TILE_INFO = "tile-info"        # Info-Marker (Tooltip: Tile/Swizzle erklärt)
ID_BASELINES = "sel-baselines"    # Multi-Select der Vergleichs-Baselines
ID_BASELINE_INFO = "baselines-info"
ID_BENCH_WARMUP, ID_BENCH_ITERS = "sel-warmup", "sel-iters"  # Mess-Einstellungen (einstellbar)
ID_BENCH_INFO = "bench-info"      # Info-Marker (Tooltip: Warmup/Iterationen erklärt)
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
# (Bleibt als (Label, Ausdruck)-Liste — die Kontraktions-Presets; FAMILY_PRESETS
# unten trägt zusätzlich die Op je Familie für die neuen memory-bound-Familien.)
PRESETS = [
    ("GEMM   ik,kj->ij", "ik,kj->ij"),
    ("Batched GEMM   bik,bkj->bij", "bik,bkj->bij"),
    ("A transponiert   ki,kj->ij", "ki,kj->ij"),
    ("Mehrdim. M   ijk,kl->ijl", "ijk,kl->ijl"),
    ("Tensor-Kontraktion   acspx,bspy->abcyx", "acspx,bspy->abcyx"),
    # n-är (TZ 7.5-3): Kettenprodukt → zwei paarweise GEMMs (Demonstrator).
    ("Kettenprodukt (n-är)   ij,jk,kl->il", "ij,jk,kl->il"),
]

# Die drei Operations-Familien (TZ 7). Label + Default-Ausdruck je Familie.
FAMILIES = [
    ("Kontraktion (GEMM, Tensor-Core)", "contraction"),
    ("Elementwise (memory-bound)", "elementwise"),
    ("Reduktion (memory-bound)", "reduction"),
]
_FAMILY_KEYS = {k for _, k in FAMILIES}
_MEMORY_BOUND = {"elementwise", "reduction"}

# Presets je Familie: (Label, Ausdruck, Op). Op=None für Kontraktion (steckt im
# GEMM); "sum" für Reduktion; add/mul/copy für Elementwise. Elementwise add/mul
# teilen den Ausdruck (ij,ij->ij) → die Op unterscheidet sie.
FAMILY_PRESETS = {
    "contraction": [(lbl, expr, None) for lbl, expr in PRESETS],
    "elementwise": [
        ("Add   ij,ij->ij", "ij,ij->ij", "add"),
        ("Mul   ij,ij->ij", "ij,ij->ij", "mul"),
        ("Copy   ij->ij", "ij->ij", "copy"),
        ("Add 3D   ijk,ijk->ijk", "ijk,ijk->ijk", "add"),
    ],
    "reduction": [
        ("Zeilensumme   ij->i", "ij->i", "sum"),
        ("Spaltensumme   ij->j", "ij->j", "sum"),
        ("Volle Summe   ij->", "ij->", "sum"),
    ],
}

# Elementwise-Op-Auswahl (nur bei family=elementwise sichtbar).
_OP_OPTIONS = [
    {"label": "add  (A + B)", "value": "add"},
    {"label": "mul  (A · B)", "value": "mul"},
    {"label": "copy (A)", "value": "copy"},
    {"label": "relu (max(A,0))", "value": "relu"},
]
_OP_KEYS = {"add", "mul", "copy", "relu"}

# Epilog-Fusion (TZ 9) — nur bei family=contraction sichtbar, spiegelt die Op-Auswahl.
# Der leere Wert "" = keine Fusion (⇒ RunConfig.epilog=None ⇒ byte-identischer
# Kernel/Slug wie TZ 1-8). Die Werte sind die Keys von
# ``codegen/templates/contraction._EPILOGS``.
_EPILOG_OPTIONS = [
    {"label": "keiner", "value": ""},
    {"label": "bias (acc + D)", "value": "bias"},
    {"label": "relu (max(acc,0))", "value": "relu"},
]
_EPILOG_KEYS = {"bias", "relu"}

# memory-bound-Familien rechnen ohne Tensor-Core → nur die arithmetisch nativen
# Formate (fp8-Arithmetik compiliert nicht, tf32 ist ein reines TC-Konzept). fp32
# ist hier ein vollwertiges Format (anders als bei der Kontraktion, wo fp32-plain
# bewusst aus der GUI-Auswahl fällt).
_MEMORY_BOUND_DTYPES = ("fp16", "bf16", "fp32")


def preset_value(expr: str, op: Optional[str]) -> str:
    """Preset-Dropdown-Wert = ``"<op>|<expr>"`` (op leer ⇒ Kontraktion). Trägt die
    Op mit, weil Elementwise add/mul denselben Ausdruck haben."""
    return f"{op or ''}|{expr}"


def parse_preset_value(value: str) -> tuple[str, Optional[str]]:
    """Umkehrung von ``preset_value`` → (expr, op|None)."""
    op, _, expr = (value or "").partition("|")
    return expr, (op or None)


def preset_options(family: str) -> list[dict]:
    """Preset-Optionen (label/value) für eine Familie."""
    return [{"label": lbl, "value": preset_value(expr, op)}
            for lbl, expr, op in FAMILY_PRESETS.get(family, [])]


def family_default_preset(family: str) -> str:
    """Preset-Default-Wert (erstes Preset) einer Familie."""
    lbl, expr, op = FAMILY_PRESETS[family][0]
    return preset_value(expr, op)

# Anzeige-Reihenfolge der wählbaren Compute-dtypes. fp32-plain (Anker ohne
# Tensor-Cores) ist baubar/verifizierbar, aber bewusst NICHT in der GUI-Auswahl.
_DTYPE_ORDER = ("fp16", "bf16", "tf32", "fp8e4m3", "fp8e5m2")

# Wählbare Tile-Werte (Zweierpotenzen).
_TILE_M_OPTIONS = (32, 64, 128, 256)
_TILE_N_OPTIONS = (32, 64, 128, 256)
_TILE_K_OPTIONS = (16, 32, 64, 128)

# Wählbare GROUP_M-Werte der L2-Swizzle-Rasterung (Zweierpotenzen). Default 8 = der
# bisher fest verdrahtete Wert (TZ 1-3, byte-identisch); 1 = plain-bid-Zuordnung
# („kein Swizzle"). GROUP_M wird zur Codegen-Zeit als Literal in den Swizzle-Kernel
# gebacken → fixer Deckel (num_pid_m=cdiv(M,TM) ist erst zur Laufzeit bekannt; die
# grouped-M-Rasterung klemmt eine partielle letzte Gruppe korrekt für JEDES GROUP_M≥1).
_SWIZZLE_GROUP_M_OPTIONS = (1, 2, 4, 8, 16, 32)

# Mess-Einstellungen: erlaubter Bereich (min, max). Iterationen ≥ 10 (unter 10 ist
# die Verteilung/das p90 kaum aussagekräftig); Obergrenze 500 hält die Laufzeit auf
# der geteilten Maschine im Rahmen. Warmup ab 0 (kein Aufwärmen zulassen).
_BENCH_WARMUP_RANGE = (0, 500)
_BENCH_ITERS_RANGE = (10, 500)

# Vergleichs-Baselines (kanonische Namen = measure.baselines.KNOWN_BASELINES).
_BASELINE_OPTIONS = [
    {"label": "cuBLAS (Obergrenze)", "value": "cublas"},
    {"label": "naive-cuTile (Untergrenze)", "value": "naive"},
]
_BASELINE_KEYS = {"cublas", "naive"}

# L2-Swizzle-Modus (Rückfall-Skalarform für execute_run/Tests): aus / an / beide.
_SWIZZLE_OPTIONS = [
    {"label": "aus", "value": "off"},
    {"label": "an", "value": "on"},
    {"label": "beide (Vergleich)", "value": "both"},
]
_SWIZZLE_KEYS = {"off", "on", "both"}

# TZ 7.5-2: Swizzle-Konfigurationen als Mehrfachauswahl — „aus" + je GROUP_M-Wert
# ein Eintrag „g<N>". Eine Auswahl ⇒ eine Menge zu vergleichender (swizzle, group_m).
_SWIZZLE_CONFIG_OPTIONS = ([{"label": "aus", "value": "off"}]
                           + [{"label": f"G{g}", "value": f"g{g}"} for g in _SWIZZLE_GROUP_M_OPTIONS])
_SWIZZLE_CONFIG_KEYS = {"off"} | {f"g{g}" for g in _SWIZZLE_GROUP_M_OPTIONS}
_DEFAULT_SWIZZLE_CONFIG = ["off"]   # Default = nur ohne Swizzle (wie bisher swizzle="off")


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

# memory-bound-Auswahl: die drei nativen Formate (fp16/bf16/fp32 → fp32).
_MEMORY_BOUND_SELECTION = [combo_key("fp16", "fp32"),
                           combo_key("bf16", "fp32"),
                           combo_key("fp32", "fp32")]


def combos_for_family(family: str) -> list[tuple[str, str]]:
    """Zulässige (dtype, acc)-Kombis je Familie. Kontraktion = alle COMBOS
    (ohne fp32-plain); memory-bound = nur fp16/bf16/fp32 (inkl. fp32)."""
    if family in _MEMORY_BOUND:
        return [(d, a) for d in _MEMORY_BOUND_DTYPES
                for a in sorted(ALLOWED_ACC[d], key=lambda x: x != "fp32")]
    return COMBOS


def dtype_options_for_family(family: str) -> list[dict]:
    """Checklist-Optionen (label/value) der Formate je Familie."""
    return [{"label": combo_label(d, a), "value": combo_key(d, a)}
            for (d, a) in combos_for_family(family)]


def default_selection_for_family(family: str) -> list[str]:
    """Default-Format-Auswahl je Familie."""
    return list(_MEMORY_BOUND_SELECTION if family in _MEMORY_BOUND else _DEFAULT_SELECTION)

# Sidebar-Typografie (Überschriften/Labels) liegt seit TZ 8 als CSS-Klassen
# ``.ctl-section`` / ``.ctl-label`` in ``assets/theme.css`` (konsolidiert, kein
# wiederholtes inline-style-dict mehr).


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


def resolve_expr(expr: str, family: str = "contraction") -> str:
    """Ausdruck in **explizite** Form bringen (impliziten Output nach einsum-
    Konvention ergänzen). Erwartet einen strukturell gültigen Ausdruck der Familie."""
    e = (expr or "").replace(" ", "")
    if "->" in e:
        return e
    ir = parse(e, {d: 2 for d in expr_indices(e)}, family=family)  # impliziter Output
    return f"{','.join(ir.inputs)}->{ir.output}"


def validate_expr(expr: str, family: str = "contraction") -> Optional[str]:
    """Prüfe den Ausdruck **strukturell** für die Familie via `parse` mit
    Dummy-Größen (Kontraktion: 2 Operanden/M-N-K; Elementwise: gleiche Form je
    Operand; Reduktion: 1 Operand + reduzierte Achse). :returns: Fehlertext/None."""
    if not expr or not expr.strip():
        return "Bitte einen einsum-Ausdruck eingeben (z. B. ik,kj->ij)."
    idx = expr_indices(expr)
    if not idx:
        return "Der Ausdruck enthält keine Indizes."
    try:
        parse(expr, {d: 2 for d in idx}, family=family)
    except (ValueError, NotImplementedError) as e:
        return str(e)
    return None


def index_categories(expr: str, family: str = "contraction") -> dict[str, str]:
    """Index → Kategorie-Label via `parse` (Dummy-Größen), family-abhängig:
    Kontraktion M/N/K/Batch, Elementwise „elem", Reduktion „bleibt"/„Σ". Leeres
    dict, wenn der Ausdruck (noch) nicht gültig ist."""
    try:
        ir = parse(expr, {d: 2 for d in expr_indices(expr)}, family=family)
    except (ValueError, NotImplementedError):
        return {}
    cat: dict[str, str] = {}
    if isinstance(ir, NAryContractionIR):
        # n-är: Output-Indizes bleiben, der Rest wird über die Kette kontrahiert (Σ).
        out_set = set(ir.output)
        for d in expr_indices(expr):
            cat[d] = "bleibt" if d in out_set else "Σ"
        return cat
    if family == "elementwise":
        for d in ir.axes:
            cat[d] = "elem"
    elif family == "reduction":
        for d in ir.kept_dims:
            cat[d] = "bleibt"
        for d in ir.reduced_dims:
            cat[d] = "Σ"
    else:
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


def _estimate_bytes(ir, family: str) -> int:
    """Grober Peak-DRAM-Traffic für den OOM-Schutz (fp16-Inputs 2 B, fp32-Output 4 B)."""
    if family == "elementwise":
        return (ir.arity * 2 + 4) * ir.num_elements
    if family == "reduction":
        return 2 * (ir.kept_size * ir.reduced_size) + 4 * ir.kept_size
    if isinstance(ir, NAryContractionIR):
        # n-är: Summe über die paarweisen Schritte INKL. Zwischentensoren (jeder
        # Schritt: 2 Operanden lesen + Ergebnis schreiben) — deckt die neue OOM-Quelle
        # (Zwischentensoren auf der geteilten 32-GiB-Maschine) ab.
        sz = ir.dim_sizes

        def _p(idx: str) -> int:
            r = 1
            for d in idx:
                r *= sz[d]
            return r
        return sum(2 * (_p(st["a_expr"]) + _p(st["b_expr"])) + 4 * _p(st["c_expr"])
                   for st in ir.steps)
    return 2 * ir.B * (ir.M * ir.K + ir.K * ir.N) + 4 * ir.B * ir.M * ir.N


def validate_dim_sizes(expr: str, dim_sizes: dict, family: str = "contraction") -> Optional[str]:
    """Prüfe die Größen je Index (positive ganze Zahl, alle vorhanden) und eine
    **Speicher-Obergrenze** (OOM-Schutz auf der geteilten Maschine), family-abhängig.

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
    # Struktur + fusionierte Größen (parse validiert erneut streng, family-abhängig).
    try:
        ir = parse(expr, sizes, family=family)
    except (ValueError, NotImplementedError) as e:
        return str(e)
    est = _estimate_bytes(ir, family)
    if est > _MAX_TENSOR_BYTES:
        return (f"Zu groß: ~{est / 2**30:.1f} GiB geschätzt (Grenze "
                f"{_MAX_TENSOR_BYTES // 2**30} GiB) — OOM-Risiko auf der geteilten "
                f"Maschine. Bitte kleinere Größen wählen.")
    return None


def _resolve_op(family: str, op: Optional[str]) -> Optional[str]:
    """Family-abhängige Op für die RunConfig: Reduktion=sum, Elementwise=gewählte
    Op, Kontraktion=None (die Op steckt im GEMM)."""
    if family == "reduction":
        return "sum"
    if family == "elementwise":
        return op
    return None


def epilog_from_controls(v, family: str = "contraction") -> Optional[str]:
    """Epilog-Dropdown-Wert → ``RunConfig.epilog``. Der leere Wert (""/None) und jede
    memory-bound-Familie ergeben ``None`` ⇒ Kernel/Slug byte-identisch zu TZ 1-8.
    Erwartet einen vorher via ``validate_epilog`` geprüften Wert."""
    if family != "contraction" or not v:
        return None
    return str(v)


def validate_epilog(epilog, family: str = "contraction",
                    expr: Optional[str] = None) -> Optional[str]:
    """Prüfe die Epilog-Auswahl (TZ 9): leer ⇒ immer in Ordnung; gesetzt nur bei
    ``family=contraction`` **und** einer 2-Operanden-Kontraktion.

    Die n-är-Sperre ist der wichtige Teil: ``run()`` lehnt Epilog+n-är laut ab —
    dieselbe Regel hier VOR dem GPU-Lauf gibt eine verständliche Meldung statt eines
    Compile-Fehler-Tabs."""
    if not epilog:
        return None
    if epilog not in _EPILOG_KEYS:
        return f"Unbekannter Epilog {epilog!r} (erlaubt: {sorted(_EPILOG_KEYS)} oder keiner)."
    if family != "contraction":
        return (f"Epilog-Fusion gibt es nur bei der Kontraktion — die Familie "
                f"{family!r} rechnet ohnehin memory-bound (der Epilog wäre dort die "
                f"Operation selbst).")
    if expr:
        try:
            ir = parse(expr, {d: 2 for d in expr_indices(expr)}, family=family)
        except Exception:  # noqa: BLE001 — Ausdrucksfehler meldet validate_expr
            return None
        if isinstance(ir, NAryContractionIR):
            return ("Epilog-Fusion ist nur für 2-Operanden-Kontraktionen umgesetzt — "
                    "dieser Ausdruck ist eine n-äre Kette (mehrere paarweise GEMMs). "
                    "Bitte Epilog auf „keiner“ stellen oder einen 2-Operanden-Ausdruck wählen.")
    return None


def config_from_controls(expr: str, dim_sizes: dict, family: str = "contraction",
                         op: Optional[str] = None,
                         epilog: Optional[str] = None) -> RunConfig:
    """Ausdruck + Größen (+ Familie/Op/Epilog) → eine ``RunConfig`` (fp16→fp32-Default;
    von der Einzellauf-Naht weiterbenutzt). Erwartet validierte Eingaben; coerct tolerant
    über ``float`` und normalisiert den Ausdruck auf die explizite Form. ``epilog=None``
    (Default) ⇒ unveränderte Config wie TZ 1-8."""
    idx = expr_indices(expr)
    ds = {d: int(float(dim_sizes[d])) for d in idx}
    return RunConfig(family=family, op=_resolve_op(family, op),
                     epilog=epilog_from_controls(epilog, family),
                     expr=resolve_expr(expr, family), dim_sizes=ds)


def validate_selection(selection, family: str = "contraction") -> Optional[str]:
    """Prüfe die Format-Auswahl (Liste von ``combo_key``-Strings) gegen die für die
    Familie zulässigen Kombis (memory-bound: nur fp16/bf16/fp32)."""
    if not selection:
        return "Bitte mindestens ein Zahlenformat für den Vergleich auswählen."
    valid = {combo_key(d, a) for (d, a) in combos_for_family(family)}
    unknown = [s for s in selection if s not in valid]
    if unknown:
        if family in _MEMORY_BOUND:
            return (f"Für memory-bound ({family}) nicht unterstützte Format-Auswahl: "
                    f"{unknown}. Erlaubt sind fp16/bf16/fp32 (fp8-Arithmetik "
                    f"compiliert nicht, tf32 ist ein reines Tensor-Core-Format).")
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


def validate_bench(warmup, iters) -> Optional[str]:
    """Prüfe Warmup und getaktete Iterationen (ganze Zahlen im zulässigen Bereich)."""
    for name, v, (lo, hi) in (("Warmup", warmup, _BENCH_WARMUP_RANGE),
                              ("Iterationen", iters, _BENCH_ITERS_RANGE)):
        if v is None or v == "":
            return f"{name} fehlt — bitte einen Wert zwischen {lo} und {hi} eingeben."
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return f"{name} ist keine Zahl: {v!r}."
        if fv != int(fv):
            return f"{name} muss ganzzahlig sein (bekommen: {v!r})."
        iv = int(fv)
        if not (lo <= iv <= hi):
            return f"{name}={iv} außerhalb des zulässigen Bereichs [{lo}, {hi}]."
    return None


def bench_from_controls(warmup, iters) -> dict:
    """Warmup/Iterationen (evtl. als Strings aus den Feldern) → dict für ``RunConfig``.
    Erwartet vorher via ``validate_bench`` geprüfte Werte."""
    return {"bench_warmup": int(float(warmup)), "bench_iters": int(float(iters))}


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


def validate_group_m(v) -> Optional[str]:
    """Prüfe die GROUP_M-Auswahl (L2-Swizzle-Gruppengröße) gegen die zulässigen
    Zweierpotenzen. GROUP_M wirkt nur bei aktivem Swizzle, wird aber immer geprüft
    (es wird zur Codegen-Zeit als Literal in den Kernel gebacken)."""
    if v is None or v == "":
        return "GROUP_M fehlt — bitte eine Swizzle-Gruppengröße wählen."
    try:
        iv = int(float(v))
    except (TypeError, ValueError):
        return f"GROUP_M ist keine Zahl: {v!r}."
    if iv not in _SWIZZLE_GROUP_M_OPTIONS:
        return f"GROUP_M={iv} ist kein zulässiger Wert (erlaubt: {list(_SWIZZLE_GROUP_M_OPTIONS)})."
    return None


def tile_from_controls(tm, tn, tk) -> dict:
    """TM/TN/TK (evtl. als Strings aus den Dropdowns) → Tile-dict für ``RunConfig``."""
    return {"TM": int(float(tm)), "TN": int(float(tn)), "TK": int(float(tk))}


def group_m_from_controls(v) -> int:
    """GROUP_M-Dropdown-Wert (evtl. String) → int für ``RunConfig``. Erwartet einen
    via ``validate_group_m`` geprüften Wert."""
    return int(float(v))


# --- TZ 7.5-2: mehrere Swizzle-Konfigurationen + mehrere Tile-Zeilen ----------
def swizzle_configs_from_state(values) -> list:
    """Mehrfachauswahl-Werte (``["off","g8","g16",…]``) → Liste von ``(swizzle, group_m)``.
    ``"off"`` ⇒ ``(False, 8)`` (group_m irrelevant); ``"g<N>"`` ⇒ ``(True, N)``. Leer ⇒
    ``[(False, 8)]`` (mind. ein Lauf). Deterministisch: ohne-Swizzle zuerst, dann
    GROUP_M aufsteigend."""
    out = []
    for v in values or []:
        if v == "off":
            out.append((False, 8))
        elif isinstance(v, str) and v.startswith("g") and v[1:].isdigit():
            out.append((True, int(v[1:])))
    out = sorted(set(out), key=lambda sg: (sg[0], sg[1]))
    return out or [(False, 8)]


def validate_swizzle_configs(values) -> Optional[str]:
    """Prüfe die Swizzle-Konfig-Mehrfachauswahl (Teilmenge aus {off, g1..g32})."""
    if not values:
        return None   # leer ⇒ Default (nur ohne Swizzle)
    unknown = [v for v in values if v not in _SWIZZLE_CONFIG_KEYS]
    if unknown:
        return f"Unbekannte Swizzle-Konfiguration: {unknown} (erlaubt: aus / G1..G32)."
    return None


def default_tile_row() -> dict:
    """Eine neue Tile-Zeile mit den Default-Kachelwerten (RunConfig-Default 128/128/64)."""
    t = RunConfig().tile
    return {"TM": t["TM"], "TN": t["TN"], "TK": t["TK"]}


def tiles_from_state(tm_vals, tn_vals, tk_vals) -> list[dict]:
    """Pattern-Matching-Werte der Tile-Zeilen (drei index-parallele ALL-Listen) →
    Liste von Tile-Roh-dicts. Die drei ALL-States sind index-sortiert (jede Zeile trägt
    genau ein TM/TN/TK) → positionsweises Zippen richtet sie zeilenweise aus."""
    return [{"TM": tm, "TN": tn, "TK": tk}
            for tm, tn, tk in zip(tm_vals or [], tn_vals or [], tk_vals or [])]


def validate_tiles(tiles) -> Optional[str]:
    """Prüfe eine Liste von Tile-Zeilen: jede Zeile gültig (``validate_tile``) UND
    keine Duplikate (zwei identische Tiles = doppelter, verwirrender Vergleich)."""
    if not tiles:
        return "Mindestens eine Tile-Konfiguration nötig."
    seen = set()
    for t in tiles:
        err = validate_tile(t.get("TM"), t.get("TN"), t.get("TK"))
        if err:
            return err
        sig = (int(float(t["TM"])), int(float(t["TN"])), int(float(t["TK"])))
        if sig in seen:
            return (f"Doppelte Tile-Konfiguration TM{sig[0]}/TN{sig[1]}/TK{sig[2]} — "
                    f"bitte je Zeile eine andere Kachelung wählen.")
        seen.add(sig)
    return None


def mutate_tile_rows(rows, triggered_id) -> list[dict]:
    """Reine Zeilen-Mutation für den +/-Callback (headless testbar). ``triggered_id``
    ist ``ID_TILE_ADD`` (Zeile anhängen) oder ``{'type': TILE_RM_TYPE, 'index': i}``
    (Zeile i entfernen; mindestens **eine** Zeile bleibt erhalten)."""
    rows = [dict(r) for r in (rows or [])] or [default_tile_row()]
    if triggered_id == ID_TILE_ADD:
        rows.append(default_tile_row())
    elif isinstance(triggered_id, dict) and triggered_id.get("type") == TILE_RM_TYPE:
        i = triggered_id.get("index")
        if isinstance(i, int) and 0 <= i < len(rows) and len(rows) > 1:
            rows.pop(i)
    return rows


def configs_from_selection(expr, dim_sizes, selection, tiles=None, swizzle_configs=None,
                           baselines=None, bench=None, family="contraction",
                           op=None, epilog=None) -> list[RunConfig]:
    """Ausdruck + Größen + Format-Auswahl → das **Kreuzprodukt** an ``RunConfig``s
    ``selection × tiles × swizzle_configs`` (TZ 7.5-2).

    * ``tiles``: Liste von Tile-dicts ``{"TM","TN","TK"}``; ``None`` ⇒ **ein** Config
      je Format mit dem RunConfig-Default-Tile.
    * ``swizzle_configs``: Liste von ``(swizzle: bool, group_m: int)``; ``None`` ⇒
      ``[(False, 8)]`` (ohne Swizzle, wie bisher). So werden mehrere Tiles UND mehrere
      GROUP_M gegeneinander gemessen.

    Reihenfolge (deterministisch, erstes Element = primäres Format/Tile/Swizzle):
    äußere Schleife über die Formate in kanonischer COMBOS-Reihenfolge, dann Tiles,
    dann Swizzle-Konfigurationen. **Baselines werden nur an der ersten
    (Tile, Swizzle)-Kombi je Format angehängt** (nicht je Tile erneut gemessen).

    memory-bound (Elementwise/Reduktion): **kein Swizzle** (die Templates kennen
    keinen) und **keine GEMM-Baselines**. Die Op wird family-abhängig gesetzt
    (Reduktion=sum, Elementwise=`op`, Kontraktion=None). ``bench`` ist das optionale
    dict ``{"bench_warmup", "bench_iters"}``; ``None`` ⇒ RunConfig-Defaults (10/30).
    Erwartet vorher validierte Eingaben.

    ``epilog`` (TZ 9) gilt **für alle** Configs des Kreuzprodukts (Fusion ist eine
    Eigenschaft der Operation, keine Tuning-Achse) und nur bei der Kontraktion;
    ``None`` ⇒ byte-identische Configs/Slugs wie TZ 1-8.
    """
    norm_expr = resolve_expr(expr, family)
    the_op = _resolve_op(family, op)
    the_epilog = epilog_from_controls(epilog, family)
    idx = expr_indices(expr)
    sizes = {d: int(float(dim_sizes[d])) for d in idx}
    memory_bound = family in _MEMORY_BOUND
    bl = [] if memory_bound else (list(baselines) if baselines else [])
    tile_list = list(tiles) if tiles else [None]
    # memory-bound: Swizzle ist ein Kontraktions-Konzept → genau eine (False,8)-Konfig.
    sw_list = ([(False, 8)] if memory_bound
               else (list(swizzle_configs) if swizzle_configs else [(False, 8)]))
    chosen = set(selection)
    out: list[RunConfig] = []
    for (d, a) in combos_for_family(family):
        if combo_key(d, a) not in chosen:
            continue
        first = True   # Baselines NUR an der ersten (tile,swizzle)-Kombi je Format
        for tile in tile_list:
            for (sw, gm) in sw_list:
                kwargs = dict(family=family, op=the_op, epilog=the_epilog,
                              expr=norm_expr, dim_sizes=dict(sizes),
                              dtype=d, acc_dtype=a, swizzle=bool(sw), group_m=int(gm),
                              baselines=list(bl) if first else [])
                if tile is not None:
                    kwargs["tile"] = {"TM": int(float(tile["TM"])),
                                      "TN": int(float(tile["TN"])),
                                      "TK": int(float(tile["TK"]))}
                if bench is not None:
                    kwargs.update(bench)   # bench_warmup / bench_iters
                out.append(RunConfig(**kwargs))
                first = False
    return out


# ---------------------------------------------------------------------------
# Dash-Komponentenbaum
# ---------------------------------------------------------------------------
def _family_select() -> html.Div:
    """Familien-Dropdown: contraction/elementwise/reduction. Steuert Presets,
    Op-Sichtbarkeit und die zulässigen Formate (Callback in callbacks.py)."""
    return html.Div([
        html.Label("Operations-Familie", className="ctl-label"),
        dbc.Select(id=ID_FAMILY, value="contraction",
                   options=[{"label": lbl, "value": k} for lbl, k in FAMILIES]),
    ])


def _preset_select() -> html.Div:
    """Preset-Dropdown: setzt Ausdruck (+ Op) der aktuellen Familie (Komfort; der
    Freitext bleibt maßgeblich). Der Wert ist ``"<op>|<expr>"`` (s. preset_value)."""
    return html.Div([
        html.Label("Preset", className="ctl-label"),
        dbc.Select(id=ID_PRESET, value=family_default_preset("contraction"),
                   options=preset_options("contraction")),
    ])


def _op_select() -> html.Div:
    """Elementwise-Op-Auswahl (add/mul/copy). Nur bei family=elementwise sichtbar
    (die Sichtbarkeit schaltet ein Callback über ID_OP_WRAP)."""
    return html.Div(
        id=ID_OP_WRAP, style={"display": "none"},
        children=[
            html.Label("Elementwise-Op", className="ctl-label"),
            dbc.RadioItems(id=ID_OP, options=_OP_OPTIONS, value="add", inline=True,
                           style={"fontSize": "13px"}, inputStyle={"marginRight": "5px"},
                           labelStyle={"marginRight": "14px"}),
        ],
    )


def _epilog_select() -> html.Div:
    """Epilog-Fusion-Auswahl (keiner/bias/relu). Nur bei family=contraction sichtbar
    (die Sichtbarkeit schaltet ein Callback über ID_EPILOG_WRAP) — spiegelt bewusst
    die Bauform der Elementwise-Op-Auswahl.

    Der Epilog wird auf dem Akkumulator-Tile VOR dem Store angewandt und spart damit
    den DRAM-Umweg des Zwischentensors; die KPI-Karten zeigen fused vs. sequentiell."""
    return html.Div(
        id=ID_EPILOG_WRAP, style={"display": "block", "marginTop": "10px"},
        children=[
            html.Label("Epilog-Fusion", className="ctl-label"),
            dbc.RadioItems(id=ID_EPILOG, options=_EPILOG_OPTIONS, value="", inline=True,
                           style={"fontSize": "13px"}, inputStyle={"marginRight": "5px"},
                           labelStyle={"marginRight": "14px"}),
        ],
    )


def _expr_input() -> html.Div:
    """Freitext-einsum-Ausdruck (Source of Truth). '->' optional (impliziter Output)."""
    return html.Div([
        html.Label("einsum-Ausdruck", className="ctl-label"),
        dbc.Input(id=ID_EXPR, type="text", value=_DEFAULT_EXPR, debounce=True,
                  placeholder="z. B. bik,bkj->bij",
                  style={"fontFamily": "ui-monospace, monospace"}),
    ])


def index_size_inputs(expr: str, family: str = "contraction",
                      values: Optional[dict] = None) -> list:
    """Ein Größen-Eingabefeld je Index des Ausdrucks (Pattern-Matching-ID), mit
    family-abhängigem Kategorie-Label. ``values`` erhält bereits eingegebene Größen."""
    values = values or {}
    cats = index_categories(expr, family)
    fields = []
    for d in expr_indices(expr):
        label = d + (f"  ({cats[d]})" if d in cats else "")
        fields.append(html.Div(
            style={"flex": "1 1 64px", "minWidth": "64px"},
            children=[
                html.Label(label, className="ctl-label",
                           style={"fontFamily": "ui-monospace, monospace"}),
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
    return [html.H2(["Zahlenformate (Vergleich)", info], className="ctl-section"), tip]


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
            html.Label(label, className="ctl-label"),
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
            html.Div("Mehrere Tile-Zeilen (+ / ✕) werden gegeneinander gemessen — jede "
                     "Zeile eine eigene Kachelung.", style={"marginTop": "5px", "opacity": 0.8}),
            html.Div("L2-Swizzle: ordnet die Block→Kachel-Zuordnung L2-freundlicher um "
                     "(gleiches Ergebnis, oft weniger Speicherverkehr). Als Mehrfachauswahl: "
                     "„aus“ plus je gewünschtem GROUP_M ein Eintrag — der Batch misst jede "
                     "Format×Tile×Swizzle-Kombination gegeneinander.",
                     style={"marginTop": "5px", "opacity": 0.8}),
            html.Div("GROUP_M (G8/G16/…): wie viele M-Kachel-Zeilen der Swizzle zu einer "
                     "L2-Gruppe bündelt (Standard 8, 1 = keine Umordnung). Größere Werte "
                     "teilen mehr B-Spalten im L2-Cache.",
                     style={"marginTop": "5px", "opacity": 0.8}),
        ],
        target=ID_TILE_INFO, placement="right",
        style={"maxWidth": "330px", "textAlign": "left", "fontSize": "12px"},
    )
    return [html.H2(["Kachelung (Tile)", info], className="ctl-section"), tip]


def _tile_row(i: int, tile: dict) -> html.Div:
    """Eine Tile-Zeile: TM/TN/TK-Dropdowns (Pattern-Matching-IDs mit Index i) +
    Entfernen-Button. Wird vom +/-Callback über den Index angesprochen."""
    def _sel(type_: str, options: tuple, val) -> dbc.Select:
        return dbc.Select(
            id={"type": type_, "index": i}, value=str(val), size="sm",
            options=[{"label": str(o), "value": str(o)} for o in options])
    return html.Div(
        style={"display": "flex", "gap": "6px", "alignItems": "center", "marginBottom": "6px"},
        children=[
            html.Div(_sel(TILE_TM_TYPE, _TILE_M_OPTIONS, tile.get("TM", 128)), style={"flex": 1}),
            html.Div(_sel(TILE_TN_TYPE, _TILE_N_OPTIONS, tile.get("TN", 128)), style={"flex": 1}),
            html.Div(_sel(TILE_TK_TYPE, _TILE_K_OPTIONS, tile.get("TK", 64)), style={"flex": 1}),
            dbc.Button("✕", id={"type": TILE_RM_TYPE, "index": i}, color="link", size="sm",
                       n_clicks=0, title="Diese Tile-Zeile entfernen",
                       style={"color": "#b91c1c", "padding": "0 6px", "flex": "0 0 28px"}),
        ],
    )


def tile_rows(rows) -> list:
    """Renderer der dynamischen Tile-Zeilen (rein/headless testbar). Zeilen werden
    fortlaufend 0..n-1 re-indiziert → der Entfernen-Index deckt sich mit der
    Listenposition (``mutate_tile_rows``). Kopfzeile mit TM/TN/TK-Labels voran."""
    rows = rows or [default_tile_row()]
    header = html.Div(
        style={"display": "flex", "gap": "6px", "marginBottom": "2px"},
        children=[html.Label(x, className="ctl-label", style={"flex": 1, "margin": 0})
                  for x in ("TM", "TN", "TK")] + [html.Span(style={"flex": "0 0 28px"})],
    )
    return [header] + [_tile_row(i, dict(t)) for i, t in enumerate(rows)]


def _tile_select() -> html.Div:
    """Mehrere Tile-Zeilen (+/-) + Swizzle-Konfigurationen (Mehrfachauswahl). Der Batch
    misst das Kreuzprodukt Format × Tile × Swizzle-Konfig (``configs_from_selection``)."""
    return html.Div([
        html.Div(id=ID_TILE_ROWS, children=tile_rows([default_tile_row()])),
        dbc.Button("+ Tile-Konfiguration", id=ID_TILE_ADD, color="link", size="sm",
                   n_clicks=0, style={"padding": "2px 0", "fontSize": "12.5px"}),
        html.Div([
            html.Label("L2-Swizzle-Konfigurationen (Mehrfachauswahl)", className="ctl-label"),
            dbc.Checklist(id=ID_SWIZZLE_CONFIGS, options=_SWIZZLE_CONFIG_OPTIONS,
                          value=list(_DEFAULT_SWIZZLE_CONFIG), inline=True,
                          style={"fontSize": "13px"}, inputStyle={"marginRight": "5px"},
                          labelStyle={"marginRight": "12px"}),
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
    return [html.H2(["Baselines (Vergleich)", info], className="ctl-section"), tip]


def _baseline_select() -> html.Div:
    return dbc.Checklist(
        id=ID_BASELINES, options=_BASELINE_OPTIONS, value=[],
        style={"fontSize": "13px"}, inputStyle={"marginRight": "6px"},
    )


def _bench_header() -> list:
    info = html.Span(" ⓘ", id=ID_BENCH_INFO,
                     style={"cursor": "help", "color": "#8b5cf6", "fontWeight": 700,
                            "textTransform": "none"})
    tip = dbc.Tooltip(
        [
            html.Div("Mess-Einstellungen", style={"fontWeight": 600, "marginBottom": "5px"}),
            html.Div("Iterationen: getaktete warme Läufe je Format → daraus die "
                     "Verteilung (Median/min/p90/σ). Mehr Iterationen = stabiler, "
                     "aber längere Messung."),
            html.Div("Warmup: ungetaktete Vorläufe, die Takt/Caches stabilisieren "
                     "(zählen nicht in die Messung).", style={"marginTop": "5px"}),
            html.Div(f"Erlaubt: Warmup {_BENCH_WARMUP_RANGE[0]}–{_BENCH_WARMUP_RANGE[1]}, "
                     f"Iterationen {_BENCH_ITERS_RANGE[0]}–{_BENCH_ITERS_RANGE[1]}. Der "
                     f"Fortschritt zeigt live „Iteration k/N“.",
                     style={"marginTop": "5px", "opacity": 0.8}),
        ],
        target=ID_BENCH_INFO, placement="right",
        style={"maxWidth": "330px", "textAlign": "left", "fontSize": "12px"},
    )
    return [html.H2(["Messung", info], className="ctl-section"), tip]


def _bench_select() -> html.Div:
    c = RunConfig()   # Defaults (10 Warmup / 30 getaktet) als Startwerte
    return html.Div(
        style={"display": "flex", "gap": "8px"},
        children=[
            html.Div(style={"flex": 1}, children=[
                html.Label("Warmup", className="ctl-label"),
                dbc.Input(id=ID_BENCH_WARMUP, type="number", value=c.bench_warmup,
                          min=_BENCH_WARMUP_RANGE[0], max=_BENCH_WARMUP_RANGE[1],
                          step=1, debounce=True),
            ]),
            html.Div(style={"flex": 1}, children=[
                html.Label("Iterationen", className="ctl-label"),
                dbc.Input(id=ID_BENCH_ITERS, type="number", value=c.bench_iters,
                          min=_BENCH_ITERS_RANGE[0], max=_BENCH_ITERS_RANGE[1],
                          step=1, debounce=True),
            ]),
        ],
    )


def build_controls() -> html.Div:
    """Sidebar-Inhalt: Ausdruck (Preset + Freitext + Größen je Index) + Format-
    Auswahl + Kachelung/Swizzle + Baselines + Messung (Warmup/Iterationen) +
    Run/Cancel + Progress."""
    return html.Div([
        html.H2("Operation", className="ctl-section", style={"marginTop": 0}),
        _family_select(),
        _preset_select(),
        _expr_input(),
        _op_select(),
        _epilog_select(),
        # Aufgelöster Output / Klassifikation / Fehler (vom Callback gefüllt).
        html.Div(id=ID_EXPR_INFO, style={"fontSize": "12px", "margin": "6px 0 0",
                                         "minHeight": "16px"}),

        html.H2("Größen je Index", className="ctl-section"),
        html.Div(id=ID_INDEX_SIZES,
                 style={"display": "flex", "flexWrap": "wrap", "gap": "8px"},
                 children=index_size_inputs(_DEFAULT_EXPR)),

        *_dtype_header(),
        _dtype_select(),

        *_tile_header(),
        _tile_select(),

        *_baseline_header(),
        _baseline_select(),

        *_bench_header(),
        _bench_select(),

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
