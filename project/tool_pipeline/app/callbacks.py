"""Live-Vergleichs-Callback (TZ 3) — das Herzstück des Live-Loops.

Klick auf „Vergleichen" → Background-Callback (Worker-Prozess) → je gewähltem
Format eine RunConfig → **ein** prozessübergreifender GPU-Lock über den ganzen
Batch → die **eine Naht** ``run(config)`` je Format → zwei Vergleichs-Charts
(Durchsatz + Genauigkeit↔Durchsatz) plus KPIs/Verify/Code des primären Formats
in den Main-Bereich.

Zwei Design-Punkte (PLAN §2/§8 + die TZ-2/TZ-3-Entscheidungen):

* **Fork-Sicherheit:** ``run``/torch/cuda werden **lazy im Worker** importiert
  (in ``execute_run``), nie im Modulkopf → der Haupt-Dash-Prozess bleibt CUDA-frei.
  Der DiskcacheManager forkt den Haupt-Prozess; hielte der einen CUDA-Kontext,
  wäre er im Fork kaputt.
* **GPU-Lock:** ``filelock.FileLock`` (fcntl) serialisiert die ``run()``-Aufrufe
  prozessübergreifend. Der ganze Batch läuft unter EINEM Lock (eine Vergleichs-
  Aktion = eine GPU-Session). Bei Prozess-Tod (Cancel) gibt das OS den flock
  **automatisch** frei — kein verwaister Lock.

Fortschritt ist **determinat je Format** (TZ 3): ``set_progress`` meldet
„Format i/N …" (echte Sub-Schritte); ``running=`` blendet den Balken ein/aus und
schaltet die Buttons. Die Kernlogik ``execute_run`` ist Dash-frei und wird headless
(mit echtem GPU-Lauf) in ``tests/test_app_execute.py`` geprüft.
"""

from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from filelock import FileLock, Timeout

from .components import charts, code_panel, controls, kpis

# GPU-Lock + Timeout. .cache/ ist gitignored; Parent wird vor dem Lock sichergestellt.
_PROJECT_DIR = Path(__file__).resolve().parents[2]
_GPU_LOCK = _PROJECT_DIR / ".cache" / "gpu.lock"
_LOCK_TIMEOUT = 60  # s — danach freundliche „GPU belegt"-Meldung statt endlos zu warten

# Progress-Balken sichtbar/verborgen (via running=)
_PROG_SHOW = {"display": "block", "marginTop": "12px", "height": "8px"}
_PROG_HIDE = {"display": "none", "marginTop": "12px", "height": "8px"}

# Plotly-Toolbar der Charts: PNG-Export (Kamera-Knopf) an, unnötige Werkzeuge weg.
_GRAPH_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "filename": "cutile-vergleich", "scale": 2},
}


def _alert(title: str, body: str, color: str):
    return dbc.Alert([html.Strong(title), html.Br(), html.Span(body)], color=color, className="mb-3")


def _format_status_strip(results) -> html.Div:
    """Kompakte Statuszeile je gewähltem Format — auch fehlgeschlagene bleiben
    sichtbar (statt still aus den Charts zu verschwinden)."""
    badges = []
    for r in results:
        cfg = r.config or {}
        lbl = f"{cfg.get('dtype')} → {cfg.get('acc_dtype')}"
        if cfg.get("swizzle"):
            lbl += " · sw"
        ok = r.status == "ok"
        badges.append(dbc.Badge(f"{lbl}: {'PASS' if ok else r.status}",
                                color="success" if ok else "danger",
                                className="me-2 mb-1"))
    return html.Div(badges, className="mb-3")


# Abschnitts-Überschrift (wie in den Controls)
_SECTION = {"fontSize": "11px", "letterSpacing": "0.08em", "textTransform": "uppercase",
            "color": "#6b7280", "margin": "10px 0 4px"}


def _format_label(result) -> str:
    cfg = result.config or {}
    base = f"{cfg.get('dtype')} → {cfg.get('acc_dtype')}"
    return base + " · sw" if cfg.get("swizzle") else base   # einheitlich '· sw' (Tab/Badge/Legende)


def _tab_content(result) -> html.Div:
    """Detail EINES Formats: KPIs (Durchsatz/Median/Compile) · Verify · Kontext ·
    generierter Kernel. Bei Fehl-Status zusätzlich der Status (Grund)."""
    parts = []
    if result.status != "ok":
        parts.append(kpis.render_status(result))   # Fehlergrund im Tab sichtbar machen
    parts += [
        kpis.render_context(result),
        kpis.render_kpis(result),
        kpis.render_verify(result),
        code_panel.render_code_panel(result.kernel_source, result.kernel_path),
    ]
    return html.Div([p for p in parts if p is not None], className="pt-3")


def render_comparison(results) -> list:
    """Batch-Ergebnisse → Main (von oben nach unten):

    1. Headline-Status des primären Formats,
    2. beide Vergleichs-Charts **untereinander** (je volle Breite → größer),
    3. Status-Badges je Format,
    4. **Tabs je Format** — Durchsatz/Median/Verify/Kernel jedes Formats einzeln
       anschaubar und durchklickbar.
    """
    if not results:
        return [_alert("Kein Ergebnis", "Keine Formate ausgewählt.", "warning")]
    primary = results[0]
    pcfg = primary.config or {}
    primary_key = f"{pcfg.get('dtype')}:{pcfg.get('acc_dtype')}"
    n_ok = sum(1 for r in results if r.status == "ok")

    charts_stacked = html.Div(
        [
            dcc.Graph(figure=charts.figure_throughput(results, primary_key),
                      config=_GRAPH_CONFIG, style={"height": "420px", "width": "100%"}),
            dcc.Graph(figure=charts.figure_accuracy_throughput(results, primary_key),
                      config=_GRAPH_CONFIG, style={"height": "440px", "width": "100%"}),
        ],
        className="mb-2",
    )
    summary = html.Div(f"{n_ok}/{len(results)} Formate verifiziert",
                       style={"fontSize": "12px", "color": "#6b7280", "margin": "2px 0 8px"})

    tabs = dbc.Tabs(
        [dbc.Tab(_tab_content(r), label=_format_label(r), tab_id=f"fmt-{i}")
         for i, r in enumerate(results)],
        active_tab="fmt-0",
    )

    parts = [
        kpis.render_status(primary),                 # 1) Headline ganz oben
        summary,
        charts_stacked,                              # 2) Charts untereinander
        _format_status_strip(results),               # 3) Status je Format
        html.Hr(),
        html.Div("Detail je Format", style=_SECTION),
        tabs,                                        # 4) je Format einzeln, durchklickbar
    ]
    return [p for p in parts if p is not None]


def execute_run(m, n, k, selection, tm=None, tn=None, tk=None,
                swizzle=False, baselines=None, progress=None) -> list:
    """Reine Ablauflogik des Batch-Vergleichs (Dash-frei, headless testbar):
    validieren → RunConfig je Format → EIN GPU-Lock → ``run()`` je Format → rendern.

    Gibt **immer** eine Liste von Main-Komponenten zurück (nie eine Exception):
    ungültige Größen/Tile/Auswahl → Warnung (kein GPU-Lauf); GPU belegt →
    freundliche Meldung; sonst der gerenderte Vergleich (inkl. sauber angezeigter
    Fehler-Stati).

    ``tm/tn/tk`` (Tile, gelten für die ganze Auswahl) und ``swizzle`` steuern die
    Kachelung; ``tm/tn/tk=None`` ⇒ RunConfig-Default-Tile. ``baselines`` ist die
    (evtl. leere) Liste zuzuschaltender Vergleiche. ``progress`` ist der optionale
    Dash-``set_progress``-Callback → headless mit ``None`` testbar.
    """
    def _set(pct: int, text: str) -> None:
        if progress is not None:
            progress((pct, text))

    err = controls.validate_sizes(m, n, k)
    if err:
        return [_alert("Ungültige Eingabe", err, "warning")]
    err = controls.validate_selection(selection)
    if err:
        return [_alert("Ungültige Auswahl", err, "warning")]
    err = controls.validate_baselines(baselines)
    if err:
        return [_alert("Ungültige Baseline-Auswahl", err, "warning")]
    err = controls.validate_swizzle(swizzle)
    if err:
        return [_alert("Ungültiger Swizzle-Modus", err, "warning")]

    # Tile: nur wenn ein Wert gesetzt ist (GUI liefert immer welche); sonst None →
    # RunConfig-Default. Bei gesetztem Tile hart validieren (sauberer Fehler statt
    # still nicht-baubarem Kernel).
    tile = None
    if tm is not None or tn is not None or tk is not None:
        err = controls.validate_tile(tm, tn, tk)
        if err:
            return [_alert("Ungültige Kachelung", err, "warning")]
        tile = controls.tile_from_controls(tm, tn, tk)

    # ALLES ab hier steht IM try — inkl. Config-Bau, Lazy-Import (kann ImportError
    # werfen), mkdir/Lock, die run()-Schleife und das Rendern —, damit execute_run
    # die Zusage „gibt immer eine Liste zurück, nie eine Exception" hält (Naht-
    # Vertrag; Fund A des Error-Audits). ``finally`` setzt Balken/Text zurück.
    try:
        configs = controls.configs_from_selection(m, n, k, selection, tile=tile,
                                                   swizzle=swizzle, baselines=baselines)
        from tool_pipeline.run import run  # lazy → Haupt-Prozess bleibt CUDA-frei
        _GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
        results = []
        total = len(configs)
        with FileLock(str(_GPU_LOCK)).acquire(timeout=_LOCK_TIMEOUT):
            for i, cfg in enumerate(configs, 1):
                _set(int(100 * (i - 1) / total),
                     f"Format {i}/{total}: {cfg.dtype} → {cfg.acc_dtype} …")
                results.append(run(cfg))
        return render_comparison(results)
    except Timeout:
        return [_alert("GPU belegt",
                       f"Ein anderer Lauf hält die GPU seit über {_LOCK_TIMEOUT}s. "
                       f"Bitte erneut versuchen.", "warning")]
    except Exception as e:  # noqa: BLE001 — Import-/Lock-/Infra-Fehler dürfen die UI nicht crashen
        return [_alert("Interner Fehler", f"{type(e).__name__}: {e}", "danger")]
    finally:
        _set(100, "")  # Balken/Statustext zurücksetzen, egal wie der Batch endete


def register(app) -> None:
    """Den Background-Vergleichs-Callback an der App registrieren (aus ``create_app``)."""

    @app.callback(
        Output("main", "children"),
        Input(controls.ID_RUN, "n_clicks"),
        State(controls.ID_M, "value"),
        State(controls.ID_N, "value"),
        State(controls.ID_K, "value"),
        State(controls.ID_DTYPES, "value"),
        State(controls.ID_TILE_TM, "value"),
        State(controls.ID_TILE_TN, "value"),
        State(controls.ID_TILE_TK, "value"),
        State(controls.ID_SWIZZLE, "value"),
        State(controls.ID_BASELINES, "value"),
        background=True,
        running=[
            (Output(controls.ID_RUN, "disabled"), True, False),
            (Output(controls.ID_CANCEL, "disabled"), False, True),
            (Output(controls.ID_PROGRESS, "style"), _PROG_SHOW, _PROG_HIDE),
        ],
        progress=[Output(controls.ID_PROGRESS, "value"), Output(controls.ID_STATUS, "children")],
        cancel=[Input(controls.ID_CANCEL, "n_clicks")],
        prevent_initial_call=True,
    )
    def _on_run(set_progress, n_clicks, m, n, k, selection, tm, tn, tk, swizzle, baselines):
        return execute_run(m, n, k, selection, tm=tm, tn=tn, tk=tk,
                           swizzle=swizzle, baselines=baselines, progress=set_progress)
