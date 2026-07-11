"""Live-Vergleichs-Callback (TZ 3) — das Herzstück des Live-Loops.

Klick auf „Vergleichen" → Background-Callback (Worker-Prozess) → je gewähltem
Format eine RunConfig → **ein** prozessübergreifender GPU-Lock über den ganzen
Batch → die **eine Naht** ``run(config)`` je Format → drei Vergleichs-Charts
(Durchsatz + Genauigkeit↔Durchsatz + Roofline) plus KPIs/Verify/Code des primären
Formats in den Main-Bereich.

Zwei Design-Punkte (PLAN §2/§8 + die TZ-2/TZ-3-Entscheidungen):

* **Fork-Sicherheit:** ``run``/torch/cuda werden **lazy im Worker** importiert
  (in ``execute_run``), nie im Modulkopf → der Haupt-Dash-Prozess bleibt CUDA-frei.
  Der DiskcacheManager forkt den Haupt-Prozess; hielte der einen CUDA-Kontext,
  wäre er im Fork kaputt.
* **GPU-Lock:** ``filelock.FileLock`` (fcntl) serialisiert die ``run()``-Aufrufe
  prozessübergreifend. Der ganze Batch läuft unter EINEM Lock (eine Vergleichs-
  Aktion = eine GPU-Session). Bei Prozess-Tod (Cancel) gibt das OS den flock
  **automatisch** frei — kein verwaister Lock.

Fortschritt ist **determinat je Format** (TZ 3): ``set_progress`` meldet je Format
„Format i/N …" und füllt den Balken **pro fertigem Schritt** (nach jedem ``run()``
einen Schritt voller, am Ende 100 %). ``running=`` blendet den Balken zum Lauf ein
und lässt ihn danach **voll stehen** (bis der nächste Lauf ihn zurücksetzt) und
schaltet die Buttons. Die Kernlogik ``execute_run`` ist Dash-frei und wird headless
(mit echtem GPU-Lauf) in ``tests/test_app_execute.py`` geprüft.
"""

from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, dcc, html
from filelock import FileLock, Timeout

from .components import charts, code_panel, controls, kpis

# GPU-Lock + Timeout. .cache/ ist gitignored; Parent wird vor dem Lock sichergestellt.
_PROJECT_DIR = Path(__file__).resolve().parents[2]
_GPU_LOCK = _PROJECT_DIR / ".cache" / "gpu.lock"
_LOCK_TIMEOUT = 60  # s — danach freundliche „GPU belegt"-Meldung statt endlos zu warten

# Progress-Balken: der Track (Hülle) wird via running= eingeblendet und bleibt danach
# sichtbar (voll) stehen — bewusst kein Ausblenden; der nächste Lauf setzt die Füllung
# oben in execute_run auf 0 zurück. Die Füllbreite selbst läuft über progress= (Style
# des inneren Balkens, siehe controls.prog_fill_style).

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
    2. die drei Vergleichs-Charts **untereinander** (je volle Breite → größer:
       Durchsatz · Genauigkeit↔Durchsatz · Roofline),
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
            # Dritter Chart (TZ 5): Roofline — macht die memory- vs compute-bound-
            # Einordnung sichtbar (Punkte nur aus verifizierten Läufen).
            dcc.Graph(figure=charts.figure_roofline(results, primary_key),
                      config=_GRAPH_CONFIG, style={"height": "520px", "width": "100%"}),
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


def execute_run(expr, dim_sizes, selection, tm=None, tn=None, tk=None,
                swizzle=False, baselines=None, warmup=None, iters=None,
                progress=None) -> list:
    """Reine Ablauflogik des Batch-Vergleichs (Dash-frei, headless testbar):
    validieren → RunConfig je Format → EIN GPU-Lock → ``run()`` je Format → rendern.

    Gibt **immer** eine Liste von Main-Komponenten zurück (nie eine Exception):
    ungültiger Ausdruck/Größen/Tile/Auswahl → Warnung (kein GPU-Lauf); GPU belegt →
    freundliche Meldung; sonst der gerenderte Vergleich (inkl. sauber angezeigter
    Fehler-Stati).

    ``expr`` ist der einsum-Ausdruck (Presets/Freitext), ``dim_sizes`` das Roh-dict
    Index→Größe (aus den dynamischen Feldern). ``tm/tn/tk`` (Tile) + ``swizzle``
    steuern die Kachelung; ``tm/tn/tk=None`` ⇒ RunConfig-Default-Tile. ``baselines``
    ist die (evtl. leere) Liste zuzuschaltender Vergleiche. ``warmup``/``iters`` sind
    die Mess-Einstellungen (``None`` ⇒ RunConfig-Defaults 10/30). ``progress`` ist der
    optionale Dash-``set_progress``-Callback → headless mit ``None`` testbar.
    """
    def _set(pct: int, text: str) -> None:
        # Füllbreite (Style des inneren Balkens) + Statustext in EINEM set_progress.
        if progress is not None:
            progress((controls.prog_fill_style(pct), text))

    _set(0, "")  # neuer Lauf → Balken sofort leeren (löst den vollen Balken des
                 # Vorlaufs ab; auch bei anschließendem Validierungsfehler ehrlich leer)

    err = controls.validate_expr(expr)
    if err:
        return [_alert("Ungültiger Ausdruck", err, "warning")]
    err = controls.validate_dim_sizes(expr, dim_sizes)
    if err:
        return [_alert("Ungültige Größe", err, "warning")]
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

    # Mess-Einstellungen (Warmup/Iterationen): analog zum Tile nur wenn gesetzt (GUI
    # liefert immer welche); sonst None ⇒ RunConfig-Default (10/30).
    bench = None
    if warmup is not None or iters is not None:
        err = controls.validate_bench(warmup, iters)
        if err:
            return [_alert("Ungültige Mess-Einstellungen", err, "warning")]
        bench = controls.bench_from_controls(warmup, iters)

    # ALLES ab hier steht IM try — inkl. Config-Bau, Lazy-Import (kann ImportError
    # werfen), mkdir/Lock, die run()-Schleife und das Rendern —, damit execute_run
    # die Zusage „gibt immer eine Liste zurück, nie eine Exception" hält (Naht-
    # Vertrag; Fund A des Error-Audits). Der Balken füllt sich in der Schleife je
    # fertigem Format; Fehlerpfade lassen ihn stehen (statt ihn irreführend auf
    # 100 % zu setzen) — der Alert erklärt den Grund.
    try:
        configs = controls.configs_from_selection(expr, dim_sizes, selection, tile=tile,
                                                   swizzle=swizzle, baselines=baselines,
                                                   bench=bench)
        from tool_pipeline.run import run  # lazy → Haupt-Prozess bleibt CUDA-frei
        _GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
        results = []
        total = len(configs)
        with FileLock(str(_GPU_LOCK)).acquire(timeout=_LOCK_TIMEOUT):
            for i, cfg in enumerate(configs, 1):
                label = f"Format {i}/{total}: {cfg.dtype} → {cfg.acc_dtype}"
                # Vor dem Lauf: welches Format gerade rechnet; der Balken steht auf den
                # bereits fertigen Schritten ((i-1)/total). Bis die Messung startet
                # (Compile/Verify) bleibt er hier stehen.
                _set(int(100 * (i - 1) / total), f"{label} · kompiliere/verifiziere …")

                # Live-Fortschritt der Messung: run() ruft diesen Callback nach jeder
                # getakteten Iteration mit (done, iters). Der Balken wächst innerhalb
                # des Formats von (i-1)/total bis i/total; der Text zeigt „Iteration k/N".
                def _bench_progress(done, n_iters, _label=label, _i=i):
                    frac = (_i - 1 + done / n_iters) / total
                    _set(int(100 * frac), f"{_label} · Iteration {done}/{n_iters}")

                results.append(run(cfg, progress=_bench_progress))
                # Nach dem Lauf: Schritt erledigt → Balken einen Schritt voller (i/total).
                # Nach dem letzten Format steht er auf 100 % und bleibt dort (running=
                # blendet ihn NICHT mehr aus) bis der nächste Lauf ihn oben zurücksetzt.
                _set(int(100 * i / total), f"{label} ✓")
        return render_comparison(results)
    except Timeout:
        return [_alert("GPU belegt",
                       f"Ein anderer Lauf hält die GPU seit über {_LOCK_TIMEOUT}s. "
                       f"Bitte erneut versuchen.", "warning")]
    except Exception as e:  # noqa: BLE001 — Import-/Lock-/Infra-Fehler dürfen die UI nicht crashen
        return [_alert("Interner Fehler", f"{type(e).__name__}: {e}", "danger")]


def _expr_info(expr):
    """Info-Zeile unter dem Ausdrucksfeld: aufgelöster (expliziter) Ausdruck +
    Kategorie je Index (M/N/K/Batch) — macht die Klassifikation sichtbar."""
    resolved = controls.resolve_expr(expr)
    cats = controls.index_categories(expr)
    parts = ", ".join(f"{d}:{c}" for d, c in cats.items())
    children = [html.Span("→ ", style={"color": "#6b7280"}), html.Code(resolved)]
    if parts:
        children.append(html.Span(f"   ·   {parts}", style={"color": "#6b7280"}))
    return html.Span(children)


def register(app) -> None:
    """Callbacks registrieren: Preset→Ausdruck, Ausdruck→Größenfelder, Background-Vergleich."""

    # 1) Preset-Dropdown füllt den Ausdruck (der Freitext bleibt danach editierbar).
    @app.callback(
        Output(controls.ID_EXPR, "value"),
        Input(controls.ID_PRESET, "value"),
        prevent_initial_call=True,
    )
    def _apply_preset(preset_expr):
        return preset_expr or controls._DEFAULT_EXPR

    # 2) Ausdruck → dynamische Größenfelder (je Index) + Info/Fehler. Bereits
    #    eingegebene Größen bleiben erhalten; ungültiger Ausdruck → nur Fehlertext,
    #    keine Felder (der Run-Callback lehnt ihn dann ebenfalls sauber ab).
    @app.callback(
        Output(controls.ID_INDEX_SIZES, "children"),
        Output(controls.ID_EXPR_INFO, "children"),
        Input(controls.ID_EXPR, "value"),
        State({"type": controls.INDEX_SIZE_TYPE, "index": ALL}, "id"),
        State({"type": controls.INDEX_SIZE_TYPE, "index": ALL}, "value"),
    )
    def _rebuild_index_sizes(expr, cur_ids, cur_vals):
        err = controls.validate_expr(expr)
        prev = controls.dim_sizes_from_state(cur_ids, cur_vals)  # eingegebene Größen erhalten
        if err:
            return [], html.Span(err, style={"color": "#b91c1c"})
        return controls.index_size_inputs(expr, values=prev), _expr_info(expr)

    # 3) Haupt-Callback (Background): Ausdruck + Größen (Pattern-Matching) + Achsen.
    @app.callback(
        Output("main", "children"),
        Input(controls.ID_RUN, "n_clicks"),
        State(controls.ID_EXPR, "value"),
        State({"type": controls.INDEX_SIZE_TYPE, "index": ALL}, "id"),
        State({"type": controls.INDEX_SIZE_TYPE, "index": ALL}, "value"),
        State(controls.ID_DTYPES, "value"),
        State(controls.ID_TILE_TM, "value"),
        State(controls.ID_TILE_TN, "value"),
        State(controls.ID_TILE_TK, "value"),
        State(controls.ID_SWIZZLE, "value"),
        State(controls.ID_BASELINES, "value"),
        State(controls.ID_BENCH_WARMUP, "value"),
        State(controls.ID_BENCH_ITERS, "value"),
        background=True,
        running=[
            (Output(controls.ID_RUN, "disabled"), True, False),
            (Output(controls.ID_CANCEL, "disabled"), False, True),
            # Track (Hülle) zum Lauf einblenden und danach sichtbar STEHEN lassen
            # (voll bis zum nächsten Lauf) — bewusst kein Ausblenden. Der Füllbalken
            # (ID_PROGRESS) steht NICHT in running=, sondern nur in progress= → seine
            # Live-Updates werden nicht mehr verschluckt.
            (Output(controls.ID_PROGRESS_WRAP, "style"),
             controls.PROG_TRACK_SHOW, controls.PROG_TRACK_SHOW),
        ],
        progress=[Output(controls.ID_PROGRESS, "style"), Output(controls.ID_STATUS, "children")],
        cancel=[Input(controls.ID_CANCEL, "n_clicks")],
        prevent_initial_call=True,
    )
    def _on_run(set_progress, n_clicks, expr, size_ids, size_vals, selection,
                tm, tn, tk, swizzle, baselines, warmup, iters):
        dim_sizes = controls.dim_sizes_from_state(size_ids, size_vals)
        return execute_run(expr, dim_sizes, selection, tm=tm, tn=tn, tk=tk,
                           swizzle=swizzle, baselines=baselines, warmup=warmup,
                           iters=iters, progress=set_progress)

    # 4) Nach jedem Lauf im Browser prüfen: ist die Kopfmeldung im Main eine Warnung
    #    oder ein Fehler (dbc.Alert vom Typ warning/danger — also ein abgebrochener
    #    Lauf: ungültige Eingabe, „GPU belegt", interner Fehler …), dann Main UND
    #    Fenster nach ganz oben scrollen, damit die Meldung sofort sichtbar ist.
    #    Bei Erfolg ist die Kopfmeldung ein 'alert-success' → kein Scroll.
    #    Clientside, weil Scrollen nur im Browser (nicht serverseitig) geht; der
    #    ``_scroll_dummy``-Store ist nur ein Pflicht-Output (bleibt via no_update leer).
    app.clientside_callback(
        """
        function(children) {
            setTimeout(function() {
                var main = document.getElementById('main');
                if (!main) return;
                var head = main.firstElementChild;
                if (head && head.classList &&
                    (head.classList.contains('alert-warning') ||
                     head.classList.contains('alert-danger'))) {
                    main.scrollTo({top: 0, behavior: 'smooth'});
                    window.scrollTo({top: 0, behavior: 'smooth'});
                }
            }, 0);
            return window.dash_clientside.no_update;
        }
        """,
        Output("_scroll_dummy", "data"),
        Input("main", "children"),
        prevent_initial_call=True,
    )
