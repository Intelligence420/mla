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

import uuid
from datetime import datetime
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from filelock import FileLock, Timeout

from .components import charts, code_panel, controls, history, kpis
from ..store import store   # torch-frei (pandas lazy) → im Haupt-Prozess GPU-frei ladbar

# GPU-Lock + Timeout. .cache/ ist gitignored; Parent wird vor dem Lock sichergestellt.
_PROJECT_DIR = Path(__file__).resolve().parents[2]
_GPU_LOCK = _PROJECT_DIR / ".cache" / "gpu.lock"
_LOCK_TIMEOUT = 60  # s — danach freundliche „GPU belegt"-Meldung statt endlos zu warten
# TZ 7.5-2: weiche Warnung (keine harte Sperre) ab so vielen Configs im Batch —
# |Formate|×|Tiles|×|Swizzle-Konfigs| kann auf der geteilten Maschine lange dauern / OOM.
_SOFT_CONFIG_WARN = 12


def _default_run_name(family: str, expr: str, created_at: str) -> str:
    """Default-Name eines Testlaufs (TZ 7.5-4): Familie · Ausdruck · Uhrzeit (HH:MM
    aus dem ISO-created_at). Umbenennbar in der History."""
    hhmm = created_at[11:16] if len(created_at) >= 16 else created_at
    return f"{family} · {expr} · {hhmm}"


def _reload_source(result):
    """History-Läufe: ``kernel_source`` steht bewusst NICHT im JSONL → aus
    ``kernels/<slug>.py`` (``kernel_path``, projekt-relativ) nachladen, damit das
    Code-Panel den Kernel zeigt. Lesefehler ⇒ still ``None`` (render_code_panel zeigt
    dann „kein Kernel"). Mutiert das übergebene RunResult und gibt es zurück."""
    if getattr(result, "kernel_source", None) or not getattr(result, "kernel_path", None):
        return result
    try:
        result.kernel_source = (_PROJECT_DIR / result.kernel_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        pass
    return result

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


def _history_feedback(msg: str, ok: bool = True):
    """Farbcodierte History-Rückmeldung: grün bei Erfolg (umbenannt/gelöscht), amber
    bei einem Hinweis (z. B. „bitte Auswahl treffen"). Ersetzt den bisher einheitlich
    grauen Feedback-Text und macht Erfolg vs. Hinweis auf einen Blick unterscheidbar."""
    color = "#15803d" if ok else "#b45309"   # grün / amber
    return html.Span(msg, style={"color": color, "fontWeight": 500})


def _format_status_strip(results) -> html.Div:
    """Kompakte Statuszeile je gewähltem Format — auch fehlgeschlagene bleiben
    sichtbar (statt still aus den Charts zu verschwinden)."""
    badges = []
    for r in results:
        cfg = r.config or {}
        lbl = f"{cfg.get('dtype')} → {cfg.get('acc_dtype')}"
        if cfg.get("op"):
            lbl += f" · {cfg.get('op')}"
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
    if cfg.get("op"):            # memory-bound: Op mit (add/mul/copy/sum) — disambiguiert
        base += f" · {cfg.get('op')}"
    # TZ 7.5-2: Tile nur zeigen, wenn es vom Default (128/128/64) abweicht → Multi-
    # Config-Tabs disambiguiert, Einzel-Default-Tabs bleiben schlicht 'dtype → acc'.
    t = cfg.get("tile") or {}
    if t and (t.get("TM"), t.get("TN"), t.get("TK")) != (128, 128, 64):
        base += f" · TM{t.get('TM')}/{t.get('TN')}/{t.get('TK')}"
    if cfg.get("swizzle"):
        gm = int(cfg.get("group_m", 8) or 8)
        base += " · sw" + (f" G{gm}" if gm != 8 else "")   # einheitlich '· sw' (Tab/Badge/Legende)
    if cfg.get("epilog"):        # TZ 9: fused-Läufe im Tab/in der Legende erkennbar
        base += f" · ep {cfg.get('epilog')}"
    return base


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


def execute_run(expr, dim_sizes, selection, family="contraction", op=None,
                tm=None, tn=None, tk=None, swizzle=False, group_m=8,
                tiles=None, swizzle_configs=None, baselines=None,
                warmup=None, iters=None, progress=None, epilog=None) -> list:
    """Reine Ablauflogik des Batch-Vergleichs (Dash-frei, headless testbar):
    validieren → RunConfig je Format → EIN GPU-Lock → ``run()`` je Format → rendern.

    ``family`` (contraction/elementwise/reduction) + ``op`` (Elementwise: add/mul/
    copy) wählen die Operations-Familie; alle Validierungen und der Config-Bau sind
    entsprechend family-abhängig.

    Gibt **immer** eine Liste von Main-Komponenten zurück (nie eine Exception):
    ungültiger Ausdruck/Größen/Tile/Auswahl → Warnung (kein GPU-Lauf); GPU belegt →
    freundliche Meldung; sonst der gerenderte Vergleich (inkl. sauber angezeigter
    Fehler-Stati).

    ``expr`` ist der einsum-Ausdruck (Presets/Freitext), ``dim_sizes`` das Roh-dict
    Index→Größe (aus den dynamischen Feldern). ``tm/tn/tk`` (Tile) + ``swizzle``
    steuern die Kachelung; ``tm/tn/tk=None`` ⇒ RunConfig-Default-Tile. ``group_m``
    (L2-Swizzle-Gruppengröße, Default 8) wirkt nur bei aktivem Swizzle. ``baselines``
    ist die (evtl. leere) Liste zuzuschaltender Vergleiche. ``warmup``/``iters`` sind
    die Mess-Einstellungen (``None`` ⇒ RunConfig-Defaults 10/30). ``progress`` ist der
    optionale Dash-``set_progress``-Callback → headless mit ``None`` testbar.

    TZ 7.5-2 (Multi-Config): ``tiles`` (Liste von Tile-dicts) und ``swizzle_configs``
    (Liste von ``(swizzle, group_m)``) sind der **GUI-Pfad** — der Batch misst das
    Kreuzprodukt Format × Tile × Swizzle-Konfig. Sind sie ``None``, gilt der obige
    Skalar-Rückfall (``tm/tn/tk`` + ``swizzle``/``group_m`` — für Tests/CLI). Ab
    ``_SOFT_CONFIG_WARN`` Configs erscheint eine **weiche** Warnung (keine Sperre).

    TZ 9 (Fusion): ``epilog`` (``None``/``"bias"``/``"relu"``) gilt für **alle** Configs
    des Kreuzprodukts und nur bei ``family="contraction"``; ``None`` (Default) ⇒
    unveränderter Pfad wie TZ 1-8.
    """
    def _set(pct: int, text: str) -> None:
        # Füllbreite (Style des inneren Balkens) + Statustext in EINEM set_progress.
        if progress is not None:
            progress((controls.prog_fill_style(pct), text))

    _set(0, "")  # neuer Lauf → Balken sofort leeren (löst den vollen Balken des
                 # Vorlaufs ab; auch bei anschließendem Validierungsfehler ehrlich leer)

    err = controls.validate_expr(expr, family)
    if err:
        return [_alert("Ungültiger Ausdruck", err, "warning")]
    err = controls.validate_dim_sizes(expr, dim_sizes, family)
    if err:
        return [_alert("Ungültige Größe", err, "warning")]
    err = controls.validate_selection(selection, family)
    if err:
        return [_alert("Ungültige Auswahl", err, "warning")]
    err = controls.validate_baselines(baselines)
    if err:
        return [_alert("Ungültige Baseline-Auswahl", err, "warning")]
    # TZ 9: Epilog nur bei der Kontraktion und nur für 2-Operanden-Ausdrücke — VOR
    # dem GPU-Lauf abfangen (run() würde sonst mit einem Compile-Fehler-Tab antworten).
    err = controls.validate_epilog(epilog, family, expr)
    if err:
        return [_alert("Ungültiger Epilog", err, "warning")]
    # --- Tile-Konfiguration(en) (TZ 7.5-2: eine ODER mehrere Zeilen) ------------
    # GUI-Pfad: `tiles` = Liste von Roh-Tile-dicts (aus den dynamischen +/-Zeilen).
    # Rückfall/Skalarpfad (Tests/CLI): einzelnes tm/tn/tk ⇒ eine Zeile; nichts ⇒ Default.
    if tiles is not None:
        err = controls.validate_tiles(tiles)
        if err:
            return [_alert("Ungültige Kachelung", err, "warning")]
        tile_list = [controls.tile_from_controls(t["TM"], t["TN"], t["TK"]) for t in tiles]
    elif tm is not None or tn is not None or tk is not None:
        err = controls.validate_tile(tm, tn, tk)
        if err:
            return [_alert("Ungültige Kachelung", err, "warning")]
        tile_list = [controls.tile_from_controls(tm, tn, tk)]
    else:
        tile_list = [None]   # RunConfig-Default-Tile

    # --- Swizzle-Konfiguration(en) (TZ 7.5-2: Mehrfachauswahl von GROUP_M) -------
    # GUI-Pfad: `swizzle_configs` = Liste (swizzle: bool, group_m: int).
    # Rückfall/Skalarpfad: Swizzle-Modus (off/on/both) × einzelnes group_m.
    if swizzle_configs is not None:
        for (_sw, gm) in swizzle_configs:
            err = controls.validate_group_m(gm)
            if err:
                return [_alert("Ungültige Swizzle-Gruppengröße", err, "warning")]
        sw_list = [(bool(sw), int(gm)) for (sw, gm) in swizzle_configs] or [(False, 8)]
    else:
        err = controls.validate_swizzle(swizzle)
        if err:
            return [_alert("Ungültiger Swizzle-Modus", err, "warning")]
        err = controls.validate_group_m(group_m)
        if err:
            return [_alert("Ungültige Swizzle-Gruppengröße", err, "warning")]
        sw_list = [(s, int(float(group_m))) for s in controls.swizzles_from_value(swizzle)]

    # Mess-Einstellungen (Warmup/Iterationen): nur wenn gesetzt (GUI liefert immer
    # welche); sonst None ⇒ RunConfig-Default (10/30).
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
        configs = controls.configs_from_selection(expr, dim_sizes, selection,
                                                   tiles=tile_list, swizzle_configs=sw_list,
                                                   baselines=baselines, bench=bench,
                                                   family=family, op=op, epilog=epilog)
        # TZ 7.5-4: EINE Batch-Identität je „Vergleichen"-Klick (uuid4 + Default-Name +
        # Zeitstempel), an jedes run() dieses Batches durchgereicht → ein benannter,
        # wieder-ansehbarer Lauf. created_at EINMAL außen (nicht je Zeile via now()).
        batch_id = uuid.uuid4().hex
        created_at = datetime.now().isoformat(timespec="seconds")
        run_name = _default_run_name(family, expr, created_at)
        from tool_pipeline.run import run  # lazy → Haupt-Prozess bleibt CUDA-frei
        _GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
        results = []
        total = len(configs)
        with FileLock(str(_GPU_LOCK)).acquire(timeout=_LOCK_TIMEOUT):
            for i, cfg in enumerate(configs, 1):
                op_txt = f" · {cfg.op}" if cfg.op else ""
                t = cfg.tile
                # Tile + Swizzle-Variante ins Label (TZ 7.5-2: mehrere Configs je Format
                # sind sonst im Fortschritt nicht auseinanderzuhalten).
                tile_txt = f" · TM{t.get('TM')}/{t.get('TN')}/{t.get('TK')}"
                sw_txt = f" · sw G{cfg.group_m}" if cfg.swizzle else ""
                label = (f"Config {i}/{total}: {cfg.dtype}→{cfg.acc_dtype}{op_txt}"
                         f"{tile_txt}{sw_txt}")
                # Vor dem Lauf: welche Config gerade rechnet; der Balken steht auf den
                # bereits fertigen Schritten ((i-1)/total). Bis die Messung startet
                # (Compile/Verify) bleibt er hier stehen.
                _set(int(100 * (i - 1) / total), f"{label} · kompiliere/verifiziere …")

                # Live-Fortschritt der Messung: run() ruft diesen Callback nach jeder
                # getakteten Iteration mit (done, iters). Der Balken wächst innerhalb
                # der Config von (i-1)/total bis i/total; der Text zeigt „Iteration k/N".
                def _bench_progress(done, n_iters, _label=label, _i=i):
                    frac = (_i - 1 + done / n_iters) / total
                    _set(int(100 * frac), f"{_label} · Iteration {done}/{n_iters}")

                results.append(run(cfg, progress=_bench_progress, run_id=batch_id,
                                   run_name=run_name, created_at=created_at))
                # Nach dem Lauf: Schritt erledigt → Balken einen Schritt voller (i/total).
                # Nach der letzten Config steht er auf 100 % und bleibt dort (running=
                # blendet ihn NICHT mehr aus) bis der nächste Lauf ihn oben zurücksetzt.
                _set(int(100 * i / total), f"{label} ✓")
        rendered = render_comparison(results)
        # Weiche Warnung (keine harte Sperre) bei großem Kreuzprodukt — informiert
        # für den nächsten Lauf (der Batch selbst ist bereits gefahren).
        if total > _SOFT_CONFIG_WARN:
            rendered = [_alert(
                "Großer Vergleich",
                f"{total} Konfigurationen (Formate × Tiles × Swizzle) in einem Batch — "
                f"das dauert und belastet die geteilte GPU. Ggf. Achsen reduzieren.",
                "info")] + rendered
        return rendered
    except Timeout:
        return [_alert("GPU belegt",
                       f"Ein anderer Lauf hält die GPU seit über {_LOCK_TIMEOUT}s. "
                       f"Bitte erneut versuchen.", "warning")]
    except Exception as e:  # noqa: BLE001 — Import-/Lock-/Infra-Fehler dürfen die UI nicht crashen
        return [_alert(
            "Interner Fehler",
            "Ein unerwarteter Fehler hat den Vergleich abgebrochen. Bitte die "
            "Eingaben prüfen und erneut versuchen; besteht das Problem fort, hilft "
            f"das technische Detail bei der Diagnose: {type(e).__name__}: {e}",
            "danger")]


def _expr_info(expr, family="contraction"):
    """Info-Zeile unter dem Ausdrucksfeld: aufgelöster (expliziter) Ausdruck +
    Kategorie je Index (family-abhängig: M/N/K/Batch bzw. elem bzw. bleibt/Σ) —
    macht die Klassifikation sichtbar."""
    resolved = controls.resolve_expr(expr, family)
    cats = controls.index_categories(expr, family)
    parts = ", ".join(f"{d}:{c}" for d, c in cats.items())
    children = [html.Span("→ ", style={"color": "#6b7280"}), html.Code(resolved)]
    if parts:
        children.append(html.Span(f"   ·   {parts}", style={"color": "#6b7280"}))
    return html.Span(children)


def register(app) -> None:
    """Callbacks registrieren: Preset→Ausdruck, Ausdruck→Größenfelder, Background-Vergleich."""

    # 0) Familien-Auswahl: aktualisiert Presets, setzt das erste Preset der Familie
    #    (löst dann Preset→Ausdruck+Op aus), blendet die Op-Auswahl nur bei
    #    Elementwise ein und passt die Format-Auswahl an (memory-bound: fp16/bf16/fp32).
    @app.callback(
        Output(controls.ID_PRESET, "options"),
        Output(controls.ID_PRESET, "value"),
        Output(controls.ID_OP_WRAP, "style"),
        Output(controls.ID_EPILOG_WRAP, "style"),
        Output(controls.ID_EPILOG, "value"),
        Output(controls.ID_DTYPES, "options"),
        Output(controls.ID_DTYPES, "value"),
        Input(controls.ID_FAMILY, "value"),
        prevent_initial_call=True,
    )
    def _apply_family(family):
        family = family or "contraction"
        op_style = ({"display": "block", "marginTop": "10px"}
                    if family == "elementwise" else {"display": "none"})
        # TZ 9: Epilog-Fusion ist ein Kontraktions-Konzept → nur dort sichtbar; beim
        # Wechsel auf memory-bound wird die Auswahl zurückgesetzt, damit kein
        # unsichtbarer Epilog im Zustand hängt (er würde ohnehin abgelehnt).
        ep_style = ({"display": "block", "marginTop": "10px"}
                    if family == "contraction" else {"display": "none"})
        ep_value = no_update if family == "contraction" else ""
        return (controls.preset_options(family),
                controls.family_default_preset(family),
                op_style, ep_style, ep_value,
                controls.dtype_options_for_family(family),
                controls.default_selection_for_family(family))

    # 1) Preset-Dropdown füllt Ausdruck **und** (Elementwise-)Op. Der Preset-Wert ist
    #    "<op>|<expr>"; der Freitext/die Op bleiben danach editierbar. Op wird nur
    #    für die Elementwise-Ops (add/mul/copy) gesetzt (sum/None → unverändert).
    @app.callback(
        Output(controls.ID_EXPR, "value"),
        Output(controls.ID_OP, "value"),
        Input(controls.ID_PRESET, "value"),
        prevent_initial_call=True,
    )
    def _apply_preset(preset_value):
        expr, op = controls.parse_preset_value(preset_value)
        op_out = op if op in controls._OP_KEYS else no_update
        return (expr or controls._DEFAULT_EXPR), op_out

    # 2) Ausdruck → dynamische Größenfelder (je Index) + Info/Fehler (family-abhängig).
    #    Bereits eingegebene Größen bleiben erhalten; ungültiger Ausdruck → nur
    #    Fehlertext, keine Felder (der Run-Callback lehnt ihn dann ebenfalls sauber ab).
    @app.callback(
        Output(controls.ID_INDEX_SIZES, "children"),
        Output(controls.ID_EXPR_INFO, "children"),
        Input(controls.ID_EXPR, "value"),
        State(controls.ID_FAMILY, "value"),
        State({"type": controls.INDEX_SIZE_TYPE, "index": ALL}, "id"),
        State({"type": controls.INDEX_SIZE_TYPE, "index": ALL}, "value"),
    )
    def _rebuild_index_sizes(expr, family, cur_ids, cur_vals):
        family = family or "contraction"
        err = controls.validate_expr(expr, family)
        prev = controls.dim_sizes_from_state(cur_ids, cur_vals)  # eingegebene Größen erhalten
        if err:
            return [], html.Span(err, style={"color": "#b91c1c"})
        return controls.index_size_inputs(expr, family, values=prev), _expr_info(expr, family)

    # 2b) Tile-Zeilen +/- (TZ 7.5-2): „+ Tile" hängt eine Zeile an, „✕" entfernt eine.
    #     Reine Zeilen-Mutation (controls.mutate_tile_rows) über den aktuellen Feld-
    #     zustand; der Container hält die Wahrheit (wie ID_INDEX_SIZES). GPU-/torch-frei.
    @app.callback(
        Output(controls.ID_TILE_ROWS, "children"),
        Input(controls.ID_TILE_ADD, "n_clicks"),
        Input({"type": controls.TILE_RM_TYPE, "index": ALL}, "n_clicks"),
        State({"type": controls.TILE_TM_TYPE, "index": ALL}, "value"),
        State({"type": controls.TILE_TN_TYPE, "index": ALL}, "value"),
        State({"type": controls.TILE_TK_TYPE, "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def _mutate_tile_rows(add_clicks, rm_clicks, tm_vals, tn_vals, tk_vals):
        # Nur auf einen ECHTEN Klick reagieren: das An-/Abhängen von Zeilen ändert die
        # ALL-Menge und kann den Callback ohne echten Klick (Wert None/0) erneut
        # auslösen → dann nichts tun (sonst spurious Zeilen).
        val = ctx.triggered[0]["value"] if ctx.triggered else None
        if not val:
            raise PreventUpdate
        rows = controls.tiles_from_state(tm_vals, tn_vals, tk_vals)
        rows = controls.mutate_tile_rows(rows, ctx.triggered_id)
        return controls.tile_rows(rows)

    # 3) Haupt-Callback (Background): Ausdruck + Größen (Pattern-Matching) + Achsen.
    @app.callback(
        Output("main", "children"),
        Input(controls.ID_RUN, "n_clicks"),
        State(controls.ID_FAMILY, "value"),
        State(controls.ID_OP, "value"),
        State(controls.ID_EXPR, "value"),
        State({"type": controls.INDEX_SIZE_TYPE, "index": ALL}, "id"),
        State({"type": controls.INDEX_SIZE_TYPE, "index": ALL}, "value"),
        State(controls.ID_DTYPES, "value"),
        # TZ 7.5-2: dynamische Tile-Zeilen (Pattern-Matching, ALL) + Swizzle-Konfig-
        # Mehrfachauswahl statt der bisherigen drei festen Tile-Dropdowns + Radio.
        State({"type": controls.TILE_TM_TYPE, "index": ALL}, "value"),
        State({"type": controls.TILE_TN_TYPE, "index": ALL}, "value"),
        State({"type": controls.TILE_TK_TYPE, "index": ALL}, "value"),
        State(controls.ID_SWIZZLE_CONFIGS, "value"),
        State(controls.ID_BASELINES, "value"),
        State(controls.ID_BENCH_WARMUP, "value"),
        State(controls.ID_BENCH_ITERS, "value"),
        State(controls.ID_EPILOG, "value"),          # TZ 9: Epilog-Fusion (nur Kontraktion)
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
    def _on_run(set_progress, n_clicks, family, op, expr, size_ids, size_vals, selection,
                tm_vals, tn_vals, tk_vals, swizzle_cfg_vals, baselines, warmup, iters,
                epilog):
        # Rohe Swizzle-Konfig-Auswahl früh laut prüfen (statt ungültige Werte in
        # swizzle_configs_from_state still zu verwerfen) — die Checkliste liefert zwar
        # nur gültige Optionen, aber ein Loud-Fail bleibt konsistent mit dem übrigen
        # verify-before-trust-Ansatz.
        sw_err = controls.validate_swizzle_configs(swizzle_cfg_vals)
        if sw_err:
            return [_alert("Ungültige Swizzle-Konfiguration", sw_err, "warning")]
        dim_sizes = controls.dim_sizes_from_state(size_ids, size_vals)
        tiles = controls.tiles_from_state(tm_vals, tn_vals, tk_vals)
        swizzle_configs = controls.swizzle_configs_from_state(swizzle_cfg_vals)
        return execute_run(expr, dim_sizes, selection, family=family, op=op,
                           tiles=tiles, swizzle_configs=swizzle_configs,
                           baselines=baselines, warmup=warmup, iters=iters,
                           progress=set_progress, epilog=epilog)

    # 4) Nach jedem Lauf im Browser prüfen: ist die Kopfmeldung im Main eine Warnung,
    #    ein Fehler oder ein Hinweis (dbc.Alert vom Typ warning/danger/info — ein
    #    abgebrochener Lauf: ungültige Eingabe, „GPU belegt", interner Fehler … ODER
    #    die weiche „viele Configs"-Warnung), dann Main UND Fenster nach ganz oben
    #    scrollen, damit die Meldung sofort sichtbar ist.
    #    Bei reinem Erfolg ist die Kopfmeldung kein solcher Alert → kein Scroll.
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
                     head.classList.contains('alert-danger') ||
                     head.classList.contains('alert-info'))) {
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

    # ------------------------------------------------------------------
    # 5) History (TZ 7.5-4): vergangene Läufe ansehen/vergleichen/umbenennen/löschen.
    #    Alle NORMAL (nicht background=True) — reiner, GPU-/torch-freier Store-Zugriff.
    # ------------------------------------------------------------------
    # (5a) Liste füllen: beim Laden + nach jedem Lauf (main.children ändert sich) +
    #      auf „Aktualisieren". store.list_runs ist verlustfrei (ohne pandas).
    @app.callback(
        Output(history.ID_HISTORY_LIST, "options"),
        Input(history.ID_HISTORY_REFRESH, "n_clicks"),
        Input("main", "children"),
    )
    def _refresh_history(_n, _main):
        return history.history_options(store.list_runs())

    # (5b) Ansehen/Vergleichen: ausgewählte Läufe aus dem Store rekonstruieren
    #      (read_all → runs_for_ids), Kernel-Quelltext nachladen, render_comparison
    #      WIEDERVERWENDEN. DIE eine nicht-additive Stelle: allow_duplicate auf main.
    @app.callback(
        Output("main", "children", allow_duplicate=True),
        Input(history.ID_HISTORY_LOAD, "n_clicks"),
        State(history.ID_HISTORY_LIST, "value"),
        prevent_initial_call=True,
    )
    def _load_history(_n, run_ids):
        if not run_ids:
            return [_alert("Keine Auswahl", "Bitte mindestens einen Lauf auswählen.", "warning")]
        results = history.runs_for_ids(store.read_all(), run_ids)
        if not results:
            return [_alert("Nichts gefunden",
                           "Die ausgewählten Läufe sind nicht mehr im Store.", "warning")]
        results = [_reload_source(r) for r in results]
        return render_comparison(results)

    # (5c) Löschen: Button öffnet die Bestätigung (nur bei nicht-leerer Auswahl).
    @app.callback(
        Output(history.ID_HISTORY_CONFIRM_DELETE, "displayed"),
        Input(history.ID_HISTORY_DELETE_BTN, "n_clicks"),
        State(history.ID_HISTORY_LIST, "value"),
        prevent_initial_call=True,
    )
    def _ask_delete(_n, run_ids):
        return bool(run_ids)

    # (5d) Löschen bestätigt: store.delete_run je Lauf (atomar; Kernel-Dateien bleiben)
    #      → Rückmeldung + Liste aktualisieren.
    @app.callback(
        Output(history.ID_HISTORY_FEEDBACK, "children"),
        Output(history.ID_HISTORY_LIST, "options", allow_duplicate=True),
        Input(history.ID_HISTORY_CONFIRM_DELETE, "submit_n_clicks"),
        State(history.ID_HISTORY_LIST, "value"),
        prevent_initial_call=True,
    )
    def _do_delete(_submit, run_ids):
        run_ids = run_ids or []
        removed = sum(store.delete_run(rid) for rid in run_ids)
        msg = (f"{removed} Zeile(n) aus {len(run_ids)} Lauf/Läufen gelöscht "
               f"(Kernel-Dateien bleiben unberührt).")
        return _history_feedback(msg, ok=True), history.history_options(store.list_runs())

    # (5e) Umbenennen: store.rename_run auf jede Auswahl → Rückmeldung + Liste neu.
    @app.callback(
        Output(history.ID_HISTORY_FEEDBACK, "children", allow_duplicate=True),
        Output(history.ID_HISTORY_LIST, "options", allow_duplicate=True),
        Input(history.ID_HISTORY_RENAME_BTN, "n_clicks"),
        State(history.ID_HISTORY_RENAME_INPUT, "value"),
        State(history.ID_HISTORY_LIST, "value"),
        prevent_initial_call=True,
    )
    def _rename_history(_n, new_name, run_ids):
        new_name = (new_name or "").strip()
        if not run_ids:
            return _history_feedback("Bitte mindestens einen Lauf auswählen.", ok=False), no_update
        if not new_name:
            return _history_feedback("Bitte einen neuen Namen eingeben.", ok=False), no_update
        n = sum(store.rename_run(rid, new_name) for rid in run_ids)
        return (_history_feedback(f"{len(run_ids)} Lauf/Läufe umbenannt ({n} Zeilen).", ok=True),
                history.history_options(store.list_runs()))
