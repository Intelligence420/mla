"""Live-Lauf-Callback (TZ 2 / TODO 6) — das Herzstück des Live-Loops.

Klick auf „Run" → Background-Callback (Worker-Prozess) → RunConfig aus M/N/K →
prozessübergreifender GPU-Lock → die **eine Naht** ``run(config)`` → Ergebnis
(Status, KPIs, Verify, generierter Code) in den Main-Bereich.

Zwei Design-Punkte (siehe PLAN §2/§8 + die TZ-2-Entscheidungen):

* **Fork-Sicherheit:** ``run``/torch/cuda werden **lazy im Worker** importiert
  (in ``execute_run``), nie im Modulkopf → der Haupt-Dash-Prozess bleibt CUDA-frei.
  Der DiskcacheManager forkt den Haupt-Prozess; hielte der einen CUDA-Kontext,
  wäre er im Fork kaputt.
* **GPU-Lock:** ``filelock.FileLock`` (fcntl) serialisiert die ``run()``-Aufrufe
  prozessübergreifend. Bei Prozess-Tod (Cancel terminiert den Worker) gibt das OS
  den flock **automatisch** frei — kein verwaister Lock. Doppelklick in derselben
  Session verhindert zusätzlich ``running=`` (Button-Disable).

Progress ist bewusst **indeterminat** (freigegebene Entscheidung): ``running=``
zeigt einen animierten Balken + Statustext, solange der Lauf läuft. ``run()`` ist
ein einziger, opaker Aufruf ohne echte Sub-Schritte → kein erfundener ``set_progress``.

Die Kernlogik ``execute_run`` ist Dash-frei und wird headless (mit echtem GPU-Lauf)
in ``tests/test_app_execute.py`` geprüft; ``register`` verdrahtet nur den Callback.
"""

from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, html
from filelock import FileLock, Timeout

from .components import code_panel, controls, kpis

# GPU-Lock + Timeout. .cache/ ist gitignored; Parent wird vor dem Lock sichergestellt.
_PROJECT_DIR = Path(__file__).resolve().parents[2]
_GPU_LOCK = _PROJECT_DIR / ".cache" / "gpu.lock"
_LOCK_TIMEOUT = 60  # s — danach freundliche „GPU belegt"-Meldung statt endlos zu warten

# Progress-Balken sichtbar/verborgen (via running=)
_PROG_SHOW = {"display": "block", "marginTop": "12px", "height": "8px"}
_PROG_HIDE = {"display": "none", "marginTop": "12px", "height": "8px"}
# Ehrlich: der Background-Job kann rechnen ODER (bei belegtem GPU-Lock) darauf warten
# — beides während running=True aktiv ist. Daher nicht bloß „läuft" (Fund F des Audits).
_RUNNING_TEXT = "GPU-Lauf aktiv… (rechnet oder wartet · Abbrechen möglich)"


def _alert(title: str, body: str, color: str):
    return dbc.Alert([html.Strong(title), html.Br(), html.Span(body)], color=color, className="mb-3")


def render_result(result) -> list:
    """RunResult → Liste der Main-Komponenten (Status · Kontext · KPIs · Verify · Code)."""
    parts = [
        kpis.render_status(result),
        kpis.render_context(result),
        kpis.render_kpis(result),
        kpis.render_verify(result),
        code_panel.render_code_panel(result.kernel_source, result.kernel_path),
    ]
    return [p for p in parts if p is not None]


def execute_run(m, n, k) -> list:
    """Reine Ablauflogik eines Laufs (Dash-frei, headless testbar):
    validieren → RunConfig → GPU-Lock → ``run()`` → rendern.

    Gibt **immer** eine Liste von Main-Komponenten zurück (nie eine Exception):
    ungültige Eingabe → Warnung (kein GPU-Lauf); GPU belegt → freundliche Meldung;
    sonst das gerenderte RunResult (inkl. sauber angezeigter Fehler-Stati).
    """
    err = controls.validate_sizes(m, n, k)
    if err:
        return [_alert("Ungültige Eingabe", err, "warning")]

    # ALLES ab hier steht IM try — inkl. Config-Bau, Lazy-Import (kann ImportError
    # werfen, falls torch/cuda.tile im Worker fehlen/kaputt sind), mkdir/Lock und
    # das Rendern —, damit execute_run die Zusage „gibt immer eine Liste zurück,
    # nie eine Exception" wirklich hält (Naht-Vertrag; Fund A des Error-Audits).
    try:
        cfg = controls.config_from_controls(m, n, k)
        from tool_pipeline.run import run  # lazy → Haupt-Prozess bleibt CUDA-frei
        _GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(_GPU_LOCK)).acquire(timeout=_LOCK_TIMEOUT):
            result = run(cfg)
        return render_result(result)
    except Timeout:
        return [_alert("GPU belegt",
                       f"Ein anderer Lauf hält die GPU seit über {_LOCK_TIMEOUT}s. "
                       f"Bitte erneut versuchen.", "warning")]
    except Exception as e:  # noqa: BLE001 — Import-/Lock-/Infra-Fehler dürfen die UI nicht crashen
        return [_alert("Interner Fehler", f"{type(e).__name__}: {e}", "danger")]


def register(app) -> None:
    """Den Background-Lauf-Callback an der App registrieren (aus ``create_app``)."""

    @app.callback(
        Output("main", "children"),
        Input(controls.ID_RUN, "n_clicks"),
        State(controls.ID_M, "value"),
        State(controls.ID_N, "value"),
        State(controls.ID_K, "value"),
        background=True,
        running=[
            (Output(controls.ID_RUN, "disabled"), True, False),
            (Output(controls.ID_CANCEL, "disabled"), False, True),
            (Output(controls.ID_PROGRESS, "style"), _PROG_SHOW, _PROG_HIDE),
            (Output(controls.ID_STATUS, "children"), _RUNNING_TEXT, ""),
        ],
        cancel=[Input(controls.ID_CANCEL, "n_clicks")],
        prevent_initial_call=True,
    )
    def _on_run(n_clicks, m, n, k):
        return execute_run(m, n, k)
