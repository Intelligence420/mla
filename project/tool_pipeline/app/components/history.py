"""History-Panel (TZ 7.5-4): vergangene Läufe **ansehen · vergleichen · umbenennen ·
löschen**.

Reine, Dash-/GPU-/torch-freie Komponente — importiert nur ``schema``/``store``
(NIE ``run``/torch/cuda; fork-sichere GUI). Die reine Logik (Label/Optionen/Filter)
ist headless testbar; die Callback-Verdrahtung liegt in ``callbacks.py`` (normale,
GPU-freie Callbacks — der einzige nicht-additive Punkt ist ``allow_duplicate=True``
auf ``Output('main','children')`` beim „Laden").
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

# --- Komponenten-IDs (von callbacks.py importiert) ---------------------------
ID_HISTORY_ACCORDION = "history-accordion"
ID_HISTORY_LIST = "history-list"                # Mehrfachauswahl vergangener Läufe
ID_HISTORY_LOAD = "history-load"                # „Ansehen / Vergleichen"
ID_HISTORY_RENAME_INPUT = "history-rename-input"
ID_HISTORY_RENAME_BTN = "history-rename-btn"
ID_HISTORY_DELETE_BTN = "history-delete-btn"    # öffnet die Bestätigung
ID_HISTORY_CONFIRM_DELETE = "history-confirm-delete"  # dcc.ConfirmDialog
ID_HISTORY_FEEDBACK = "history-feedback"        # Statuszeile (umbenannt/gelöscht/…)
ID_HISTORY_REFRESH = "history-refresh"          # Liste neu laden


def history_label(run: dict) -> str:
    """Menschlich lesbares Label eines Laufs für die Auswahl-Checkliste:
    Name · Ausdruck · n_ok/n · Zeitpunkt."""
    ok = f"{run.get('n_ok', 0)}/{run.get('n', 0)} ok"
    ca = (run.get("created_at") or "").replace("T", " ")[:16]
    name = run.get("run_name") or run.get("run_id") or "?"
    parts = [name, run.get("expr", ""), ok]
    if ca:
        parts.append(ca)
    return "  ·  ".join(p for p in parts if p)


def history_options(runs) -> list[dict]:
    """``list_runs``-Ausgabe → Checklist-Optionen (label/value=run_id)."""
    return [{"label": history_label(r), "value": r["run_id"]} for r in (runs or [])]


def runs_for_ids(results, run_ids):
    """Filtere eine ``RunResult``-Liste (aus ``store.read_all``) auf die ausgewählten
    ``run_id``s — in der Auswahl-Reihenfolge der ids (deterministisch, primäres
    Format zuerst wie beim Live-Lauf). Reine Funktion (headless testbar)."""
    chosen = list(run_ids or [])
    by_id: dict = {}
    for r in results or []:
        by_id.setdefault(getattr(r, "run_id", None), []).append(r)
    out = []
    for rid in chosen:
        out.extend(by_id.get(rid, []))
    return out


def _controls_row() -> html.Div:
    return html.Div(
        style={"display": "flex", "flexWrap": "wrap", "gap": "8px", "alignItems": "center",
               "marginTop": "8px"},
        children=[
            dbc.Button("Ansehen / Vergleichen", id=ID_HISTORY_LOAD, color="primary",
                       size="sm", n_clicks=0),
            dbc.Input(id=ID_HISTORY_RENAME_INPUT, placeholder="Neuer Name …", size="sm",
                      style={"maxWidth": "220px"}),
            dbc.Button("Umbenennen", id=ID_HISTORY_RENAME_BTN, color="secondary",
                       outline=True, size="sm", n_clicks=0),
            dbc.Button("Löschen", id=ID_HISTORY_DELETE_BTN, color="danger",
                       outline=True, size="sm", n_clicks=0),
            dbc.Button("⟳ Aktualisieren", id=ID_HISTORY_REFRESH, color="link",
                       size="sm", n_clicks=0, style={"marginLeft": "auto"}),
        ],
    )


def render_history(runs=None) -> dbc.Accordion:
    """Ausklappbares History-Panel (collapsed) — Mehrfachauswahl vergangener Läufe +
    Ansehen/Vergleichen/Umbenennen/Löschen (mit Bestätigung). Die Liste wird beim
    Laden und nach jedem Lauf über einen Callback aus ``store.list_runs`` gefüllt
    (hier initial ``runs`` = i. d. R. leer, entkoppelt das Layout vom Dateizustand)."""
    body = html.Div([
        html.Div("Wähle einen oder mehrere Läufe (Vergleich); „Umbenennen“/„Löschen“ "
                 "wirken auf die Auswahl.", style={"fontSize": "12px", "color": "#6b7280"}),
        dbc.Checklist(id=ID_HISTORY_LIST, options=history_options(runs), value=[],
                      style={"fontSize": "12.5px", "maxHeight": "220px", "overflowY": "auto",
                             "marginTop": "6px"}, inputStyle={"marginRight": "6px"}),
        _controls_row(),
        html.Div(id=ID_HISTORY_FEEDBACK, style={"fontSize": "12px", "color": "#6b7280",
                                                "minHeight": "16px", "marginTop": "6px"}),
        dcc.ConfirmDialog(
            id=ID_HISTORY_CONFIRM_DELETE,
            message=("Ausgewählte Läufe endgültig löschen? Es werden NUR die JSONL-Zeilen "
                     "entfernt — die gecachten Kernel-Dateien (kernels/) bleiben unberührt."),
        ),
    ])
    return dbc.Accordion(
        [dbc.AccordionItem(
            body, title="Vergangene Läufe  —  ansehen · vergleichen · umbenennen · löschen",
            item_id="history")],
        id=ID_HISTORY_ACCORDION, start_collapsed=True, flush=True,
        style={"margin": "0 0 6px"},
    )
