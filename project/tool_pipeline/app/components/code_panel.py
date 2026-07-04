"""Read-only, syntaxhervorgehobenes Panel für den generierten cuTile-Quelltext (TZ 2 / TODO 4).

Reine Funktion ``render_code_panel(source, kernel_path) -> Dash-Komponente``. Die
Quelle liefert der Callback (TODO 6) aus ``RunResult.kernel_source`` (additiv in
TODO 5); ``kernel_path`` dient als Beschriftung. Bewusst als Argument (nicht aus
dem RunResult gelesen) → in TODO 4 ohne Schema-Erweiterung testbar.

Syntax-Highlighting via ``dcc.Markdown`` (bundelt highlight.js). Der generierte
cuTile-Quelltext enthält keine ```-Zäune, daher ist die Markdown-Einbettung sicher.

Naht-Regel: kein run/torch-Import.
"""

from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
from dash import dcc, html

_MUTED = {"color": "#6b7280", "fontSize": "12px"}


def _header(kernel_path: Optional[str]):
    children = [html.Strong("Generiertes cuTile-Kernel")]
    if kernel_path:
        children.append(html.Span(f"  {kernel_path}",
                                   style={**_MUTED, "fontFamily": "ui-monospace, monospace"}))
    return html.Div(children, className="mb-2")


def render_code_panel(source: Optional[str], kernel_path: Optional[str] = None):
    """Karte mit dem generierten Quelltext (oder Hinweis, wenn keiner vorliegt)."""
    if not source:
        return dbc.Card(dbc.CardBody(
            [_header(kernel_path),
             html.Span("Kein generierter Kernel verfügbar.", style=_MUTED)]
        ))
    code = dcc.Markdown(
        f"```python\n{source}\n```",
        style={"maxHeight": "440px", "overflow": "auto", "fontSize": "12.5px", "marginBottom": 0},
    )
    return dbc.Card(dbc.CardBody([_header(kernel_path), code]))
