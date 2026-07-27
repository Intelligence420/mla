"""Read-only, syntaxhervorgehobenes Panel für den generierten cuTile-Quelltext.

Reine Funktion ``render_code_panel(source, kernel_path) -> Dash-Komponente``. Die
Quelle liefert der Callback aus ``RunResult.kernel_source``; ``kernel_path`` dient
als Beschriftung. Beides wird bewusst als Argument übergeben und nicht aus dem
RunResult gelesen — so ist das Panel ohne Schema-Kenntnis testbar.

Syntax-Highlighting via ``dcc.Markdown`` (bundelt highlight.js). Der generierte
cuTile-Quelltext enthält keine ```-Zäune, daher ist die Markdown-Einbettung sicher.

Naht-Regel: kein run/torch-Import.
"""

from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
from dash import dcc, html

_MUTED = {"color": "#6b7280", "fontSize": "12px"}


# Kopier-Button (dcc.Clipboard): kopiert den Kernel-Quelltext komplett clientseitig
# in die Zwischenablage (kein Callback nötig, mit Browser-Fallbacks). Als kleiner
# Button gestylt; das Icon wechselt beim Klick kurz auf ein Häkchen.
_COPY_STYLE = {
    "cursor": "pointer",
    "color": "#5b21b6",
    "fontSize": "15px",
    "border": "1px solid #e4e7ec",
    "borderRadius": "6px",
    "padding": "3px 8px",
    "background": "#fff",
    "flexShrink": 0,
}


def _header(kernel_path: Optional[str], source: Optional[str] = None):
    title = [html.Strong("Generiertes cuTile-Kernel")]
    if kernel_path:
        title.append(html.Span(f"  {kernel_path}",
                               style={**_MUTED, "fontFamily": "ui-monospace, monospace"}))
    row = [html.Div(title)]
    # Kopier-Button nur, wenn es überhaupt Quelltext zu kopieren gibt.
    if source:
        row.append(dcc.Clipboard(content=source, title="Kernel in die Zwischenablage kopieren",
                                 style=_COPY_STYLE))
    return html.Div(row, className="mb-2",
                    style={"display": "flex", "alignItems": "center",
                           "justifyContent": "space-between", "gap": "8px"})


def render_code_panel(source: Optional[str], kernel_path: Optional[str] = None):
    """Karte mit dem generierten Quelltext (oder Hinweis, wenn keiner vorliegt).
    Bei vorhandenem Quelltext trägt der Header einen Kopier-Button (dcc.Clipboard),
    der den kompletten Kernel-Quelltext in die Zwischenablage kopiert."""
    if not source:
        return dbc.Card(dbc.CardBody(
            [_header(kernel_path),
             html.Span("Kein generierter Kernel verfügbar.", style=_MUTED)]
        ))
    code = dcc.Markdown(
        f"```python\n{source}\n```",
        style={"maxHeight": "440px", "overflow": "auto", "fontSize": "12.5px", "marginBottom": 0},
    )
    return dbc.Card(dbc.CardBody([_header(kernel_path, source), code]))
