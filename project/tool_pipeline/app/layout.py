"""Seiten-Layout: Controls-Sidebar (links) + Ergebnis-Main (KPIs / Code) rechts.

Grundgerüst (Topbar · Sidebar · Main) mit CSS-Klassen; das Styling liegt in
``assets/theme.css`` (TZ 2 / TODO 7). Sidebar-Inhalt = ``components/controls``;
den Main-Bereich (id='main') füllt der Callback mit ``components/kpis`` +
``components/code_panel`` (siehe ``callbacks.py``).
"""

from __future__ import annotations

from dash import dcc, html

from .components import controls

APP_TITLE = "cuTile Performance Lab"
APP_SUBTITLE = "einsum/GEMM · live generiert · verifiziert · gemessen"


def _topbar() -> html.Header:
    return html.Header(
        className="topbar",
        children=[
            html.H1(APP_TITLE),
            html.Span(APP_SUBTITLE, className="topbar-sub"),
        ],
    )


def _main_placeholder() -> html.Div:
    return html.Div(
        className="main-placeholder",
        children=html.P("Noch kein Lauf. Ausdruck & Größen wählen und „Vergleichen“ starten."),
    )


def build_layout() -> html.Div:
    """Top-Level-Layout: Topbar + Sidebar (id='sidebar') + Main (id='main')."""
    return html.Div(
        className="app-shell",
        children=[
            _topbar(),
            html.Div(
                className="app-body",
                children=[
                    html.Aside(id="sidebar", className="sidebar", children=controls.build_controls()),
                    html.Main(id="main", className="main", children=_main_placeholder()),
                ],
            ),
            # Wegwerf-Ziel für den Clientside-Scroll-Callback (siehe callbacks.register):
            # bei einer Abbruch-Meldung im Main wird nach ganz oben gescrollt.
            dcc.Store(id="_scroll_dummy"),
        ],
    )
