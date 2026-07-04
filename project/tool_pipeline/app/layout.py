"""Seiten-Layout: Controls-Sidebar (links) + Ergebnis-Main (KPIs / Code) rechts.

TZ 2 / TODO 2 = **Skelett**: Topbar + zweispaltiges Grundgerüst mit klaren
Einhängepunkten. Die Platzhalter werden inkrementell ersetzt:

* Sidebar-Inhalt  -> `components/controls.py`   (TODO 3)
* Main-Inhalt     -> `components/kpis.py` + `components/code_panel.py` (TODO 4)
* Verdrahtung     -> `callbacks.py`             (TODO 6)

Nur **strukturelle** Inline-Styles (Flex/Breiten/Abstände), damit das Gerüst ohne
Theme lauffähig aussieht; Farben/Politur macht `assets/theme.css` (TODO 7).
"""

from __future__ import annotations

from dash import html

APP_TITLE = "einsum / GEMM Performance-Explorer"
APP_SUBTITLE = "cuTile · live auf der GPU"

# --- strukturelle Inline-Styles (TODO 7 migriert das nach theme.css) ---------
_TOPBAR = {
    "display": "flex",
    "alignItems": "center",
    "gap": "12px",
    "padding": "14px 22px",
    "background": "#2d2a45",
    "color": "#fff",
}
_BODY = {"display": "flex", "minHeight": "calc(100vh - 52px)"}
_SIDEBAR = {
    "width": "320px",
    "flexShrink": 0,
    "padding": "18px 16px",
    "borderRight": "1px solid #e4e7ec",
    "background": "#fafafc",
}
_MAIN = {"flex": 1, "padding": "20px 24px", "overflow": "auto"}
_PLACEHOLDER = {"color": "#6b7280", "fontSize": "14px"}


def _topbar() -> html.Header:
    return html.Header(
        style=_TOPBAR,
        children=[
            html.Span("⚡", style={"fontSize": "18px"}),
            html.H1(APP_TITLE, style={"fontSize": "17px", "margin": 0, "fontWeight": 650}),
            html.Span(APP_SUBTITLE, style={"marginLeft": "auto", "opacity": 0.85, "fontSize": "12.5px"}),
        ],
    )


def _sidebar_placeholder() -> html.Div:
    return html.Div(
        style=_PLACEHOLDER,
        children=[
            html.H2("Steuerung", style={"fontSize": "12px", "letterSpacing": "0.08em",
                                        "textTransform": "uppercase", "color": "#6b7280"}),
            html.P("Größen-Eingaben + Run folgen (TODO 3)."),
        ],
    )


def _main_placeholder() -> html.Div:
    return html.Div(
        style=_PLACEHOLDER,
        children=html.P("Noch kein Lauf. Größen wählen und „Run“ starten (TODO 4/6)."),
    )


def build_layout() -> html.Div:
    """Top-Level-Layout: Topbar + Sidebar (id='sidebar') + Main (id='main')."""
    return html.Div(
        children=[
            _topbar(),
            html.Div(
                style=_BODY,
                children=[
                    html.Aside(id="sidebar", style=_SIDEBAR, children=_sidebar_placeholder()),
                    html.Main(id="main", style=_MAIN, children=_main_placeholder()),
                ],
            ),
        ]
    )
