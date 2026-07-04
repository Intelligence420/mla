"""Dash-Einstieg: App-Instanz, DiskcacheManager (Background-Jobs), Layout-Mount, Server-Start.

`create_app()` baut die Dash-App (inkl. Background-Callback-Manager) und hängt das
Layout ein; `main()` startet den Server. Aufgerufen wird das über
``python -m tool_pipeline`` (siehe `tool_pipeline/__main__.py`).

**Bewusst CUDA-/torch-frei:** dieses Modul importiert **nicht** `tool_pipeline.run`
(und damit weder torch noch cuda.tile). Der DiskcacheManager startet
Background-Callbacks in Worker-**Prozessen** (fork); hielte der Haupt-Prozess bereits
einen CUDA-Kontext, wäre der im Fork kaputt. Deshalb importiert erst der
Callback-*Body* (im Worker) die eine Naht `run()` — siehe `callbacks.py` (TZ 2 / TODO 6).

Host/Port sind über die Umgebungsvariablen ``TP_HOST`` / ``TP_PORT`` überschreibbar
(Default ``127.0.0.1:8050``); z. B. ``TP_HOST=0.0.0.0`` für Zugriff über SSH-Tunnel/LAN.
"""

from __future__ import annotations

import os
from pathlib import Path

import dash_bootstrap_components as dbc
import diskcache
from dash import Dash, DiskcacheManager

from .layout import build_layout

# Background-Cache = prozessübergreifende Job-Koordination (Main <-> Worker).
# Bewusst OHNE Import von `store` (Naht-Regel: app/ importiert nur run + schema) —
# der Pfad wird aus dem eigenen __file__ abgeleitet: app/ -> tool_pipeline/ -> project/.
_PROJECT_DIR = Path(__file__).resolve().parents[2]
_CACHE_DIR = _PROJECT_DIR / ".cache" / "dash_bg"


def create_app() -> Dash:
    """Baue die Dash-App: Background-Manager + Bootstrap-Theme + Layout."""
    # Cache-Init kann auf read-only/rechtebeschränktem FS scheitern (mkdir → OSError,
    # diskcache → sqlite OperationalError). Statt eines rohen Tracebacks eine klare,
    # handlungsleitende Meldung (fail-fast: ohne Background-Cache läuft die GUI nicht).
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        manager = DiskcacheManager(diskcache.Cache(str(_CACHE_DIR)))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Background-Cache unter {_CACHE_DIR} nicht nutzbar ({type(e).__name__}: {e}). "
            f"Ist das Projektverzeichnis beschreibbar? Rechte prüfen oder .cache/ freigeben."
        ) from e

    app = Dash(
        __name__,
        title="cuTile Performance Lab",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        background_callback_manager=manager,
        update_title=None,  # kein "Updating..."-Flackern im Browser-Tab
    )
    app.layout = build_layout()

    # Callbacks werden in TZ 2 / TODO 6 registriert (Import mit Seiteneffekt: die
    # @callback-Dekoratoren tragen sich in die globale Registry ein). Bewusst hier
    # (nicht im Modulkopf) und erst wenn callbacks.py gefüllt ist, damit der
    # Haupt-Prozess `run()`/torch NICHT importiert.
    from . import callbacks  # noqa: F401

    if hasattr(callbacks, "register"):
        callbacks.register(app)

    return app


def _host() -> str:
    """TP_HOST oder sicherer Default. Ein **leer** gesetztes TP_HOST fällt bewusst
    auf 127.0.0.1 zurück (nicht 0.0.0.0) — sonst öffnete `TP_HOST=` die GUI auf der
    geteilten Maschine ungewollt im LAN (``os.environ.get(k, default)`` liefert bei
    gesetztem, leerem Key den Leerstring, nicht den Default)."""
    return os.environ.get("TP_HOST") or "127.0.0.1"


def _port(default: int = 8050) -> int:
    """TP_PORT als bindbarer Port; leer/nicht-numerisch/außerhalb 1–65535 → Default
    mit Warnung (statt rohem ValueError beim Start)."""
    raw = os.environ.get("TP_PORT", "")
    if not raw:
        return default
    try:
        p = int(raw)
    except ValueError:
        print(f"[cuTile Performance Lab] TP_PORT={raw!r} ist keine Zahl — nutze {default}.")
        return default
    if not (1 <= p <= 65535):
        print(f"[cuTile Performance Lab] TP_PORT={p} außerhalb 1–65535 — nutze {default}.")
        return default
    return p


def main() -> None:
    """Starte den Dash-Server (blockierend). Einstieg von ``python -m tool_pipeline``."""
    app = create_app()
    app.run(
        host=_host(),
        port=_port(),
        debug=False,  # kein Reloader (geteilte GPU-Maschine; Reloader + Fork-Manager beißen sich)
    )


if __name__ == "__main__":
    main()
