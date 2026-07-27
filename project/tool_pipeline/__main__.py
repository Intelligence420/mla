"""Einstiegspunkt: ``python -m tool_pipeline`` startet die Dash-GUI.

Dünner Wrapper um `tool_pipeline.app.app.main()`. Die einzige Kopplung der GUI an
den Core ist ``tool_pipeline.run.run(config) -> result`` — siehe project/README.md.
Aufruf (aus `project/`, venv aktiv)::

    python -m tool_pipeline            # Dash-GUI
    python -m tool_pipeline.cli ...    # headless / Batch (TZ 1)
"""

from __future__ import annotations

from .app.app import main

if __name__ == "__main__":
    main()
