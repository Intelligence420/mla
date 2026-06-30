"""Entry point: ``python -m tool_pipeline`` launches the Dash GUI.

Thin wrapper around ``tool_pipeline.app.app``. The GUI's ONLY coupling to the core
is ``tool_pipeline.run.run(config) -> result`` — see project/README.md (Architektur).
TODO: ticket D.1 (wire to tool_pipeline.app.app.main()).
"""
