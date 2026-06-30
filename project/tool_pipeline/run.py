"""tool_pipeline.run — the orchestrator AND the core<->GUI contract.

run(config: RunConfig) -> RunResult:
    parse -> canonical reshape (B1) -> emit cuTile source (C1) ->
    compile (+cache) -> verify vs fp32 -> measure -> persist to store.

This single entry point is what the Dash app calls inside its background job.
TODO: tickets A/B/C.
"""
