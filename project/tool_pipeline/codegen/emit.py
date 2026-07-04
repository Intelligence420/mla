"""tool_pipeline.codegen.emit — Config → generierter cuTile-Quelltext.

`emit(config)` ist der **Dirigent** der Codegen-Stufe (C1): Er routet auf das
Template der Operations-Familie, ruft dessen Builder und stellt dem erzeugten
Modul einen **deterministischen** Kopf-Kommentar voran (Ausdruck/dtype/Tile —
für die spätere UI-Code-Anzeige und zur Nachvollziehbarkeit des persistierten
Artefakts, Risiko ③). Bewusst **kein** Zeitstempel im Quelltext: der Text muss
byte-stabil sein, sonst weicht der Datei-Inhalt bei gleichem Slug ab.

TZ 1: nur `family="contraction"` (Plain-GEMM). Andere Familien (elementwise/
reduction = TZ 7) lösen einen klaren `NotImplementedError` aus — die Routing-
Naht steht damit schon, wird aber jetzt nicht ausgefüllt.
"""

from __future__ import annotations

from ..schema import RunConfig
from .templates.contraction import build_gemm_module


def _header(config: RunConfig) -> str:
    """Deterministischer Kopf-Kommentar (ohne Zeitstempel → byte-stabil)."""
    t = config.tile
    return (
        "# " + "=" * 74 + "\n"
        "# Auto-generiert vom cuTile Performance Lab (Codegen C1).\n"
        "# Aus einer RunConfig erzeugt.\n"
        f"# Ausdruck : {config.expr}\n"
        f"# Format   : {config.dtype} -> {config.acc_dtype} (Akku)\n"
        f"# Tile     : TM={t.get('TM')} TN={t.get('TN')} TK={t.get('TK')}"
        f" | swizzle={config.swizzle}\n"
        "# " + "=" * 74 + "\n"
    )


def emit(config: RunConfig) -> str:
    """`RunConfig` → vollständiger, ausführbarer cuTile-Modul-Quelltext.

    Raises:
        NotImplementedError: für Familien außer `contraction` (TZ 7).
    """
    if config.family == "contraction":
        body = build_gemm_module(config.tile, config.dtype, config.acc_dtype)
    else:
        raise NotImplementedError(
            f"TZ 1: nur family='contraction'; '{config.family}' "
            f"(elementwise/reduction) ist TZ 7."
        )
    return _header(config) + body
