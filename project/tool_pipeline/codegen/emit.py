"""tool_pipeline.codegen.emit — Config → generierter cuTile-Quelltext.

`emit(config)` ist der **Dirigent** der Codegen-Stufe (C1): Er routet auf das
Template der Operations-Familie, ruft dessen Builder und stellt dem erzeugten
Modul einen **deterministischen** Kopf-Kommentar voran (Ausdruck/dtype/Tile —
für die spätere UI-Code-Anzeige und zur Nachvollziehbarkeit des persistierten
Artefakts, Risiko ③). Bewusst **kein** Zeitstempel im Quelltext: der Text muss
byte-stabil sein, sonst weicht der Datei-Inhalt bei gleichem Slug ab.

Familien (TZ 7): `contraction` → GEMM-Template (Tensor-Core), `elementwise` →
Elementwise-Template (add/mul/copy), `reduction` → Reduktions-Template (Summe).
Der Kopf-Kommentar ist family-abhängig (zeigt nur die wirklich genutzten
Tile-Achsen + die Op), bleibt aber ohne Zeitstempel byte-stabil.
"""

from __future__ import annotations

from ..schema import RunConfig
from .templates.contraction import build_gemm_module
from .templates.elementwise import build_elementwise_module
from .templates.reduction import build_reduction_module


def _header(config: RunConfig) -> str:
    """Deterministischer, family-abhängiger Kopf-Kommentar (ohne Zeitstempel →
    byte-stabil). Zeigt nur die für die Familie relevanten Tile-Achsen/Op.

    Der **Kontraktions**-Header ist bewusst **byte-identisch** zu TZ 1-6 (keine
    zusätzliche Zeile) — sonst würden die git-getrackten `results/kernels/*.py`
    beim nächsten Lauf umgeschrieben. Die neuen Familien (eigene Dateien) tragen
    zusätzlich eine `Familie`-Zeile und nennen nur ihre genutzten Tile-Achsen + Op.
    """
    t = config.tile
    fam = config.family
    head = [
        "# " + "=" * 74,
        "# Auto-generiert vom cuTile Performance Lab (Codegen C1).",
        "# Aus einer RunConfig erzeugt.",
        f"# Ausdruck : {config.expr}",
    ]
    if fam == "contraction":
        head.append(f"# Format   : {config.dtype} -> {config.acc_dtype} (Akku)")
        head.append(f"# Tile     : TM={t.get('TM')} TN={t.get('TN')} TK={t.get('TK')}"
                    f" | swizzle={config.swizzle}")
    elif fam == "elementwise":
        head.append("# Familie  : elementwise")
        head.append(f"# Format   : {config.dtype} -> {config.acc_dtype} (Ausgabe)")
        head.append(f"# Tile     : TM={t.get('TM')} TN={t.get('TN')} | op={config.op}")
    elif fam == "reduction":
        head.append("# Familie  : reduction")
        head.append(f"# Format   : {config.dtype} -> {config.acc_dtype} (Ausgabe)")
        head.append(f"# Tile     : TK={t.get('TK')} (LOOP_TILE-Fallback) | op=sum")
    else:
        head.append(f"# Format   : {config.dtype} -> {config.acc_dtype}")
        head.append(f"# Tile     : {t}")
    head.append("# " + "=" * 74)
    return "\n".join(head) + "\n"


def emit(config: RunConfig) -> str:
    """`RunConfig` → vollständiger, ausführbarer cuTile-Modul-Quelltext.

    Routet auf das Template der Operations-Familie (`contraction`/`elementwise`/
    `reduction`).

    Raises:
        ValueError: unbekannte Familie oder (Elementwise) fehlendes `op`.
    """
    if config.family == "contraction":
        body = build_gemm_module(config.tile, config.dtype, config.acc_dtype,
                                 swizzle=config.swizzle, group_m=config.group_m)
    elif config.family == "elementwise":
        if not config.op:
            raise ValueError(
                "Elementwise braucht ein op (add/mul/copy) in der RunConfig."
            )
        body = build_elementwise_module(config.tile, config.dtype, config.acc_dtype,
                                        config.op)
    elif config.family == "reduction":
        body = build_reduction_module(config.tile, config.dtype, config.acc_dtype)
    else:
        raise ValueError(
            f"unbekannte Operations-Familie {config.family!r} "
            f"(erlaubt: 'contraction', 'elementwise', 'reduction')."
        )
    return _header(config) + body
