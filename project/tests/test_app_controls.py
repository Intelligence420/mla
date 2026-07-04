"""Headless-Tests der Dash-freien Controls-Logik (TZ 2 / TODO 3).

Die GUI ist schwer headless zu prüfen — deshalb ist die entscheidende Logik
(Control-Werte → RunConfig, Eingabe-Validierung) bewusst aus dem Callback
herausgezogen und hier ohne Dash-Server + ohne GPU geprüft. Der Rest der GUI
(Layout-Mount, Live-Callback) wird durch das reale Starten der App abgedeckt.

Lauffähig standalone (`python tests/test_app_controls.py`, aus `project/`) **und**
via pytest. Braucht nur `dash`/`schema`, KEIN torch/cuTile/GPU.
"""

from __future__ import annotations

import os
import sys

# project/ auf den Pfad, damit `tool_pipeline` importierbar ist (standalone-Lauf).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_pipeline.app.components.controls import (  # noqa: E402
    config_from_controls,
    validate_sizes,
)
from tool_pipeline.schema import RunConfig  # noqa: E402


def test_config_maps_sizes_to_correct_axes():
    """i=M (Zeilen), k=K (Kontraktion), j=N (Spalten) — wie cli.build_config."""
    cfg = config_from_controls(m=64, n=128, k=32)
    assert cfg.dim_sizes == {"i": 64, "k": 32, "j": 128}, cfg.dim_sizes


def test_config_keeps_tz1_defaults():
    """Nur die Größen ändern sich; family/expr/dtype/acc/tile/swizzle = Default."""
    d = RunConfig()
    cfg = config_from_controls(10, 20, 30)
    assert cfg.family == d.family == "contraction"
    assert cfg.expr == d.expr == "ik,kj->ij"
    assert (cfg.dtype, cfg.acc_dtype) == (d.dtype, d.acc_dtype) == ("fp16", "fp32")
    assert cfg.tile == d.tile == {"TM": 128, "TN": 128, "TK": 64}
    assert cfg.swizzle == d.swizzle is False
    # expr treibt inputs/output (via __post_init__) unverändert
    assert cfg.inputs == ["ik", "kj"] and cfg.output == "ij"


def test_config_tolerates_float_and_string_inputs():
    """Dash-Number-Input kann 512.0 oder "512" liefern → sauber zu int coercen."""
    assert config_from_controls(512.0, 256.0, 128.0).dim_sizes == {"i": 512, "k": 128, "j": 256}
    assert config_from_controls("64", "32", "16").dim_sizes == {"i": 64, "k": 16, "j": 32}


def test_validate_accepts_positive_ints():
    """Gültige Größen → None (kein Fehler)."""
    assert validate_sizes(512, 512, 512) is None
    assert validate_sizes(1, 1, 1) is None
    assert validate_sizes(512.0, 256, "128") is None  # float/str-Ganzzahlen ok


def test_validate_rejects_bad_inputs():
    """Leer/None/0/negativ/nicht-ganzzahlig/nicht-numerisch → deutscher Fehlertext."""
    bad = [
        (None, 8, 8), ("", 8, 8),          # fehlend
        (0, 8, 8), (8, -3, 8),             # <= 0
        (8, 8, 512.5),                     # nicht ganzzahlig
        (8, "x", 8),                       # nicht numerisch
    ]
    for m, n, k in bad:
        msg = validate_sizes(m, n, k)
        assert isinstance(msg, str) and msg, f"erwartete Fehlermeldung für {(m, n, k)}, bekam {msg!r}"


def test_validate_names_the_offending_dimension():
    """Die Fehlermeldung nennt die betroffene Dimension (M/N/K)."""
    assert validate_sizes(10, 0, 10).startswith("N")
    assert validate_sizes(-1, 10, 10).startswith("M")
    assert validate_sizes(10, 10, None).startswith("K")


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} Tests bestanden")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
