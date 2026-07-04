"""Headless-Test der Callback-Kernlogik ``execute_run`` (TZ 2 / TODO 6).

Fährt den **echten** Lauf-Pfad ohne Dash-Server: validieren → RunConfig →
GPU-Lock → ``run()`` → gerenderte Main-Komponenten. Damit ist die Logik hinter
dem Background-Callback headless geprüft; das Dash-Plumbing (running=/cancel=,
Worker-Prozess) deckt der reale Server-Smoke ab.

Braucht GPU + cuTile. ``results.jsonl`` wird in eine temp-Datei umgeleitet
(kein Pollution). Standalone (`python tests/test_app_execute.py`, aus `project/`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_pipeline.app.callbacks import execute_run  # noqa: E402

_TMP_JSONL = Path(os.environ.get("SP", "/tmp")) / "execute_probe.jsonl"


def _text(node) -> str:
    """Alle String-Blätter aus einem Dash-Komponentenbaum (rekursiv)."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, (list, tuple)):
        return " ".join(_text(n) for n in node)
    props = {}
    if hasattr(node, "to_plotly_json"):
        j = node.to_plotly_json()
        if isinstance(j, dict):
            props = j.get("props", {}) or {}
    out = [_text(props.get("children"))]
    for key in ("value", "label", "header", "children"):
        v = props.get(key)
        if isinstance(v, str):
            out.append(v)
    return " ".join(x for x in out if x)


def _redirect_store():
    """store.append_result → temp-JSONL (results.jsonl unberührt); gibt (restore)."""
    import tool_pipeline.store.store as S
    orig = S.append_result
    if _TMP_JSONL.exists():
        _TMP_JSONL.unlink()
    S.append_result = lambda r, path=None: orig(r, path=_TMP_JSONL)
    return lambda: setattr(S, "append_result", orig)


def test_execute_valid_run_renders_kpis_verify_code():
    """Gültige Größen → echter GPU-Lauf → Status ok + KPIs + Verify PASS + Code."""
    restore = _redirect_store()
    try:
        comps = execute_run(m=128, n=128, k=64)   # i=M, k=K, j=N (glatte Tile-Vielfache)
    finally:
        restore()
    assert isinstance(comps, list) and comps, "execute_run muss eine nicht-leere Liste liefern"
    txt = _text(comps)
    assert "erfolgreich" in txt, f"kein ok-Status: {txt[:200]}"
    assert "TFLOP/s" in txt and "PASS" in txt, f"KPIs/Verify fehlen: {txt[:300]}"
    assert "ct.mma" in txt, "generierter Kernel-Quelltext fehlt im Code-Panel"


def test_execute_invalid_sizes_no_run():
    """Ungültige Größe → Warnung, KEIN GPU-Lauf (keine KPIs, kein Code)."""
    comps = execute_run(m=0, n=128, k=64)
    txt = _text(comps)
    assert "Ungültige Eingabe" in txt, txt[:200]
    assert "TFLOP/s" not in txt and "ct.mma" not in txt, "es hätte kein Lauf stattfinden dürfen"


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
