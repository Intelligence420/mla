"""Headless-Test der Callback-Kernlogik ``execute_run`` (TZ 3: Batch-Vergleich).

Fährt den **echten** Lauf-Pfad ohne Dash-Server: validieren → RunConfig je Format
→ EIN GPU-Lock → ``run()`` je Format → gerenderte Main-Komponenten (zwei
Vergleichs-Charts + Primär-Detail). Damit ist die Logik hinter dem Background-
Callback headless geprüft; das Dash-Plumbing (running=/progress=/cancel=,
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
from tool_pipeline.app.components.controls import combo_key  # noqa: E402

_TMP_JSONL = Path(os.environ.get("SP", "/tmp")) / "execute_probe.jsonl"
_SEL = [combo_key("fp16", "fp32"), combo_key("bf16", "fp32")]  # kleiner 2-Format-Batch


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


def _types(node, acc=None) -> list:
    """Alle Komponenten-Typen im Baum sammeln (rekursiv) — z. B. um 'Graph' zu zählen."""
    acc = [] if acc is None else acc
    if hasattr(node, "to_plotly_json"):
        j = node.to_plotly_json()
        acc.append(j.get("type"))
        ch = (j.get("props", {}) or {}).get("children")
        if isinstance(ch, (list, tuple)):
            for c in ch:
                _types(c, acc)
        elif ch is not None:
            _types(ch, acc)
    elif isinstance(node, (list, tuple)):
        for c in node:
            _types(c, acc)
    return acc


def _redirect_store():
    """store.append_result → temp-JSONL (results.jsonl unberührt); gibt (restore)."""
    import tool_pipeline.store.store as S
    orig = S.append_result
    if _TMP_JSONL.exists():
        _TMP_JSONL.unlink()
    S.append_result = lambda r, path=None: orig(r, path=_TMP_JSONL)
    return lambda: setattr(S, "append_result", orig)


def test_execute_batch_renders_charts_kpis_verify_code():
    """Gültiger 2-Format-Batch → echter GPU-Lauf: zwei Charts + Status ok + KPIs +
    Verify PASS + Code des primären Formats; beide Formate im Status-Strip."""
    restore = _redirect_store()
    try:
        comps = execute_run(128, 128, 64, _SEL)   # i=M, k=K, j=N (glatte Tile-Vielfache)
    finally:
        restore()
    assert isinstance(comps, list) and comps, "execute_run muss eine nicht-leere Liste liefern"
    types = _types(comps)
    assert types.count("Graph") == 2, "es müssen zwei Vergleichs-Charts (dcc.Graph) da sein"
    assert types.count("Tab") == 2, "je Format ein Detail-Tab (durchklickbar)"
    txt = _text(comps)
    assert "erfolgreich" in txt, f"kein ok-Status: {txt[:200]}"
    assert "TFLOP/s" in txt and "PASS" in txt, f"KPIs/Verify fehlen: {txt[:300]}"
    assert "ct.mma" in txt, "generierter Kernel-Quelltext fehlt im Code-Panel"
    assert "fp16 → fp32" in txt and "bf16 → fp32" in txt, "Status-Strip zeigt nicht beide Formate"


def test_execute_with_tile_swizzle_baselines():
    """Nicht-Default-Tile (64/64/32) + Swizzle + beide Baselines fließen durch →
    echter Lauf mit zwei Charts; der Lauf verifiziert (kein Crash)."""
    restore = _redirect_store()
    try:
        comps = execute_run(256, 256, 128, [combo_key("fp16", "fp32")],
                            tm=64, tn=64, tk=32, swizzle=True,
                            baselines=["cublas", "naive"])
    finally:
        restore()
    assert isinstance(comps, list) and comps
    assert _types(comps).count("Graph") == 2, "zwei Vergleichs-Charts erwartet"
    txt = _text(comps)
    assert "erfolgreich" in txt and "PASS" in txt, f"kein ok/Verify: {txt[:300]}"


def test_execute_invalid_tile_no_run():
    """Unzulässiger Tile-Wert → Warnung, KEIN GPU-Lauf."""
    comps = execute_run(128, 128, 64, [combo_key("fp16", "fp32")], tm=48, tn=128, tk=64)
    txt = _text(comps)
    assert "Ungültige Kachelung" in txt, txt[:200]
    assert _types(comps).count("Graph") == 0, "kein Lauf bei ungültigem Tile erwartet"


def test_execute_invalid_sizes_no_run():
    """Ungültige Größe → Warnung, KEIN GPU-Lauf (keine Charts/KPIs/Code)."""
    comps = execute_run(0, 128, 64, _SEL)
    txt = _text(comps)
    assert "Ungültige Eingabe" in txt, txt[:200]
    assert _types(comps).count("Graph") == 0 and "ct.mma" not in txt, "kein Lauf erwartet"


def test_execute_empty_selection_no_run():
    """Leere Format-Auswahl → Warnung, KEIN GPU-Lauf."""
    comps = execute_run(128, 128, 64, [])
    txt = _text(comps)
    assert "Ungültige Auswahl" in txt, txt[:200]
    assert _types(comps).count("Graph") == 0, "kein Lauf/Chart bei leerer Auswahl erwartet"


def test_execute_survives_run_import_failure():
    """execute_run wirft NIE: schlägt der lazy `run`-Import fehl (torch/cuda.tile im
    Worker kaputt/abwesend), kommt ein 'Interner Fehler'-Alert statt einer Exception
    (Naht-Vertrag; Fund A des Error-Audits). Kein GPU nötig (Import gestubbt)."""
    import types
    stub = types.ModuleType("tool_pipeline.run")   # Modul OHNE 'run'-Attribut
    orig = sys.modules.get("tool_pipeline.run")
    sys.modules["tool_pipeline.run"] = stub
    try:
        comps = execute_run(4, 4, 4, _SEL)         # gültig → Pfad erreicht den Import
    finally:
        if orig is not None:
            sys.modules["tool_pipeline.run"] = orig
        else:
            sys.modules.pop("tool_pipeline.run", None)
    assert isinstance(comps, list) and comps, "execute_run muss eine Liste liefern, nicht werfen"
    assert "Interner Fehler" in _text(comps), _text(comps)[:200]


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
