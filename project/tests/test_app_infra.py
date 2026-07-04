"""Headless-Tests der App-Infra-Robustheit (TZ 2 / Error-Audit): Host/Port-Env-Parsing.

Prüft ``_host()``/``_port()`` (project/tool_pipeline/app/app.py) gegen
fehlkonfigurierte Umgebungsvariablen: ein leeres ``TP_HOST`` darf NICHT auf
0.0.0.0 binden (ungewollte LAN-Exposition), ein ungültiger ``TP_PORT`` darf nicht
mit rohem ValueError crashen. Kein Server, kein GPU.

Standalone (`python tests/test_app_infra.py`, aus `project/`) und via pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_pipeline.app.app import _host, _port  # noqa: E402


def _set(key: str, val):
    old = os.environ.get(key)
    if val is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = val
    return old


def _restore(key: str, old):
    if old is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = old


def test_host_empty_falls_back_to_localhost():
    """Fund B: leeres TP_HOST → 127.0.0.1 (nicht 0.0.0.0); explizit gesetzter Wert bleibt."""
    for val, want in [(None, "127.0.0.1"), ("", "127.0.0.1"),
                      ("0.0.0.0", "0.0.0.0"), ("127.0.0.1", "127.0.0.1")]:
        old = _set("TP_HOST", val)
        try:
            assert _host() == want, f"TP_HOST={val!r} -> {_host()!r}, erwartet {want!r}"
        finally:
            _restore("TP_HOST", old)


def test_port_parsing_robust():
    """Fund C: leer/nicht-numerisch/außerhalb 1–65535 → 8050 (kein Crash); gültig bleibt."""
    cases = [(None, 8050), ("", 8050), ("auto", 8050), (":8050", 8050),
             ("0", 8050), ("99999", 8050), ("-1", 8050), ("8097", 8097)]
    for val, want in cases:
        old = _set("TP_PORT", val)
        try:
            assert _port() == want, f"TP_PORT={val!r} -> {_port()}, erwartet {want}"
        finally:
            _restore("TP_PORT", old)


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
