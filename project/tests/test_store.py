"""Headless-Tests der Testlauf-Verwaltung (TZ 7.5-4): verlustfreies Lesen/Gruppieren
(`read_all`/`list_runs`), atomare Mutatoren (`rename_run`/`delete_run`) und die reine
History-Komponenten-Logik.

**GPU-frei** (reiner Store-Zugriff). Der git-getrackte `results.jsonl` wird NIE
angefasst: alle Mutationen laufen gegen eine Temp-JSONL unter ``$SP`` (Fallback
``/tmp``) — exakt die $SP-Isolation aus ``test_app_execute``, hier auf die Mutatoren
ausgeweitet. Standalone (``python tests/test_store.py``, aus ``project/``) **oder**
via pytest.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_pipeline.app.components import history as H  # noqa: E402
from tool_pipeline.codegen import compile as C  # noqa: E402
from tool_pipeline.schema import RunResult  # noqa: E402
from tool_pipeline.store import store as S  # noqa: E402

_SP = Path(os.environ.get("SP", "/tmp"))


def _fresh(name: str = "store_test.jsonl") -> Path:
    """Frische Temp-JSONL unter $SP (der git-getrackte Store bleibt unberührt)."""
    p = _SP / name
    if p.exists():
        p.unlink()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _res(rid, name, ca, dtype="fp16", status="ok") -> RunResult:
    return RunResult(
        status=status,
        config={"expr": "ik,kj->ij", "dtype": dtype, "acc_dtype": "fp32", "family": "contraction"},
        run_id=rid, run_name=name, created_at=ca,
        provenance={"timestamp": ca}, metrics={"tflops": 1.0})


def _seed(path: Path) -> None:
    """Zwei benannte Läufe (AAA=2 Zeilen, BBB=1 Zeile) + eine Altzeile ohne run_id."""
    S.append_result(_res("AAA", "Lauf A", "2026-07-12T10:00:00", "fp16"), path=path)
    S.append_result(_res("AAA", "Lauf A", "2026-07-12T10:00:00", "bf16"), path=path)
    S.append_result(_res("BBB", "Lauf B", "2026-07-12T11:00:00", status="verify_failed"), path=path)
    with path.open("a", encoding="utf-8") as f:   # Altzeile (Bestandszeile ohne Identität)
        f.write(json.dumps({"status": "ok",
                            "config": {"expr": "ab,bc->ac", "dtype": "tf32", "acc_dtype": "fp32"},
                            "provenance": {"timestamp": "2026-07-01T09:00:00"},
                            "metrics": {"tflops": 2.0}}) + "\n")


def test_read_all_objects_and_legacy_fallback():
    """read_all liefert RunResult-Objekte; Altzeilen ohne run_id bekommen einen
    synthetischen Fallback-Lauf (jede Altzeile = eigener Lauf)."""
    p = _fresh()
    _seed(p)
    allr = S.read_all(p)
    assert len(allr) == 4 and all(isinstance(x, RunResult) for x in allr)
    legacy = [x for x in allr if (x.run_id or "").startswith("legacy-")]
    assert len(legacy) == 1 and legacy[0].run_name.startswith("(Altlauf)")
    p.unlink()


def test_list_runs_grouped_sorted():
    """list_runs gruppiert nach run_id (AAA=2/2ok, BBB=1/0ok, legacy=1/1ok), neueste
    zuerst (created_at absteigend)."""
    p = _fresh()
    _seed(p)
    runs = S.list_runs(p)
    by = {r["run_id"]: r for r in runs}
    assert len(runs) == 3
    assert by["AAA"]["n"] == 2 and by["AAA"]["n_ok"] == 2
    assert by["BBB"]["n"] == 1 and by["BBB"]["n_ok"] == 0
    assert runs[0]["run_id"] == "BBB", "neuester Lauf (11:00) zuerst"
    p.unlink()


def test_rename_run_only_group():
    """rename_run trifft NUR die Zielgruppe (2 Zeilen), lässt die Zeilenzahl gleich."""
    p = _fresh()
    _seed(p)
    n = S.rename_run("AAA", "Umbenannt!", path=p)
    assert n == 2
    after = {x.run_id: x.run_name for x in S.read_all(p)}
    assert after["AAA"] == "Umbenannt!" and after["BBB"] == "Lauf B"
    assert len(S._read_rows(p)) == 4, "rename ändert die Zeilenzahl nicht"
    p.unlink()


def test_delete_run_atomic_and_keeps_kernels():
    """delete_run entfernt NUR die JSONL-Zeilen der Gruppe (atomar) und fasst die
    geteilte kernels/<slug>.py NIE an."""
    p = _fresh()
    _seed(p)
    # Sentinel-Kernel-Datei (simuliert eine geteilte kernels/<slug>.py)
    kdir = _SP / "kernels_probe"
    kdir.mkdir(parents=True, exist_ok=True)
    sentinel = kdir / "ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64.py"
    sentinel.write_text("# sentinel", encoding="utf-8")

    removed = S.delete_run("AAA", path=p)
    assert removed == 2
    rest = {x.run_id for x in S.read_all(p)}
    assert "AAA" not in rest and "BBB" in rest and len(S._read_rows(p)) == 2
    # Byte-Form sauber (gültige JSON-Zeilen, kein kernel_source)
    for line in p.read_text(encoding="utf-8").splitlines():
        assert "kernel_source" not in json.loads(line)
    # Kernel-Datei UNBERÜHRT
    assert sentinel.exists() and sentinel.read_text(encoding="utf-8") == "# sentinel"
    sentinel.unlink(); kdir.rmdir(); p.unlink()


def test_atomic_rewrite_empty_is_zero_byte():
    """Leere Liste ⇒ 0-Byte-Datei (NICHT gelöscht) → read_all/list_runs bleiben leer."""
    p = _fresh()
    _seed(p)
    S.delete_run("AAA", path=p)
    S.delete_run("BBB", path=p)
    # verbliebene Altzeile per ihrer synthetischen ID löschen
    legacy_id = next(x.run_id for x in S.read_all(p) if (x.run_id or "").startswith("legacy-"))
    S.delete_run(legacy_id, path=p)
    assert p.exists() and p.stat().st_size == 0
    assert S.read_all(p) == [] and S.list_runs(p) == []
    p.unlink()


def test_git_store_untouched():
    """Sicherheitsnetz: die Tests schreiben ausschließlich unter $SP — der Default-
    Pfad RESULTS_JSONL zeigt woanders hin (git-getrackt) und wird nie berührt."""
    assert S.RESULTS_JSONL != (_SP / "store_test.jsonl")
    assert "results" in str(S.RESULTS_JSONL)


# --- reine History-Komponenten-Logik (GPU-/Dash-frei) ------------------------
def test_history_options_and_label():
    runs = [{"run_id": "X", "run_name": "Lauf X", "created_at": "2026-07-12T11:00:00",
             "expr": "ik,kj->ij", "family": "contraction", "n": 3, "n_ok": 2}]
    opts = H.history_options(runs)
    assert opts[0]["value"] == "X"
    assert "Lauf X" in opts[0]["label"] and "2/3 ok" in opts[0]["label"] and "ik,kj->ij" in opts[0]["label"]


def test_history_runs_for_ids_filters_and_orders():
    """runs_for_ids filtert eine RunResult-Liste auf die ausgewählten run_ids in
    Auswahl-Reihenfolge (deterministisch)."""
    res = [RunResult(status="ok", config={}, run_id="A"),
           RunResult(status="ok", config={}, run_id="B"),
           RunResult(status="ok", config={}, run_id="A")]
    got = [r.run_id for r in H.runs_for_ids(res, ["B", "A"])]
    assert got == ["B", "A", "A"]
    assert H.runs_for_ids(res, []) == []


# --- Compile-Cache-Härtung (TZ 8-1): atomarer save_kernel + Korruptions-Erkennung -
def _kdir(name: str = "kernels_hardening") -> Path:
    """Frisches, leeres Kernel-Verzeichnis unter $SP (git-Store bleibt unberührt)."""
    d = _SP / name
    for f in (d.glob("*") if d.exists() else []):
        f.unlink()
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_save_kernel_atomic_and_idempotent():
    """save_kernel schreibt den Quelltext byte-genau und ist idempotent (gleicher
    Slug ⇒ gleicher Inhalt, folgenloses Überschreiben)."""
    d = _kdir()
    src = "# generierter Kernel\ndef launch(*a):\n    return None\n"
    p = S.save_kernel(src, "probe_slug", kernels_dir=d)
    assert p.exists() and p.read_text(encoding="utf-8") == src
    p2 = S.save_kernel(src, "probe_slug", kernels_dir=d)   # erneut → identisch
    assert p2 == p and p.read_text(encoding="utf-8") == src
    p.unlink(); d.rmdir()


def test_save_kernel_overwrites_corrupt():
    """Eine korrupte/halb geschriebene <slug>.py wird durch einen erneuten
    save_kernel-Aufruf vollständig ersetzt (Cache heilt sich)."""
    d = _kdir()
    path = S.kernel_file("probe_slug", d)
    path.write_bytes(b"\xff\xfe halb geschrieben \x00")   # kaputte, nicht dekodierbare Datei
    src = "def launch(*a):\n    return 42\n"
    S.save_kernel(src, "probe_slug", kernels_dir=d)
    assert path.read_text(encoding="utf-8") == src
    path.unlink(); d.rmdir()


def test_save_kernel_no_tmp_leftover():
    """Nach erfolgreichem Schreiben bleibt KEINE Temp-Datei (.kernel-*.tmp) liegen."""
    d = _kdir()
    S.save_kernel("def launch(*a):\n    return None\n", "probe_slug", kernels_dir=d)
    leftovers = [f.name for f in d.iterdir() if f.name.startswith(".kernel-")]
    assert leftovers == [], f"Temp-Reste gefunden: {leftovers}"
    for f in d.iterdir():
        f.unlink()
    d.rmdir()


def test_read_text_or_none_handles_missing_and_corrupt():
    """compile._read_text_or_none: gültiger Text zurück; fehlende/korrupte Datei → None
    (⇒ load_kernel schreibt neu statt zu crashen)."""
    d = _kdir()
    ok = d / "ok.py"
    ok.write_text("hallo", encoding="utf-8")
    assert C._read_text_or_none(ok) == "hallo"
    assert C._read_text_or_none(d / "fehlt.py") is None       # nicht vorhanden
    bad = d / "bad.py"
    bad.write_bytes(b"\xff\xfe\x00 kaputt")                    # nicht als utf-8 dekodierbar
    assert C._read_text_or_none(bad) is None
    ok.unlink(); bad.unlink(); d.rmdir()


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
